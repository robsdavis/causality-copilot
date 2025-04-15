import os
import requests
import json
import xml.etree.ElementTree as ET
from functools import wraps
from typing import Any, List, Dict, Tuple, Union

from tenacity import retry, wait_fixed, stop_after_attempt

# Synchronous package for Google Scholar; we’ll use it directly.
from scholarly import scholarly, MaxTriesExceededException

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

# Simple in-memory caches for queries.
gs_cache: Dict[Any, List[Dict]] = {}
pub_cache: Dict[Any, List[Dict]] = {}

class QueryBuilder:
    """
    A dedicated class to build search queries.
    """
    def __init__(self, feature1: str, feature2: str):
        self.feature1 = feature1
        self.feature2 = feature2
        self.causal_synonyms = [
            "causes", "leads to", "results in", "brings about", "gives rise to",
            "induces", "generates", "triggers", "produces", "engenders",
            "prompts", "begets", "sparks", "sets off", "contributes to",
            "is responsible for", "underlies", "determines", "initiates", "instigates"
        ]
    
    def build_query(self, causal: bool = True) -> str:
        if causal:
            causal_part = " OR ".join(self.causal_synonyms)
            return f"({self.feature1}) AND ({causal_part}) AND ({self.feature2})"
        else:
            return f"({self.feature1}) AND ({self.feature2})"


def search_google_scholar(query: str, limit: int = 5) -> List[Dict]:
    """
    Synchronously searches Google Scholar using the scholarly package.
    Returns records with keys: title, authors, year, doi, abstract, Full Text, Journal.
    """
    cache_key = (query, limit)
    if cache_key in gs_cache:
        return gs_cache[cache_key]
    
    results = []
    try:
        search_results = scholarly.search_pubs(query)
    except MaxTriesExceededException:
        return results  # Return empty list if unable to fetch.
    
    count = 0
    for result in search_results:
        if count >= limit:
            break
        bib = result.get('bib', {})
        title = bib.get('title', 'No title')
        abstract = bib.get('abstract', 'No abstract available')
        authors = bib.get('author', 'Unknown')
        year = bib.get('pub_year', 'N/A')
        doi = bib.get('doi', 'No DOI')
        journal = bib.get('journal', 'No Journal')
        results.append({
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "abstract": abstract,
            "Full Text": "No full text available",
            "Journal": journal
        })
        count += 1

    gs_cache[cache_key] = results
    return results


@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def fetch_url(url: str, params: Dict) -> str:
    """
    Fetch a URL using requests with retry logic.
    """
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.text


def search_pubmed(query: str, limit: int = 5) -> List[Dict]:
    """
    Synchronously searches PubMed using the E-utilities.
    This function performs ESearch, ESummary, and EFetch calls.
    Returns records with keys: title, authors, year, doi, abstract, Full Text, Journal.
    """
    cache_key = (query, limit)
    if cache_key in pub_cache:
        return pub_cache[cache_key]
    
    # ESearch: Retrieve PubMed IDs.
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": limit,
        "retmode": "json"
    }
    esearch_text = fetch_url(esearch_url, esearch_params)
    esearch_data = json.loads(esearch_text)
    id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        pub_cache[cache_key] = []
        return []
    
    id_str = ",".join(id_list)
    
    # ESummary: Get basic paper details.
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    esummary_params = {
        "db": "pubmed",
        "id": id_str,
        "retmode": "json"
    }
    esummary_text = fetch_url(esummary_url, esummary_params)
    esummary_data = json.loads(esummary_text)
    
    # EFetch: Get abstracts and DOIs.
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "id": id_str,
        "retmode": "xml"
    }
    efetch_text = fetch_url(efetch_url, efetch_params)
    abstracts = {}
    doi_map = {}
    root = ET.fromstring(efetch_text)
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_elem = medline.find("Article") if medline is not None else None
        if article_elem is not None:
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            # Extract abstract.
            abstract_elem = article_elem.find("Abstract")
            if abstract_elem is not None:
                paragraphs = [elem.text for elem in abstract_elem.findall("AbstractText") if elem.text]
                abstract_text = "\n".join(paragraphs) if paragraphs else "No abstract available"
            else:
                abstract_text = "No abstract available"
            abstracts[pmid] = abstract_text

            # Extract DOI from ArticleIdList.
            doi = "No DOI"
            article_id_list = article_elem.find("ArticleIdList")
            if article_id_list is not None:
                for article_id in article_id_list.findall("ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break
            doi_map[pmid] = doi
    
    results = []
    for uid in id_list:
        item = esummary_data.get("result", {}).get(uid, {})
        title = item.get("title", "No title")
        authors_list = item.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list)
        pub_date = item.get("pubdate", "N/A")
        journal = item.get("fulljournalname", "No Journal")
        abstract = abstracts.get(uid, "No abstract available")
        doi = doi_map.get(uid, "No DOI")
        results.append({
            "title": title,
            "authors": authors,
            "year": pub_date,
            "doi": doi,
            "abstract": abstract,
            "Full Text": "No full text available",
            "Journal": journal
        })
    
    pub_cache[cache_key] = results
    return results


def unified_search(search_func, query_builder: QueryBuilder, limit: int) -> List[Dict]:
    """
    Executes a search with a causal query first, then falls back to a non-causal query if needed.
    """
    query_causal = query_builder.build_query(causal=True)
    results = search_func(query_causal, limit)
    if not results:
        query_non_causal = query_builder.build_query(causal=False)
        results = search_func(query_non_causal, limit)
    return results


def retrieve_scientific_literature_for_edge(edge: Dict[str, str], source: str, limit: int) -> Tuple[str, Any]:
    """
    Synchronously retrieves literature for a single edge.
    Returns a tuple of the edge identifier (feature1_feature2) and its literature.
    """
    query_builder = QueryBuilder(edge["feature1"], edge["feature2"])
    edge_key = f"{edge['feature1']}_{edge['feature2']}"
    if source == "googlescholar":
        literature = unified_search(search_google_scholar, query_builder, limit)
    elif source == "pubmed":
        literature = unified_search(search_pubmed, query_builder, limit)
    elif source == "both":
        gs_results = unified_search(search_google_scholar, query_builder, limit)
        pub_results = unified_search(search_pubmed, query_builder, limit)
        literature = {"googlescholar": gs_results, "pubmed": pub_results}
    else:
        raise ValueError("Invalid source. Choose 'googlescholar', 'pubmed', or 'both'.")
    return edge_key, literature


def retrieve_scientific_literature(
    tc: ToolCommunicator,
    workspace: str,
    edges: List[Dict[str, str]],
    source: str = "pubmed",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Synchronously searches scientific literature for a list of edges from PubMed, Google Scholar, or both.
    For each edge (with keys "feature1" and "feature2"), the search is performed using a unified fallback mechanism.
    Results are saved to a JSON file.
    """
    final_results = {}
    for edge in edges:
        edge_key, literature = retrieve_scientific_literature_for_edge(edge, source, limit)
        final_results[edge_key] = literature

    results_path = os.path.join(workspace, "edges_literature.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    tc.set_returns(
        tool_return=(f"Literature saved to {results_path}"),
        user_report=(f"Literature saved to {results_path}")
    )
    return final_results


# --- Tool Class Definition ---

class RetrieveScientificLiterature(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        edges = kwargs["edges"]  # edges should be a list of dicts with keys "feature1" and "feature2"
        limit = kwargs.get("limit", 10)
        source = kwargs.get("source", "pubmed")
        thrd, out_stream = execute_tool(
            retrieve_scientific_literature,
            workspace=self.working_directory,
            edges=edges,
            source=source,
            limit=limit,
            # ---
            wd=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "retrieve_scientific_literature"

    @property
    def description(self) -> str:
        return (
            "Searches scientific literature for a list of edges, where each edge contains two features "
            "to assess their relationship. The search is performed using PubMed, Google Scholar, or both. "
            "It uses a unified fallback mechanism: if a query with causal synonyms returns no records, "
            "it automatically retries with a broader non-causal query."
        )

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "feature1": {"type": "string", "description": "The first feature of the edge."},
                                    "feature2": {"type": "string", "description": "The second feature of the edge."}
                                },
                                "required": ["feature1", "feature2"]
                            },
                            "description": "A list of edges to retrieve literature for."
                        },
                        "limit": {
                            "type": "string",
                            "description": "The number of results to return per edge. Default is 10."
                        },
                        "source": {
                            "type": "string",
                            "description": "Choose 'googlescholar', 'pubmed', or 'both'. Default is 'pubmed'."
                        },
                        "data_file_path": {"type": "string", "description": "Path to save the results."},
                    },
                    "required": [
                        "data_file_path",
                        "edges"
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "Searches scientific literature for a list of edges (each defined by two features) and returns "
            "the literature results for each edge. The search is performed synchronously using PubMed, Google Scholar, "
            "or both. A unified fallback mechanism is applied to broaden the query if needed."
        )
