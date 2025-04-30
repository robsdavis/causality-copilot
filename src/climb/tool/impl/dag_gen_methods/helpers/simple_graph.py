import numpy as np

class _Node:
    def __init__(self, name: str):
        self._name = name
    def get_name(self) -> str:
        return self._name

class _SimpleGraph:
    def __init__(self, mat: np.ndarray, node_names: list[str]):
        # mat is 0/1 binary.  We need CPDAG code: 
        #   dir i→j  => A[i,j]=-1, A[j,i]=1
        #   undir i—j=> A[i,j]=A[j,i]=-1
        n = mat.shape[0]
        signed = np.zeros((n, n), dtype=int)

        # build signed from binary
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, j] and not mat[j, i]:
                    # directed i->j
                    signed[i, j] = -1
                    signed[j, i] = 1
                elif mat[j, i] and not mat[i, j]:
                    # directed j->i
                    signed[j, i] = -1
                    signed[i, j] = 1
                elif mat[i, j] and mat[j, i]:
                    # undirected
                    signed[i, j] = signed[j, i] = -1
                # else both zero => no edge

        self.graph = signed
        self.node_names = node_names

        # cpdag_to_json also wants .nodes of Node-objects with get_name()
        self.nodes = [ _Node(n) for n in node_names ]

    def get_node_names(self) -> list[str]:
        return self.node_names

    def get_num_nodes(self) -> int:
        return len(self.node_names)

    def is_directed_from_to(self, u, v) -> bool:
        i = self.node_names.index(u.get_name())
        j = self.node_names.index(v.get_name())
        return self.graph[i, j] == -1 and self.graph[j, i] == 1

    def is_undirected_from_to(self, u, v) -> bool:
        i = self.node_names.index(u.get_name())
        j = self.node_names.index(v.get_name())
        return self.graph[i, j] == -1 and self.graph[j, i] == -1

