# sodium_bp.csv
This repository provides the sodium_bp.csv file ready to use in the experiments.

## Description
Tabular clinical data capturing the relationship between sodium intake and blood pressure.
Each row represents an individual patient. Each column is a patient feature or biomedical observation. The columns are as  follows:
- **Sodium**: Categorical, the treatment variable indicating greater or less than average sodium intake.  
- **sbp_in_mmHg**: Float, systolic blood pressure in mmHg.  
- **Age_years**: Float, patient age in years.  
- **Proteinuria_in_mg**: Float, mg of protein in urine.
- **hypertension**: Categorical, presence/absence of hypertension.


## Provenance
Simulated based on r code published in “Educational Note: Paradoxical collider effect in the analysis of non-communicable disease epidemiological data: a reproducible illustration and web application” (Luque-Fernandez et al., 2018). Original publication available (here)[https://academic.oup.com/ije/article/48/2/640/5248195].

## Source
This dataset was generated using a method based on the R code available (here)[https://academic.oup.com/ije/article/48/2/640/5248195]. Data were generated with the script `data/causality/sodium_sbp/gen_dataset.ipynb` in this code release, which mirrors the R workflow in the original paper. See that script for full details on the simulation procedure.

## Transforms
In this dataset the the sodium column is simply transformed to a binary categorical from a float in the original raw data, in order to serve as the treatment variable.

## License
This dataset is distributed under the **CC-BY-4.0** license. You are free to share and adapt, provided you give appropriate credit.

# Papers used to orient edges
The json file (full_synth_epi_hypertension_literature.json) is an example set of papers that was used for the experiments. Websearch can be enabled instead of json upload of pre-determined papers by swapping out LIT_1 db episode in this file "Causality-Copilot/src/climb/engine/engine_openai_causality.py" (swap the episode labelled "# Retrieve papers from web search" in as the `"LIT_1"` episode).
