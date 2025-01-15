# DeepMETTL3

![logo](img/figure_1.png)

## Table of content

- [**Description**](#description)

- [**Requirements**](#requirements)

- [**Installation**](#installation)

- [**Limitations**](#limitations)

- [**Examples**](#examples)

- [**Citation**](#citation)

- [**License**](#license) 


## Description

**DeepMETLL3 is deep learning based scoring function for METLL3 structure based virtual screening.** <br><br>

The User have to through the following steps:

**1. Retrieval of Molecules**
> The notebook is present in the Notebook directory. 

**2. Generation of DeepCoy decoys**
> The DeepCoy algorithm was used to generate decoys for each active molecule. 100 decoys were generated for each active and then 50 optimized decoys were used for each active. User can get the code for DeepCoys from ; https://github.com/AngelRuizMoreno/Jupyter_Dock

**3. Convert smiles to mol2**
> The generated smiles for decoys and actives should be converted to mol2 file

**4. Molecular docking**
> Molecular docking was carried out using smina 

**5. Genrate Voxel features**
> RdkitGridFeaturizer from deepchem was used to convert docked complexes into voxel features; https://deepchem.io/

**6. Train model**
> In this study 3DCNN with mulihead attention was used. 


**7. Predict**
> A user-friendly jupyternotebook is prepared for users to used for their molecules
## Requirements
> The required libraries are present in the requirments.yml file. 


Question about usage or troubleshooting? Please leave a comment here
