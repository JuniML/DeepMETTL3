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

**3. Blind Docking**
> Do you want to dock multiple ligands into whole target surface and/or its pockets? This protocol demonstrates the entire process of pocket search and their use as potential organic molecule binding sites. **(Documentation in progress)**

**4. Reverse Docking / Target fishing)**
> Interested in docking one or a few molecules into a set of proteins to identify the most promising target(s)? This notebook covers all of the steps required to achieve such a goal in a condensed manner, making the process seem like a walk in the park. **(Documentation in progress)**

**5. Scaffold-based Docking**
> Do you want to use a molecular substructure as an anchor point for your ligands? This procedure demonstrates an approximation for running molecular docking while constraining the position of a portion of the ligand. This is especially useful for investigating novel ligands with similar structure to known binders. **(In construction)**

**6. Covalent Docking**
> Is your hypothesis that your ligand can bind to the target covalently? This protocol describes how to use freely available tools to investigate the covalent binding mode of ligands to protein targets. **(In construction)**


**7. Docking Analysis**
> Have you completed your docking experiments with Jupyter Dock or another approach and want to conduct a rational analysis? You've come to the right place. This notebook summarizes the most common docking analysis techniques, including score comparisons, z-score calculation between softwares, pose clustering, molecular interactions mapping, and more. **(In construction)**


Question about usage or troubleshooting? Please leave a comment here
