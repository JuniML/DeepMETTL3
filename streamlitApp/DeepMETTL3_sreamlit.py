import streamlit as st
import os
import pandas as pd
import torch
from deepchem.feat import RdkitGridFeaturizer
from rdkit import Chem
import tempfile
from DeepMETTL3 import load_model, featurize_ligand, predict_binding_affinity

# Set page title and layout
st.set_page_config(page_title="DeepMETTL3 Prediction", layout="centered")

# Add an image at the top
st.image("juni.png", caption="Protein-Ligand Binding", use_container_width=True)

# Header
st.title("DeepMETTL3: Binding Affinity Prediction")
st.write(
    "**METTL3** is an important enzyme in RNA modification. This app allows you to predict "
    "the binding affinity of ligands to METTL3 using a deep learning model."
)

# Upload files
st.sidebar.header("Upload Your Files")
receptor_file = st.sidebar.file_uploader("Upload Receptor (.pdb)", type=["pdb"])
ligand_file = st.sidebar.file_uploader("Upload Ligand (.sdf)", type=["sdf"])

if receptor_file and ligand_file:
    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as rec_tmp:
        rec_tmp.write(receptor_file.read())
        receptor_path = rec_tmp.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as lig_tmp:
        lig_tmp.write(ligand_file.read())
        ligand_path = lig_tmp.name
    
    st.sidebar.success("Files uploaded successfully!")
    
    # Run prediction
    if st.sidebar.button("Run Prediction"):
        st.subheader("Processing and Predicting...")
        model = load_model("model_checkpoint_predict_100epochs.pth")
        features_list = featurize_ligand(receptor_path, ligand_path)
        all_predictions = predict_binding_affinity(model, features_list)
        
        # Create DataFrame
        conformation_names = [f"Conformation {i+1}" for i in range(len(all_predictions))]
        classifications = ["Active" if pred >= 0.5 else "Inactive" for pred in all_predictions]
        df = pd.DataFrame({
            "Conformation": conformation_names,
            "Probability": all_predictions,
            "Classification": classifications
        })
        
        # Display results
        st.write("### Prediction Results")
        st.dataframe(df)
        
        # Allow download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
else:
    st.sidebar.warning("Please upload both receptor and ligand files to proceed.")

# Footer
st.markdown("---")
st.markdown("**Designed by Dr. Muhammad Junaid**  ")
st.markdown("Contact: [mjunaid@szu.edu.cn](mailto:mjunaid@szu.edu.cn)")

