import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from deepchem.feat import RdkitGridFeaturizer
from torch.utils.data import DataLoader, TensorDataset
import tempfile


class MultiheadAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention3D, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Reshape (batch_size, channels, D, H, W) -> (batch_size, channels, D*H*W)
        batch_size, channels, D, H, W = x.shape
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)  # Shape: (D*H*W, batch_size, channels)

        # Apply Multihead Attention
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.norm(attn_output)

        # Reshape back to (batch_size, channels, D, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, D, H, W)
        return attn_output

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        # Input shape: (batch_size, 1539, 4, 4, 4)
        self.conv1 = nn.Conv3d(1536, 32, kernel_size=3, stride=1, padding=1)  # Output: (batch_size, 512, 4, 4, 4)
        self.bn1 = nn.BatchNorm3d(32)

        #self.attn1 = MultiheadAttention3D(embed_dim=32, num_heads=1)  # Apply Multihead Attention
        self.attn_layers = nn.ModuleList([MultiheadAttention3D(embed_dim=32, num_heads=8) for _ in range(6)])  # Apply Multihead Attention with 6 layers
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (batch_size, 256, 4, 4, 4)
        self.bn2 = nn.BatchNorm3d(64)

        self.attn2 = MultiheadAttention3D(embed_dim=64, num_heads=8)  # Apply Multihead Attention

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: (batch_size, 128, 4, 4, 4)
        self.bn3 = nn.BatchNorm3d(128)

        self.attn3 = MultiheadAttention3D(embed_dim=128, num_heads=8)  # Apply Multihead Attention

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)  # Output: (batch_size, 128, 2, 2, 2)
        self.dropout_conv = nn.Dropout3d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, 256)
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 1)  # Binary classification (e.g., active/inactive)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BatchNorm + ReLU
        #Apply 6 layers of Multihead Attention sequentially
        for attn_layer in self.attn_layers:
             x = attn_layer(x)
        #x = self.attn1(x)  # Multihead Attention after conv1

        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 + BatchNorm + ReLU
        #x = self.attn2(x)  # Multihead Attention after conv2

        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 + BatchNorm + ReLU
        #x = self.attn3(x)  # Multihead Attention after conv3

        x = self.dropout_conv(x)  # Dropout after convolutions
        x = self.pool(x)  # Pooling layer

        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout_fc1(x)  # Dropout after fc1
        x = F.relu(self.fc2(x))  # Fully connected layer 2
        x = self.dropout_fc2(x)  # Dropout after fc2
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        #x = self.fc3(x)  # No activation function for regression
        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepMETTL3: Predict binding for METTL3')
    parser.add_argument('-r', '--receptor', type=str, required=True, help='Path to receptor PDB file')
    parser.add_argument('-l', '--ligand', type=str, required=True, help='Path to ligand SDF file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV file for predictions')
    return parser.parse_args()

def featurize_ligand(receptor, ligand_sdf):
    featurizer = RdkitGridFeaturizer(box_width=24, voxel_width=6, feature_types=["splif"], ecfp_power=9, splif_power=9, flatten=False)
    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
    all_features = []
    
    for i, mol in enumerate(suppl):
        if mol is None:
            continue  # Skip invalid molecules
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as temp_sdf:
            writer = Chem.SDWriter(temp_sdf.name)
            writer.write(mol)
            writer.close()
        feature = featurizer._featurize((temp_sdf.name, receptor))
        features_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).permute(0, 4, 1, 2, 3)
        all_features.append(features_tensor)
    
    if len(all_features) == 0:
        raise ValueError(f"No valid conformations found in {ligand_sdf}")

    return all_features

def load_model(model_path):
    model = CNN3D()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_binding_affinity(model, features_list):
    predictions = []
    for features in features_list:
        dataset = TensorDataset(features)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in dataloader:
            with torch.no_grad():
                pred = model(batch[0]).item()
                predictions.append(pred)
    return predictions

def main():
    args = parse_arguments()
    receptor = args.receptor
    ligand_sdf = args.ligand
    output_csv = args.output
    
    print(f"Processing receptor: {receptor} and ligand: {ligand_sdf}")
    features_list = featurize_ligand(receptor, ligand_sdf)
    print("Loading trained model...")
    model = load_model('model_checkpoint_predict_100epochs.pth')
    print("Predicting binding affinity for all conformations...")
    all_predictions = predict_binding_affinity(model, features_list)
    
    conformation_names = [f"Conformation {i+1}" for i in range(len(all_predictions))]
    classifications = ["Active" if pred >= 0.5 else "Inactive" for pred in all_predictions]
    
    df = pd.DataFrame({
        'Conformation': conformation_names,
        'Probability': all_predictions,
        'Classification': classifications
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()

