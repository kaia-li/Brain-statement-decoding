import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nifti_dir = "D:/HCP/tf"  # Directory path where NIfTI files are located

# Get a list of NIfTI files in the directory
nifti_files = [os.path.join(nifti_dir, file) for file in os.listdir(nifti_dir) if file.endswith(".nii")]

# Load and preprocess the data
data = []
for file in nifti_files:
    img = nib.load(file)
    data.append(img.get_fdata().reshape(-1))

data = np.array(data)  # Convert to numpy array
data = np.nan_to_num(data)  # Replace any NaN values with zeros

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform group-level PCA
pca = PCA()
pca.fit(data_scaled)

# Determine the number of principal components to retain
variance_explained = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(variance_explained >= 0.95) + 1  # Adjust the threshold as needed

# Project data onto the selected principal components
data_transformed = pca.transform(data_scaled)[:, :n_components]

# Save the transformed data as a CSV file
df = pd.DataFrame(data_transformed)
df.to_csv("group_pca_results.csv", index=False)
