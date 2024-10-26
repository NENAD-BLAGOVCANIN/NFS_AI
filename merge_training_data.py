import numpy as np
import os

# Define directory containing the .npy files and output file name
input_directory = "training_data"
output_filename = "combined_training_data.npy"

# Initialize an empty list to store data from each file
all_data = []

# Load each .npy file and append its content to the all_data list
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_directory, filename)
        data = np.load(file_path, allow_pickle=True)
        all_data.append(data)
        print(f"Loaded {filename} with shape {data.shape}")

# Concatenate all loaded arrays along the first axis
combined_data = np.concatenate(all_data, axis=0)

# Save the combined array to a new .npy file
np.save(output_filename, combined_data, allow_pickle=True)
print(f"Saved combined data to '{output_filename}' with shape {combined_data.shape}")
