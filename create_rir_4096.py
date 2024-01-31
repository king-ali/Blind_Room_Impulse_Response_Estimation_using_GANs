import os
import numpy as np
import soundfile as sf

# Set the target length
target_length = 4096

# Path to the directory containing the OmniRIR dataset
omnirir_dir = "data/Omni"

# Output directory for processed RIRs
output_dir = "data/processed"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all the RIR files in the directory
rir_files = [file for file in os.listdir(omnirir_dir) if file.endswith(".wav")]

# Loop through each RIR file and process it
for rir_file in rir_files:
    # Load the original RIR
    rir_path = os.path.join(omnirir_dir, rir_file)
    rir, fs = sf.read(rir_path, dtype='float32')  # Adjust the data type as needed
    # Check the length of the RIR
    current_length = len(rir)
    print(current_length)
    if current_length < target_length:
        # If shorter, zero-pad to the target length
        zeros = np.zeros(target_length - current_length)
        processed_rir = np.concatenate([rir, zeros])
    else:
        processed_rir = rir[0:target_length]
    # Save the processed RIR using soundfile
    output_path = os.path.join(output_dir, f"processed_{rir_file}")
    sf.write(output_path, processed_rir, 16000)
