import soundfile as sf
import numpy as np

# Load the generated and ground truth RIR audio files
generated_rir, _ = sf.read('generated_rir_batch0.wav')
ground_truth_rir, _ = sf.read('ground_truth_rir_batch0.wav')




# Ensure the audio signals are of the same length
min_length = min(len(generated_rir), len(ground_truth_rir))
generated_rir = generated_rir[:min_length]
ground_truth_rir = ground_truth_rir[:min_length]

# Calculate Mean Squared Error (MSE) between the signals
mse = np.mean((generated_rir - ground_truth_rir) ** 2)
print(f"Mean Squared Error (MSE) between generated and ground truth RIR: {mse}")

