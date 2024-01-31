import os
import torchaudio
from torch.utils.data import DataLoader
import torch

def create_combined_dataset(root_dir_reverb, root_dir_rir, root_dir_clean):
    file_list_reverb = sorted([os.path.join(root_dir_reverb, file) for file in os.listdir(root_dir_reverb) if file.endswith('.wav')])
    file_list_rir = sorted([os.path.join(root_dir_rir, file) for file in os.listdir(root_dir_rir) if file.endswith('.wav')])
    file_list_clean = sorted([os.path.join(root_dir_clean, file) for file in os.listdir(root_dir_clean) if file.endswith('.wav')])

    assert len(file_list_reverb) == len(file_list_rir)
    combined_dataset = [
        (
            torchaudio.load(file_list_reverb[i])[0],
            torchaudio.load(file_list_rir[i])[0],
            torchaudio.load(file_list_clean[i])[0]
        )
        for i in range(len(file_list_reverb))
    ]
    return combined_dataset


def load_network_stage():
    from model2 import Generator, Discriminator
    netG = Generator()
    netD = Discriminator()
    return netG, netD


reverb_folder = 'new_data/reverb'
rir_folder = 'new_data/processed'
clean_folder = 'new_data/clean'

combined_dataset = create_combined_dataset(reverb_folder, rir_folder, clean_folder)

print(len(combined_dataset))
batch_size = 32
dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

for i, (reverb_sample, rir_sample, clean_sample) in enumerate(dataloader):
    if i < 5:  # Print filenames for the first few samples

        netG, netD = load_network_stage()
        netG.load_state_dict(torch.load('generator_model.pth'))
        netD.load_state_dict(torch.load('discriminator_model.pth'))
        print("Existing Model Loaded")
        generated_rir = netG(reverb_sample.to('cpu'))

        print(clean_sample.shape)
        print(reverb_sample.shape)
        print(rir_sample.shape)
        print(generated_rir.shape)

        # Save original rir_sample
        for j in range(batch_size):
            torchaudio.save(f'rir_sample_{i * batch_size + j + 1}.wav', rir_sample[j], 16000)

        reverb_filenames = [f'reverb_sample_{i + 1}.wav' for i in range(batch_size)]
        rir_filenames = [f'rir_sample_{i + 1}.wav' for i in range(batch_size)]
        print("Reverb File Names:")
        print(reverb_filenames)
        print("RIR File Names:")
        print(rir_filenames)
        print()
    else:
        break


