from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchaudio
import torchvision.transforms as transforms
import pickle
import torch.nn.functional as F
import argparse
import os
import random
import numpy as np
import torch.optim as optim
from multiprocessing import Pool
import sys
import torch.nn as nn
import pprint
import datetime
import time
import dateutil
import dateutil.tz
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import weights_init
from utils import compute_discriminator_loss, compute_generator_loss
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
from dataset import TextDataset
from config import cfg, cfg_from_file
from utils import mkdir_p
from trainer import GANTrainer
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='r1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def load_audio_files(root_dir):
    fake_files = os.listdir(os.path.join(root_dir, 'reverberant_segments'))
    real_files = os.listdir(os.path.join(root_dir, 'rir_segments'))

    audio_data = []
    for fake_file, real_file in zip(fake_files, real_files):
        fake_path = os.path.join(root_dir, 'reverberant_segments', fake_file)
        real_path = os.path.join(root_dir, 'rir_segments', real_file)

        fake_waveform, _ = torchaudio.load(fake_path)
        real_waveform, _ = torchaudio.load(real_path)

        audio_data.append((fake_waveform, real_waveform))

    return audio_data

def load_network_stage():
        from model2 import Generator, Discriminator
        netG = Generator()
        netG.apply(weights_init)
        # print(netG)
        netD = Discriminator()
        netD.apply(weights_init)
        # print(netD)
        return netG, netD


def load_rir_wav(file_path, length=4096):
    rir, sr = torchaudio.load(file_path)
    if rir.size(0) > 1:
        rir = torch.mean(rir, dim=0, keepdim=True)
    rir /= torch.max(torch.abs(rir))
    if rir.size(1) < length:
        padding = length - rir.size(1)
        rir = torch.nn.functional.pad(rir, (0, padding), 'constant', 0)
    elif rir.size(1) > length:
        rir = rir[:, :length]
    return rir


def create_combined_dataset(root_dir_reverb, root_dir_rir):
    file_list_reverb = sorted([os.path.join(root_dir_reverb, file) for file in os.listdir(root_dir_reverb) if file.endswith('.wav')])
    file_list_rir = sorted([os.path.join(root_dir_rir, file) for file in os.listdir(root_dir_rir) if file.endswith('.wav')])
    assert len(file_list_reverb) == len(file_list_rir)
    combined_dataset = [
        (
            torchaudio.load(file_list_reverb[i])[0],
            load_rir_wav(file_list_rir[i])
            # torchaudio.load(file_list_rir[i])[0]
        )
        for i in range(len(file_list_reverb))
    ]
    return combined_dataset


def calculate_edr(estimated_rir, ground_truth_rir, freq_range=(16, 4000)):
    estimated_rir_tensor = estimated_rir
    ground_truth_rir_tensor = ground_truth_rir
    stft_estimated_rir = torch.abs(torch.fft.rfft(estimated_rir_tensor, dim=-1))
    stft_ground_truth_rir = torch.abs(torch.fft.rfft(ground_truth_rir_tensor, dim=-1))
    fs = 16000
    n_fft = stft_estimated_rir.size(-1)
    f = torch.fft.fftfreq(n_fft, d=1 / fs, dtype=torch.float)
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    stft_range_estimated = stft_estimated_rir[..., freq_mask]
    stft_range_ground_truth = stft_ground_truth_rir[..., freq_mask]
    edr_loss = torch.mean((stft_range_estimated - stft_range_ground_truth) ** 2)

    return edr_loss


if __name__ == "__main__":
    args = parse_args()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 ('data/segmented_audio/rir_segments', "stage1", timestamp)
    with open('example1.pickle', 'rb') as f:
        filenames = pickle.load(f)
    dataset = filenames

    reverb_folder = 'new_data/reverb'
    rir_folder = 'new_data/processed'
    combined_dataset = create_combined_dataset(reverb_folder, rir_folder)
    print(len(combined_dataset))
    batch_size = 32
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    netG, netD = load_network_stage()
    netG.to('cpu')
    netD.to('cpu')

    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    generator_lr = cfg.TRAIN.GENERATOR_LR
    discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
    lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
    count = 0
    least_RT = 10
    max_epoch = 5

    optimizerD = optim.RMSprop(netD.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR)
    netG_para = []
    for p in netG.parameters():
        if p.requires_grad:
            netG_para.append(p)
    optimizerG = optim.RMSprop(netG_para, lr=cfg.TRAIN.GENERATOR_LR)

    criterion = nn.MSELoss()
    # loss = nn.L1Loss()
    # criterion = nn.BCELoss()

    netG.load_state_dict(torch.load('generator_model.pth'))
    netD.load_state_dict(torch.load('discriminator_model.pth'))
    print("Existing Model Loaded")

    for epoch in range(max_epoch):
        for batch_idx, (reverb, rir) in enumerate(dataloader):
            netG.zero_grad()
            # print(f"Batch {batch_idx}: Data - {reverb.shape}, Targets - {rir.shape}")
            # Discriminator Training
            netD.zero_grad()
            real_labels = torch.ones(reverb.size(0), 1).to('cpu')  # Matching size for real_labels
            fake_labels = torch.zeros(reverb.size(0), 1).to('cpu')  # Matching size for fake_labels

            reverb_first_512 = reverb[:, :, :512]
            # Forward pass real batch through Discriminator
            output = netD(rir)
            # print("netd output", output.shape)
            errD_real = criterion(output, real_labels)
            errD_real.backward()


            # Forward pass fake batch through Generator
            # print('reverb shape', reverb.shape)
            # print("rir shape", reverb_first_512)
            fake = netG(reverb)
            output = netD(fake.detach())  # Detach to avoid training Generator on these labels
            # print("netd output", output.shape)
            errD_fake = criterion(output, fake_labels)

            errD_fake.backward()
            # optimizerD.step()


            reald = netD(rir, reverb_first_512)
            faked = netG(reverb)
            # print("faked", faked.shape)
            faked = netD(faked.detach(), reverb_first_512)
            reald = torch.mean(torch.log(reald))
            # reald = torch.mean((1 - reald) ** 2)
            faked = torch.mean(torch.log(1 - faked))
            # faked = torch.mean(faked ** 2)
            d_loss = -(reald + faked)  # -ve sign to maximize it
            # d_loss = 0.5 * (reald + faked)
            print("d_loss", d_loss)
            # errD_fake_real = errD_fake + errD_real + d_loss
            # print("errD_fake_real", errD_fake_real)
            # errD_fake_real.backward()
            d_loss.backward()
            optimizerD.step()


            # Generator Training
            err_cgan = 0
            err_cgan = netG(reverb)

            err_cgan = netD(err_cgan, reverb_first_512)
            # print(1 - err_cgan)
            err_cgan = torch.mean(torch.log(1 - err_cgan))

            # print(rir.shape)
            print("err_cgan", err_cgan)
            output = netD(fake)
            errG_edr2 = calculate_edr(fake, rir)
            netG.zero_grad()
            # errG_edr2.backward()
            # optimizerG.step()
            # for i in range(rir.size(0)):
                # errG_edr1 = calculate_edr(fake.detach().cpu().numpy())
                # rir[i, 0, :].detach().cpu().numpy()
            # errG = criterion(output, real_labels)  # Generator wants the output to be "real"
            errG_mean = criterion(fake, rir)
            print("err_G mean", errG_mean)
            errG =  errG_edr2

            # errG = (0.3 * errG_edr2) + err_cgan + (0.1*errG_mean)
            # errG = (0.2 * errG_mean) + (errG_edr2) + (0.3 * err_cgan)
            errG.backward()
            optimizerG.step()  # Update Generator weights

            # Print and log losses
            print("edr loss", errG_edr2)
            # print(f"Epoch [{epoch}/{max_epoch}] Batch [{batch_idx}/{len(dataloader)}]"
            #       f"Loss_G: {errG.item()}")
            print(f"Epoch [{epoch}/{max_epoch}] Batch [{batch_idx}/{len(dataloader)}]"
                  f"Loss_D_Real: {errD_real.item()}, Loss_D_Fake: {errD_fake.item()}, Loss_G: {errG.item()}")

    generator_save_path = 'generator_model.pth'
    discriminator_save_path = 'discriminator_model.pth'
    # Save the Generator model
    torch.save(netG.state_dict(), generator_save_path)
    # Save the Discriminator model
    torch.save(netD.state_dict(), discriminator_save_path)
    # writer.flush()
    # writer.close()


    for batch_idx, (reverb, rir) in enumerate(dataloader):
        generated_rir = netG(reverb.to('cpu'))
        print(generated_rir.shape)
        generated_rir_np = generated_rir.squeeze().detach().numpy()

        output_audio = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(torch.tensor(generated_rir_np))
        generated_output_file_path = f'generated_rir_batch{batch_idx}.wav'
        torchaudio.save(generated_output_file_path, output_audio.squeeze(), sample_rate=16000)

        # Convert ground truth RIR to a WAV file
        ground_truth_rir_np = rir.squeeze().detach().numpy()
        ground_truth_output_audio = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(
            torch.tensor(ground_truth_rir_np))
        ground_truth_output_file_path = f'ground_truth_rir_batch{batch_idx}.wav'
        torchaudio.save(ground_truth_output_file_path, ground_truth_output_audio.squeeze(), sample_rate=16000)

        break

