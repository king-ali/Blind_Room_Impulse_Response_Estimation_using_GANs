from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time

import numpy as np
# import torchfile
import pickle1

import soundfile as sf
import re
import math
# from wavefile import WaveWriter, Format

from config import cfg
from utils import mkdir_p
from utils import weights_init
from utils import save_RIR_results, save_model
from utils import KL_loss
from utils import compute_discriminator_loss, compute_generator_loss


# from torch.utils.tensorboard import summary
# from torch.utils.tensorboard import FileWriter


class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.model_dir_RT = os.path.join(output_dir, 'Model_RT')
            self.RIR_dir = os.path.join(output_dir, 'RIR')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.model_dir_RT)
            mkdir_p(self.RIR_dir)
            mkdir_p(self.log_dir)
            # self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        # torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stage(self):
        from model2 import Generator, Discriminator
        netG = Generator()
        netG.apply(weights_init)
        print(netG)
        netD = Discriminator()
        netD.apply(weights_init)
        print(netD)
        return netG, netD


    def train(self, data_loader, stage=1):

        netG, netD = self.load_network_stage()

        batch_size = self.batch_size

        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        # if cfg.CUDA:
        #
        #     real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        optimizerD = \
            optim.RMSprop(netD.parameters(),
                          lr=cfg.TRAIN.DISCRIMINATOR_LR)
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)

        optimizerG = optim.RMSprop(netG_para,
                                   lr=cfg.TRAIN.GENERATOR_LR)
        count = 0
        least_RT = 10
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.7  # 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.7  # 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                real_RIR_cpu, txt_embedding = data
                real_RIRs = Variable(real_RIR_cpu)
                txt_embedding = Variable(txt_embedding)
                if cfg.CUDA:
                    real_RIRs = real_RIRs.cuda()
                    txt_embedding = txt_embedding.cuda()
                # print("trianer RIRs ",real_RIRs.size())
                # print("trianer embedding ",txt_embedding.size())

                #######################################################
                # (2) Generate fake images
                ######################################################
                # noise.data.normal_(0, 1)
                # inputs = (txt_embedding, noise)
                inputs = (txt_embedding)
                # _, fake_RIRs, mu, logvar = \
                #     nn.parallel.data_parallel(netG, inputs, self.gpus)
                _, fake_RIRs, c_code = nn.parallel.data_parallel(netG, inputs, self.gpus)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_RIRs, fake_RIRs,
                                               real_labels, fake_labels,
                                               c_code, self.gpus)

                errD_total = errD * 5
                errD_total.backward()
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                # kl_loss = KL_loss(mu, logvar)
                netG.zero_grad()
                errG, MSE_error, RT_error = compute_generator_loss(epoch, netD, real_RIRs, fake_RIRs,
                                                                   real_labels, c_code, self.gpus)
                errG_total = errG * 5  # + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()
                for p in range(2):
                    inputs = (txt_embedding)
                    # _, fake_RIRs, mu, logvar = \
                    #     nn.parallel.data_parallel(netG, inputs, self.gpus)
                    _, fake_RIRs, c_code = nn.parallel.data_parallel(netG, inputs, self.gpus)
                    netG.zero_grad()
                    errG, MSE_error, RT_error = compute_generator_loss(epoch, netD, real_RIRs, fake_RIRs, real_labels, c_code, 'cpu')
                    # kl_loss = KL_loss(mu, logvar)
                    errG_total = errG * 5  # + kl_loss * cfg.TRAIN.COEFF.KL
                    errG_total.backward()
                    optimizerG.step()

                count = count + 1
                if i % 100 == 0:


                    # save the image result for each epoch
                    inputs = (txt_embedding)
                    lr_fake, fake, _ = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)
                    if (epoch % self.snapshot_interval == 0):
                        save_RIR_results(real_RIR_cpu, fake, epoch, self.RIR_dir)
                        if lr_fake is not None:
                            save_RIR_results(None, lr_fake, epoch, self.RIR_dir)
            end_t = time.time()

            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f   MSE_ERROR  %.4f RT_error %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.data, errG.data,
                     errD_real, errD_wrong, errD_fake, MSE_error * 4096, RT_error, (end_t - start_t)))

            store_to_file = "[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_real: {:.4f} Loss_wrong:{:.4f} Loss_fake {:.4f}  MSE Error:{:.4f} RT_error{:.4f} Total Time: {:.2f}sec".format(
                epoch, self.max_epoch, i, len(data_loader),
                errD.data, errG.data, errD_real, errD_wrong, errD_fake, MSE_error * 4096, RT_error, (end_t - start_t))
            store_to_file = store_to_file + "\n"
            with open("errors.txt", "a") as myfile:
                myfile.write(store_to_file)

            if (RT_error < least_RT):
                least_RT = RT_error
                save_model(netG, netD, epoch, self.model_dir_RT)
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        #
        save_model(netG, netD, self.max_epoch, self.model_dir)
        #
        # self.summary_writer.close()

    # def WaveWriter(self, file_path, stage=1):
    #     if stage == 1:
    #         netG, _ = self.load_network_stageI()
    #     else:
    #         netG, _ = self.load_network_stageII()
    #     netG.eval()
    #
    #     time_list = []
    #
    #     embedding_path = file_path
    #     with open(embedding_path, 'rb') as f:
    #         embeddings_pickle = pickle.load(f)
    #
    #     embeddings_list = []
    #     num_embeddings = len(embeddings_pickle)
    #     for b in range(num_embeddings):
    #         embeddings_list.append(embeddings_pickle[b])
    #
    #     embeddings = np.array(embeddings_list)
    #
    #     save_dir_GAN = "Generated_RIRs"
    #     mkdir_p(save_dir_GAN)
    #
    #     normalize_embedding = []
    #
    #     batch_size = np.minimum(num_embeddings, self.batch_size)
    #
    #     count = 0
    #     count_this = 0
    #     while count < num_embeddings:
    #
    #         iend = count + batch_size
    #         if iend > num_embeddings:
    #             iend = num_embeddings
    #             count = num_embeddings - batch_size
    #         embeddings_batch = embeddings[count:iend]
    #
    #         txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
    #         if cfg.CUDA:
    #             txt_embedding = txt_embedding.cuda()
    #
    #         #######################################################
    #         # (2) Generate fake images
    #         ######################################################
    #         start_t = time.time()
    #         inputs = (txt_embedding)
    #         _, fake_RIRs, c_code = \
    #             nn.parallel.data_parallel(netG, inputs, self.gpus)
    #         end_t = time.time()
    #         diff_t = end_t - start_t
    #         time_list.append(diff_t)
    #
    #         RIR_batch_size = batch_size  # int(batch_size/2)
    #         print("batch_size ", RIR_batch_size)
    #         channel_size = 64
    #
    #         for i in range(channel_size):
    #             fs = 16000
    #             wave_name = "RIR-" + str(count + i) + ".wav"
    #             save_name_GAN = '%s/%s' % (save_dir_GAN, wave_name)
    #             print("wave : ", save_name_GAN)
    #             res = {}
    #             res_buffer = []
    #             rate = 16000
    #             res['rate'] = rate
    #
    #             wave_GAN = fake_RIRs[i].data.cpu().numpy()
    #             wave_GAN = np.array(wave_GAN[0])
    #
    #             res_buffer.append(wave_GAN)
    #             res['samples'] = np.zeros((len(res_buffer), np.max([len(ps) for ps in res_buffer])))
    #             for i, c in enumerate(res_buffer):
    #                 res['samples'][i, :len(c)] = c
    #
    #             w = WaveWriter(save_name_GAN, channels=np.shape(res['samples'])[0], samplerate=int(res['rate']))
    #             w.write(np.array(res['samples']))
    #             w.close()
    #
    #         print("counter = ", count)
    #         count = count + 64
    #         count_this = count_this + 1


