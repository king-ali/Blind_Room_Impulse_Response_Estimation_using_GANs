import torch
import torch.nn as nn


# Input Batch size of 1, 1 channel
input = torch.randn(1, 1, 16384)

# Encoder
conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=8193, stride=256, padding=4096)
LeakyRelu1 = nn.LeakyReLU(0.2, inplace=True)
conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, padding=20)
LeakyRelu2 = nn.LeakyReLU(0.2, inplace=True)
BatchNor2 = nn.BatchNorm1d(1024)

# Pass the input through Encoder
output = conv1(input)
output = LeakyRelu1(output)
output = conv2(output)
output = LeakyRelu2(output)
output = BatchNor2(output)

# Decoder
def upblock(inchannel, outchannel, scalefactor):
    return nn.Sequential(
        nn.ConvTranspose1d(inchannel, outchannel, kernel_size=41, stride=scalefactor, padding=(21-(scalefactor//2)), output_padding=1),
        nn.BatchNorm1d(outchannel),
        nn.PReLU()
    )
upblock1 = upblock(1024, 512, 4)
upblock2 = upblock(512, 256, 4)
upblock3 = upblock(256,128,4)
upblock4 = upblock(128, 64, 2)
upblock5 = upblock(64, 64, 2)
convtrans = nn.ConvTranspose1d(64, 1, kernel_size=41, stride=1, padding=20)
tanh = nn.Tanh()


# Pass the encoder input to decoder
output = upblock1(output)
output = upblock2(output)
output = upblock3(output)
output = upblock4(output)
output = upblock5(output)
output = convtrans(output)
output = tanh(output)

# Get the dimensions of the output tensor
output_shape = output.size()
print("Output shape:", output_shape)

def conv3x1(in_planes, out_planes, stride=1):
    kernel_length  = 41
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_length, stride=stride,
                     padding=20, bias=False)

def discri(input):
    nef =  128
    ndf =  64
    return nn.Sequential(
        nn.Conv1d(1, ndf, 3, 1, 1, bias=False),  # 16384 * ndf
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, ndf * 2, 16, 4, 6, bias=False),
        nn.BatchNorm1d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),  # 4096 * ndf * 2
        nn.Conv1d(ndf * 2, ndf * 4, 16, 4, 6, bias=False),
        nn.BatchNorm1d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),  # 1024 * ndf * 4
        nn.Conv1d(ndf * 4, ndf * 8, 16, 4, 6, bias=False),
        nn.BatchNorm1d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),  # 256 * ndf * 8
        nn.Conv1d(ndf * 8, ndf * 16, 16, 4, 6, bias=False),
        nn.BatchNorm1d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),  # 64 * ndf * 16
        nn.Conv1d(ndf * 16, ndf * 32, 16, 4, 6, bias=False),
        nn.BatchNorm1d(ndf * 32),
        nn.LeakyReLU(0.2, inplace=True),  # 16 * ndf * 32
        conv3x1(ndf * 32, ndf * 16),
        nn.BatchNorm1d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),  # 16 * ndf * 16
        conv3x1(ndf * 16, ndf * 8),
        nn.BatchNorm1d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)  # 16 * ndf * 8
    )(input)


RIR_embedding = discri(output)
print(RIR_embedding.shape)

