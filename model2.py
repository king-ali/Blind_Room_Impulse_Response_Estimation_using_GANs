import torch
import torch.nn as nn
import torch.optim as optim


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=512, kernel_size=8193, stride=256, padding=4096)
        self.LeakyRelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, padding=20)
        self.LeakyRelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.BatchNor2 = nn.BatchNorm1d(1024)

        self.upblock1 = self.upblock(1024, 512, 4)
        self.upblock2 = self.upblock(512, 256, 4)
        self.upblock3 = self.upblock(256, 128, 4)
        self.upblock4 = self.upblock(128, 64, 2)
        self.upblock5 = self.upblock(64, 64, 2)
        self.convtrans = nn.ConvTranspose1d(64, 1, kernel_size=41, stride=1, padding=20)
        self.tanh = nn.Tanh()

    def upblock(self, inchannel, outchannel, scalefactor):
        return nn.Sequential(
            nn.ConvTranspose1d(inchannel, outchannel, kernel_size=41, stride=scalefactor,
                               padding=(21 - (scalefactor // 2)), output_padding=1),
            nn.BatchNorm1d(outchannel),
            nn.PReLU()
        )

    def forward(self, x):
        random_noise = torch.randn_like(x)
        condition = torch.cat([random_noise, x], dim=1)
        x = condition
        # print("condition inside model shape", condition.shape)
        x = self.conv1(x)
        x = self.LeakyRelu1(x)
        x = self.conv2(x)
        x = self.LeakyRelu2(x)
        x = self.BatchNor2(x)

        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.upblock3(x)
        x = self.upblock4(x)
        x = self.upblock5(x)
        x = self.convtrans(x)
        x = self.tanh(x)

        return x


class D_GET_LOGITS(nn.Module): #not chnaged yet
    def __init__(self, ndf, nef, bcondition=True):
        def conv3x1(in_planes, out_planes, stride=1):
            kernel_length = 41
            return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_length, stride=stride,
                             padding=20, bias=False)
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        kernel_length =41
        if bcondition:
            self.convd1d =  nn.ConvTranspose1d(ndf*8,ndf //2,kernel_size=kernel_length,stride=1, padding=20)
            self.outlogits = nn.Sequential(
                conv3x1(in_planes=ndf //2 + nef, out_planes=ndf //2, stride=1 ),
                nn.BatchNorm1d(ndf //2 ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(ndf //2 , 1, kernel_size=16, stride=4),
                nn.Sigmoid()
                )
        else:
            self.convd1d =  nn.ConvTranspose1d(ndf*8,ndf //2,kernel_size=kernel_length,stride=1, padding=20)
            self.outlogits = nn.Sequential(
                nn.Conv1d(ndf // 2 , 1, kernel_size=16, stride=4),
                nn.Sigmoid())


    def forward(self, h_code, c_code=None):
        # conditioning output
        h_code = self.convd1d(h_code)
        print('h_code', h_code.shape)
        if self.bcondition and c_code is not None:
            print("mode c_code1 ",c_code.size())
            c_code = c_code.reshape(-1, self.ef_dim, 1)
            print("mode c_code2 ",c_code.size())

            c_code = c_code.repeat(1, 1, 16)
            # state size (ngf+egf) x 16
            print("mode c_code ",c_code.size())

            # c_code = c_code.expand(32, 512, 4)
            # h_code = h_code.reshape(-1, self.ef_dim, 1)
            h_code = h_code.repeat(1, 1, 16)
            print("mode h_code ", h_code.size())
            #
            h_c_code = torch.cat((h_code, c_code), 1)
            print('h_c_code', h_c_code.shape)
        else:
            h_c_code = h_code

        output = self.outlogits(h_code)

        return output.view(-1)
#
# # Discriminator
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.nef = 128
#         self.ndf = 64
#         self.layers = nn.Sequential(
#             nn.Conv1d(1, self.ndf, 3, 1, 1, bias=False),  # 16384 * ndf
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(self.ndf, self.ndf * 2, 16, 4, 6, bias=False),
#             nn.BatchNorm1d(self.ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),  # 4096 * ndf * 2
#             nn.Conv1d(self.ndf * 2, self.ndf * 4, 16, 4, 6, bias=False),
#             nn.BatchNorm1d(self.ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),  # 1024 * ndf * 4
#             nn.Conv1d(self.ndf * 4, self.ndf * 8, 16, 4, 6, bias=False),
#             nn.BatchNorm1d(self.ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),  # 256 * ndf * 8
#             nn.Conv1d(self.ndf * 8, self.ndf * 16, 16, 4, 6, bias=False),
#             nn.BatchNorm1d(self.ndf * 16),
#             nn.LeakyReLU(0.2, inplace=True),  # 64 * ndf * 16
#             nn.Conv1d(self.ndf * 16, self.ndf * 32, 16, 4, 6, bias=False),
#             nn.BatchNorm1d(self.ndf * 32),
#             nn.LeakyReLU(0.2, inplace=True),  # 16 * ndf * 32
#             self.conv3x1(self.ndf * 32, self.ndf * 16),
#             nn.BatchNorm1d(self.ndf * 16),
#             nn.LeakyReLU(0.2, inplace=True),  # 16 * ndf * 16
#             self.conv3x1(self.ndf * 16, self.ndf * 8),
#             nn.BatchNorm1d(self.ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True)  # 16 * ndf * 8
#         )
#
#     # convd1d = nn.ConvTranspose1d(ndf * 16, ndf // 2, kernel_size=kernel_length, stride=1, padding=20)
#     # https://github.com/anton-jeran/MULTI-AUDIODEC/blob/main/Single_Multi_AudioDec/models/vocoder/FASTRIR.py
#     # outlogits = nn.Sequential(nn.Conv1d(ndf // 2, 1, kernel_size=16, stride=4), nn.Sigmoid())
#
#
#     get_cond_logits = D_GET_LOGITS(128, 64, bcondition=True)
#     get_uncond_logits = D_GET_LOGITS(128, 64, bcondition=False)
#
#     def conv3x1(self, in_planes, out_planes, stride=1):
#         kernel_length = 41
#         return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_length, stride=stride,
#                          padding=20, bias=False)
#
#     def forward(self, x):
#         x = self.layers(x)
#         return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=41, stride=4, padding=20),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, kernel_size=41, stride=4, padding=20),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, kernel_size=41, stride=4, padding=20),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1),
            nn.Sigmoid()
        )


        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 16 + 512, 256),
            nn.LeakyReLU(0.2),  # Adding activation function
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond = None):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # print("x shape is", x.shape)
        if cond is not None:
            cond = cond.view(cond.size(0), -1)
            # print("cond shape", cond.shape)
            x = torch.cat((x, cond), dim=1)
            # print("shape is after", x.shape)
            x = self.classifier1(x)
        # x = x.view(x.size(0), -1)
        # print("cond shape is", cond.shape)
        else:
            x = self.classifier(x)
        return x





# Define the networks and optimizers
generator = Generator()
discriminator = Discriminator()
optimizer_G = optim.RMSprop(generator.parameters(), lr=0.0002)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.0002)


# Loss functions
# criterion_EDR =
# criterion_CGAN =
criterion_MSE = nn.MSELoss()
lambda_EDR = 0.1
lambda_CGAN = 0.1
lambda_MSE = 0.1

input = torch.randn(1, 1, 16384)

output = generator(input)
print(output.shape)
fake_output = discriminator(output.detach())
print(fake_output.shape)


num_epochs = 10
# for epoch in range(num_epochs):
    # print("in loop")

print("done")