import utils
import torch.nn as nn

class encoder(nn.Module):
    # initializers
    def __init__(self, in_nc, nf=32, img_size=64):
        super(encoder, self).__init__()
        self.input_nc = in_nc
        self.nf = nf
        self.img_size = img_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.independent_feature = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
        )
        self.specific_feature = nn.Sequential(
            nn.Linear((nf * 4) * (img_size // 8) * (img_size // 8), nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Linear(nf * 8, nf * 8),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.conv(input)
        i = self.independent_feature(x)
        f = x.view(-1, (self.nf * 4) * (self.img_size // 8) * (self.img_size // 8))
        s = self.specific_feature(f)
        s = s.unsqueeze(2)
        s = s.unsqueeze(3)

        return i, s


class decoder(nn.Module):
    # initializers
    def __init__(self, out_nc, nf=32):
        super(decoder, self).__init__()
        self.output_nc = out_nc
        self.nf = nf
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, out_nc, 4, 2, 1),
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.deconv(input)

        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32, img_size=64):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.img_size = img_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Sequential(
            nn.Linear((nf * 8) * (img_size // 16) * (img_size // 16), nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Linear(nf * 8, out_nc),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.conv(input)
        f = x.view(-1, (self.nf * 8) * (self.img_size // 16) * (self.img_size // 16))
        d = self.fc(f)
        d = d.unsqueeze(2)
        d = d.unsqueeze(3)

        return d