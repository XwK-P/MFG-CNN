import torch
import torch.nn as nn


class MFGCNN(nn.Module):
    def __init__(self):
        super(MFGCNN, self).__init__()
        self.peNet = EnhancingNet(input_nc=4, output_nc=1, ngf=32, n_blocks=8)
        self.cUnet = ClassificationUnet(input_nc=1, output_nc=1, ngf=16)

    def forward(self, input):
        # if input is {US, BSE, LPT, LP}, then apply PE first.
    	if input.shape[1] == 4:
            input = self.peNet(input)
        seg = self.cUnet.forward(input)
        label = self.cUnet.forward_C(input)
        return input, seg, label


# Define our Pre-enhancing Net that features multiple conv layers
class EnhancingNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, n_blocks=8):
        super(EnhancingNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]

        for i in range(int(n_blocks/2)):
            model += [nn.Conv2d(ngf * (2**i), ngf * (2**(i+1)), kernel_size=4, stride=2, padding=1),
                    norm_layer(ngf* (2**(i+1))),
                    nn.ReLU(True),
                    nn.Dropout(0.1)
                    ]

        for i in range(int(n_blocks/2)-1,-1,-1):
            model += [nn.ConvTranspose2d(ngf * (2**(i+1)), ngf* (2**i), kernel_size=4, stride=2, padding=1),
                    norm_layer(ngf* (2**i)),
                    nn.ReLU(True),
                    nn.Dropout(0.1)
                    ]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define Unet blocks
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv2 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1)
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=3, stride=1, padding=1)
        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        downrelu2 = nn.ReLU(True)
        uprelu2 = nn.ReLU(True)
        upconv = nn.Conv2d(inner_nc * 2, inner_nc,
                                     kernel_size=3, stride=1,
                                     padding=1)

        downpool = nn.MaxPool2d(2)

        if outermost:
            upconv2 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0)
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(inner_nc)

            down = [downconv, downnorm, downrelu, downconv2, downpool]
            up = [uprelu, upconv, upnorm, uprelu2, upconv2, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv2 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0)
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

            down = [downrelu, downconv, downnorm, downrelu2, downconv2, downpool]
            up = [uprelu, upconv2, upnorm]
            model = down + up
        else:
            upconv2 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0)
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(inner_nc)
            upnorm2 = norm_layer(outer_nc)

            down = [downrelu, downconv, downnorm, downrelu2, downconv2]
            up = [uprelu, upconv, upnorm, uprelu2, upconv2, upnorm2]

            if use_dropout:
                model = down + [nn.Dropout(0.1)] + [downpool] + [submodule] + up
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Define our Classification U-net
class ClassificationUnet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=16,
                 norm_layer=nn.BatchNorm2d, use_dropout=True):
        super(ClassificationUnet, self).__init__()

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure num_downs=4
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16,
                                              norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer,
                                              use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer,
                                              use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

        self.classifier = nn.Linear(256, 4)
    def forward(self, input):
        return self.model(input)

    def forward_C(self, input):
        x = self.model.model[0](input)
        x = self.model.model[1](x)
        x = self.model.model[2](x)
        x = self.model.model[3](x)
        x = self.model.model[4](x)
        for i in range(7):
            x = self.model.model[5].model[i](x)
        for i in range(7):
            x = self.model.model[5].model[7].model[i](x)
        for i in range(7):
            x = self.model.model[5].model[7].model[7].model[i](x)
        for i in range(6):
            x = self.model.model[5].model[7].model[7].model[7].model[i](x)
        x = x[:, 0, :,:].contiguous().view(-1, 256)
        return self.classifier(x)

