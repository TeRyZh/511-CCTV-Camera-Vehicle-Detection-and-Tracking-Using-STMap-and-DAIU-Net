import torch
import torch.nn as nn




class ResidualStem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualStem, self).__init__()

        self.conv_block1 = nn.Sequential(
            
            nn.Conv2d(input_dim, int(output_dim/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(output_dim/2)),
            nn.ReLU(),
            nn.Conv2d(int(output_dim/2), int(output_dim/2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(output_dim/2))
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(int(output_dim/2), output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim)
            )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(int(output_dim/2), output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1) + self.conv_skip(x1)
        x3 = self.relu(x2)
        return x3



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class InceptionResNetA(nn.Module):

    #"""Channel depth goes from 64 to 128"""
    def __init__(self, input_channels):

        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.reduction1x1 = nn.Conv2d(128, 128, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
    
    
    def forward(self, x):

        residual = [
            self.branch1x1(x),
            self.branch3x3(x),
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)

        output = self.bn(shortcut + residual)
        output = self.relu(output)

        return output



class InceptionResNetB(nn.Module):

    #"""Channel depth goes from 128 to 256"""
    def __init__(self, input_channels):

        super().__init__()
        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 128, kernel_size=(5, 5), padding=(2, 2))
        )
        
        self.branch1x1 = BasicConv2d(input_channels, 128, kernel_size=1)

        self.reduction1x1 = nn.Conv2d(256, 256, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 256, kernel_size=1)

        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
    
    
    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch5x5(x)
        ]

        residual = torch.cat(residual, 1)

        #"""In general we picked some scaling factors between 0.1 and 0.3 to scale the residuals
        #before their being added to the accumulated layer activations (cf. Figure 20)."""
        residual = self.reduction1x1(residual) * 0.1

        shortcut = self.shortcut(x)

        output = self.bn(residual + shortcut)
        output = self.relu(output)

        return output



class InceptionResNetC(nn.Module):
    #"""Channel depth goes from 256 to 512"""

    def __init__(self, input_channels):

        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 128, kernel_size=1),
            BasicConv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        )

        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(512, 512, kernel_size=1)
        self.shorcut = nn.Conv2d(input_channels, 512, kernel_size=1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch3x3(x)
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1

        shorcut = self.shorcut(x)

        output = self.bn(shorcut + residual)
        output = self.relu(output)

        return output


class BridgeConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BridgeConv, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x
        

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out



class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size = 1)
    
    def forward(self, F):
        # first branch
        F1 = F.reshape((F.size(0), F.size(1), -1))      # (C, W*H)
        F1 = torch.transpose(F1, -2, -1)                # (W*H, C)
        # second branch
        F2 = F.reshape((F.size(0), F.size(1), -1))      # (C, W*H)
        F2 = nn.Softmax(dim = -1)(torch.matmul(F2, F1)) # (C, C)
        # third branch
        F3 = F.reshape((F.size(0), F.size(1), -1))      # (C, W*H)
        F3 = torch.matmul(F2, F3)                       # (C, W*H)
        F3 = F3.reshape(F.shape)                        # (C, W, H)
        return self.output_conv(F3*F)



class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride)  

    def forward(self, x):
        return self.upsample(x)


class DAIUnet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(DAIUnet, self).__init__()

        self.encoder1 = ResidualStem(input_dim = img_ch, output_dim = 64)

        self.encoder2 = InceptionResNetA(input_channels = 64)

        self.encoder3 = InceptionResNetB(input_channels = 128)

        self.encoder4 = InceptionResNetC(input_channels = 256)

        self.encoder5 = BridgeConv(in_channels = 512, out_channels = 1024)

        self.ChanAtt = ChannelAttentionModule(in_channels = 1024)

        self.decoder5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)

        self.decoder4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)

        self.decoder3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)

        self.decoder2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)

        self.decoder1=  nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_ch, kernel_size=3, stride=1, padding=1)
            )

        self.MaxPool = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.UpConv5 = BasicConv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.UpConv4 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.UpConv3 = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.UpConv2 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)



    def forward(self, x):
        
        # print("input: ", list(x.size()))
        # Encode
        e1 = self.encoder1(x)   # C*W*H 64*320*320
        # print("encoder 1", list(e1.size()))

        maxpool1 = self.MaxPool(e1)   # C*W*H 64*160*160
        e2 = self.encoder2(maxpool1)  # C*W*H 128*160*160

        maxpool2 = self.MaxPool(e2)   # C*W*H 128*80*80
        e3 = self.encoder3(maxpool2)  # C*W*H 256*80*80

        maxpool3 = self.MaxPool(e3)   # C*W*H 256*40*40
        e4 = self.encoder4(maxpool3)  # C*W*H 512*40*40
        
        maxpool4 = self.MaxPool(e4)   # C*W*H 512*20*20
        e5 = self.encoder5(maxpool4)  # C*W*H 1024*20*20
        # print("bridge before attention",list(e5.size()))

        # Decode
        att_x5 = self.ChanAtt(e5)    # C*W*H 1024*20*20
        # print("bridge after attention", list(att_x5.size()))
        d5 = self.decoder5(att_x5)   # C*W*H 512*40*40
        # print("after decoding 5",list(d5.size()))
        # print("after encoding 4",list(e4.size()))
        s4 = self.Att5(gate=d5, skip_connection=e4)   # C*W*H 512*40*40

        d5 = torch.cat((s4, d5), dim=1)  # C*W*H 1024*40*40
        d5 = self.UpConv5(d5)            # C*W*H 512*40*40

        d4 = self.decoder4(d5)           # C*W*H 256*80*80
        s3 = self.Att4(gate=d4, skip_connection=e3)   # C*W*H 256*80*80
        d4 = torch.cat((s3, d4), dim=1)  # C*W*H 512*80*80
        d4 = self.UpConv4(d4)            # C*W*H 256*80*80

        d3 = self.decoder3(d4)           # C*W*H 128*160*160
        s2 = self.Att3(gate=d3, skip_connection=e2)   # C*W*H 128*160*160
        d3 = torch.cat((s2, d3), dim=1)  # C*W*H 256*160*160
        d3 = self.UpConv3(d3)            # C*W*H 128*160*160

        d2 = self.decoder2(d3)           # C*W*H 64*320*320
        s1 = self.Att2(gate=d2, skip_connection=e1)  # C*W*H 64*320*320
        d2 = torch.cat((s1, d2), dim=1)  # C*W*H 128*320*320
        d2 = self.UpConv2(d2)            # C*W*H 64*320*320

        output = self.decoder1(d2)   # C*W*H output_ch*320*320

        return output
