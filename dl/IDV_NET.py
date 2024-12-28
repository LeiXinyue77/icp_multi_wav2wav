import numpy as np
import onnx
from torchsummary import torchsummary
import netron
from Blocks import *
import math
from utils import *
import torch


def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(2)
    diff[0] = org_shape[2] - finalShape[2]
    # diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(1, dtype=int)
    croppBorders[0] = int(diff[0] / 2)

    return tensorToCrop[:,
           :,
           croppBorders[0]:org_shape[2] - croppBorders[0]]


class Conv_residual_conv_Inception_Dilation(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_2_2 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2_3 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv_2_4 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_2_5 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv_2_output = conv_block(self.out_dim * 5, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)
        conv_2_5 = self.conv_2_5(conv_1)

        out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3


class Conv_residual_conv_Inception_Dilation_asymmetric(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation_asymmetric, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1,
                                                  padding=0, dilation=1)
        self.conv_2_2 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=1, dilation=1)
        self.conv_2_3 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1,
                                                  padding=2, dilation=1)
        self.conv_2_4 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=2, dilation=2)
        self.conv_2_5 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=4, dilation=4)

        self.conv_2_output = conv_block(self.out_dim * 5, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)
        conv_2_5 = self.conv_2_5(conv_1)

        out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3


class IVD_Net_asym(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super(IVD_Net_asym, self).__init__()
        print('~' * 55)
        print(' ----- Creating FUSION_NET HD (Assymetric) network...')
        print('~' * 55)

        self.in_dim = input_nc  # 1
        self.out_dim = ngf  # 32
        self.final_out_dim = output_nc  # 2

        # act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn = nn.ReLU()

        act_fn_2 = nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)
        self.down_1_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)  #
        self.pool_1_0 = maxpool()
        self.down_2_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_0 = maxpool()
        self.down_3_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_0 = maxpool()
        self.down_4_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2)
        self.down_1_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_1 = maxpool()
        self.down_2_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_1 = maxpool()
        self.down_3_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_1 = maxpool()
        self.down_4_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3)
        self.down_1_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_2 = maxpool()
        self.down_2_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_2 = maxpool()
        self.down_3_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_2 = maxpool()
        self.down_4_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4)
        self.down_1_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_3 = maxpool()
        self.down_2_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_3 = maxpool()
        self.down_3_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_3 = maxpool()
        self.down_4_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_3 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 60, self.out_dim * 16, act_fn)

        # ~~~ Decoding Path ~~~~~~ #
        self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv_Inception_Dilation(self.out_dim * 8, self.out_dim * 8, act_fn_2)

        self.deconv_2 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv_Inception_Dilation(self.out_dim * 4, self.out_dim * 4, act_fn_2)

        self.deconv_3 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv_Inception_Dilation(self.out_dim * 2, self.out_dim * 2, act_fn_2)

        self.deconv_4 = conv_decod_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv_Inception_Dilation(self.out_dim, self.out_dim, act_fn_2)

        self.out = nn.Conv1d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        # Params initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_uniform(m.weight.data)
                # init.xavier_uniform(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #

        i0 = input[:, 0:1, :].double()  # bz * 1  * height * width   #(1,1,64,64)
        i1 = input[:, 1:2, :].double()  # (1,1,64,64)
        i2 = input[:, 2:3, :].double()  # (1,1,64,64)
        i3 = input[:, 3:4, :].double()  # (1,1,64,64)

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)    # (1,32,64,64)
        down_1_1 = self.down_1_1(i1)    # (1,32,64,64)
        down_1_2 = self.down_1_2(i2)    # (1,32,64,64)
        down_1_3 = self.down_1_3(i3)    # (1,32,64,64)

        # -----  Second Level --------
        # input_2nd = torch.cat((down_1_0,down_1_1,down_1_2,down_1_3),dim=1)
        # (1,128,32,32)
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0),    # (1,32,32,32)
                                 self.pool_1_1(down_1_1),    # (1,32,32,32)
                                 self.pool_1_2(down_1_2),    # (1,32,32,32)
                                 self.pool_1_3(down_1_3)), dim=1)     # (1,32,32,32)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1),  # (1,32,32,32)
                                 self.pool_1_2(down_1_2),  # (1,32,32,32)
                                 self.pool_1_3(down_1_3),  # (1,32,32,32)
                                 self.pool_1_0(down_1_0)), dim=1)  # (1,32,32,32)

        input_2nd_2 = torch.cat((self.pool_1_2(down_1_2),  # (1,32,32,32)
                                 self.pool_1_3(down_1_3),   # (1,32,32,32)
                                 self.pool_1_0(down_1_0),   # (1,32,32,32)
                                 self.pool_1_1(down_1_1)), dim=1)   # (1,32,32,32)

        input_2nd_3 = torch.cat((self.pool_1_3(down_1_3),  # (1,32,32,32)
                                 self.pool_1_0(down_1_0),  # (1,32,32,32)
                                 self.pool_1_1(down_1_1),  # (1,32,32,32)
                                 self.pool_1_2(down_1_2)), dim=1)  # (1,32,32,32)

        down_2_0 = self.down_2_0(input_2nd_0)    # (1,64,32,32)
        down_2_1 = self.down_2_1(input_2nd_1)    # (1,64,32,32)
        down_2_2 = self.down_2_2(input_2nd_2)    # (1,64,32,32)
        down_2_3 = self.down_2_3(input_2nd_3)    # (1,64,32,32)

        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)    # (1,64,16,16)
        down_2_1m = self.pool_2_0(down_2_1)    # (1,64,16,16)
        down_2_2m = self.pool_2_0(down_2_2)    # (1,64,16,16)
        down_2_3m = self.pool_2_0(down_2_3)    # (1,64,16,16)

        input_3rd_0 = torch.cat((down_2_0m, down_2_1m, down_2_2m, down_2_3m), dim=1)   #(1,256,16,16)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)  #(1,384,16,16)

        input_3rd_1 = torch.cat((down_2_1m, down_2_2m, down_2_3m, down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        input_3rd_2 = torch.cat((down_2_2m, down_2_3m, down_2_0m, down_2_1m), dim=1)
        input_3rd_2 = torch.cat((input_3rd_2, croppCenter(input_2nd_2, input_3rd_2.shape)), dim=1)

        input_3rd_3 = torch.cat((down_2_3m, down_2_0m, down_2_1m, down_2_2m), dim=1)
        input_3rd_3 = torch.cat((input_3rd_3, croppCenter(input_2nd_3, input_3rd_3.shape)), dim=1)

        down_3_0 = self.down_3_0(input_3rd_0)   #(1,128,16,16)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)
        down_3_3 = self.down_3_3(input_3rd_3)

        # -----  Fourth Level --------

        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)   #(1,128,8,8)
        down_3_1m = self.pool_3_0(down_3_1)
        down_3_2m = self.pool_3_0(down_3_2)
        down_3_3m = self.pool_3_0(down_3_3)

        input_4th_0 = torch.cat((down_3_0m, down_3_1m, down_3_2m, down_3_3m), dim=1)   #(1,512,8,8)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)   #(1,896,8,8)

        input_4th_1 = torch.cat((down_3_1m, down_3_2m, down_3_3m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        input_4th_2 = torch.cat((down_3_2m, down_3_3m, down_3_0m, down_3_1m), dim=1)
        input_4th_2 = torch.cat((input_4th_2, croppCenter(input_3rd_2, input_4th_2.shape)), dim=1)

        input_4th_3 = torch.cat((down_3_3m, down_3_0m, down_3_1m, down_3_2m), dim=1)
        input_4th_3 = torch.cat((input_4th_3, croppCenter(input_3rd_3, input_4th_3.shape)), dim=1)

        down_4_0 = self.down_4_0(input_4th_0)  #(1,256,8,8)
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)
        down_4_3 = self.down_4_3(input_4th_3)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)  #(1,256,4,4)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)
        down_4_3m = self.pool_4_0(down_4_3)

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m, down_4_3m), dim=1)   #(1,1024,4,4)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)  #(1,1920,4,4)
        bridge = self.bridge(inputBridge)    #(1,512,4,4)

        #
        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #
        deconv_1 = self.deconv_1(bridge)   #(1,256,8,8)
        skip_1 = (deconv_1 + down_4_0 + down_4_1 + down_4_2 + down_4_3) / 5    #(1,256,8,8)# Residual connection
        up_1 = self.up_1(skip_1)   #(1,256,8,8)
        deconv_2 = self.deconv_2(up_1)   #(1,128,16,16)
        skip_2 = (deconv_2 + down_3_0 + down_3_1 + down_3_2 + down_3_3) / 5  # Residual connection
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)   #(1,64,32,32)
        skip_3 = (deconv_3 + down_2_0 + down_2_1 + down_2_2 + down_2_3) / 5  # Residual connection
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)    #(1,32,64,64)
        skip_4 = (deconv_4 + down_1_0 + down_1_1 + down_1_2 + down_1_3) / 5  # Residual connection
        up_4 = self.up_4(skip_4)
        out = self.out(up_4)

        # Last output
        # return F.softmax(self.out(up_4))
        return out   #(1,1,64,64)


# cpu版本测试
if __name__ == "__main__":
    batch_size = 2
    num_classes = 1
    ngf = 32

    model = IVD_Net_asym(input_nc=1, output_nc=num_classes, ngf=ngf).double()
    # print("total parameter:" + str(netSize(model)))
    MRI = torch.randn(batch_size, 4, 512)  # bz*modal*W*H     (bz,4,64,64)=>(bz,modal,T,1)
    predict = model(MRI)
    print(predict.shape)  # (bz, 2, 64, 64)=>(bz,1,T,1)

    torchsummary.summary(model, input_size=(4, 512), batch_size=2, device='cpu')

    modelFile = 'demo.onnx'
    torch.onnx.export(model,
                      MRI,
                      modelFile,
                      opset_version=10,  # 使用版本 10
                      export_params=True,
                      )
    onnx_model = onnx.load(modelFile)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), modelFile)

    netron.start(modelFile)
