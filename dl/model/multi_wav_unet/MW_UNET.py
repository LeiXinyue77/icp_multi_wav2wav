import onnx
from torchsummary import torchsummary
import netron
from dl.model.Blocks import *
import math
import torch


class Conv_Down(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, dropout_prob=0.0):
        super(Conv_Down, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=1)
        self.conv_2 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=1)

        # 如果 dropout_prob > 0，添加 Dropout 层
        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None  # 默认为 None，不进行 dropout

    def forward(self, input):
        conv_1 = self.conv_1(input)
        out = self.conv_2(conv_1)

        if self.dropout:
            out = self.dropout(out)

        return out


class Conv_Up(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, dropout_prob=0.0):
        super(Conv_Up, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=1)
        self.conv_2 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=1)

        # 如果 dropout_prob > 0，添加 Dropout 层
        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None  # 默认为 None，不进行 dropout

    def forward(self, input):
        conv_1 = self.conv_1(input)
        out = self.conv_2(conv_1)

        if self.dropout:
            out = self.dropout(out)

        return out


class Multi_Wav_UNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Multi_Wav_UNet, self).__init__()

        self.in_dim = input_nc  # 1
        self.out_dim = ngf  # 32
        self.final_out_dim = output_nc  # 2

        # act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn = nn.ReLU()
        act_fn_2 = nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)
        self.down_1_0 = Conv_Down(self.in_dim, self.out_dim, act_fn)  #
        self.pool_1_0 = maxpool()
        self.down_2_0 = Conv_Down(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2_0 = maxpool()
        self.down_3_0 = Conv_Down(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3_0 = maxpool()
        self.down_4_0 = Conv_Down(self.out_dim * 4, self.out_dim * 8, act_fn, dropout_prob=0.5)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2)
        self.down_1_1 = Conv_Down(self.in_dim, self.out_dim, act_fn)
        self.pool_1_1 = maxpool()
        self.down_2_1 = Conv_Down(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2_1 = maxpool()
        self.down_3_1 = Conv_Down(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3_1 = maxpool()
        self.down_4_1 = Conv_Down(self.out_dim * 4, self.out_dim * 8, act_fn, dropout_prob=0.5)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3)
        self.down_1_2 = Conv_Down(self.in_dim, self.out_dim, act_fn)
        self.pool_1_2 = maxpool()
        self.down_2_2 = Conv_Down(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2_2 = maxpool()
        self.down_3_2 = Conv_Down(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3_2 = maxpool()
        self.down_4_2 = Conv_Down(self.out_dim * 4, self.out_dim * 8, act_fn, dropout_prob=0.5)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4)
        self.down_1_3 = Conv_Down(self.in_dim, self.out_dim, act_fn)
        self.pool_1_3 = maxpool()
        self.down_2_3 = Conv_Down(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2_3 = maxpool()
        self.down_3_3 = Conv_Down(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3_3 = maxpool()
        self.down_4_3 = Conv_Down(self.out_dim * 4, self.out_dim * 8, act_fn, dropout_prob=0.5)
        self.pool_4_3 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_Down(self.out_dim * 24, self.out_dim * 16, act_fn, dropout_prob=0.5)

        # ~~~ Decoding Path ~~~~~~ #
        self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_Up(self.out_dim * 24, self.out_dim * 8, act_fn_2)

        self.deconv_2 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_Up(self.out_dim * 16,  self.out_dim * 4, act_fn_2)

        self.deconv_3 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_Up(self.out_dim * 8, self.out_dim * 2, act_fn_2)

        self.deconv_4 = conv_decod_block(self.out_dim*2, self.out_dim, act_fn_2)
        self.up_4 = Conv_Up(self.out_dim * 4, self.out_dim, act_fn_2)

        self.out_1 = nn.Conv1d(self.out_dim, 2, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Conv1d(2, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # ~~~~~~ Encoding path ~~~~~~~  #
        i0 = input[:, 0:1, :].double()  # bz * 1  * width   #(n,1,1024)
        i1 = input[:, 1:2, :].double()  # (n,1,1024)
        i2 = input[:, 2:3, :].double()  # (n,1,1024)
        # i3 = input[:, 3:4, :].double()  # (2,1,1024)

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)  # (n,32,1024)
        down_1_1 = self.down_1_1(i1)  # (n,32,1024)
        down_1_2 = self.down_1_2(i2)  # (n,32,1024)
        # down_1_3 = self.down_1_3(i3)    # (n,32,1024)

        # -----  Second Level ----- ---
        input_2nd_0 = self.pool_1_0(down_1_0)  #
        input_2nd_1 = self.pool_1_0(down_1_0)  # (n,32,512)
        input_2nd_2 = self.pool_1_0(down_1_0)  # (n,32,512)

        down_2_0 = self.down_2_0(input_2nd_0)  # (n,64,512)
        down_2_1 = self.down_2_1(input_2nd_1)
        down_2_2 = self.down_2_2(input_2nd_2)

        # -----  Third Level --------
        # Max-pool
        input_3rd_0 = self.pool_2_0(down_2_0)  # (n,64,256)
        input_3rd_1 = self.pool_2_0(down_2_1)  # (n,64,256)
        input_3rd_2 = self.pool_2_0(down_2_2)  # (n,64,256)

        down_3_0 = self.down_3_0(input_3rd_0)  # (n,128,256)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)

        # -----  Fourth Level --------
        # Max-pool
        input_4th_0 = self.pool_3_0(down_3_0)  # (n,128,128)
        input_4th_1 = self.pool_3_0(down_3_1)
        input_4th_2 = self.pool_3_0(down_3_2)

        down_4_0 = self.down_4_0(input_4th_0)  # (n,256,128)
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)  # (n,256,64)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m), dim=1)  # (n,768,64)
        bridge = self.bridge(inputBridge)  # (n,256,64)

        # ~~~~~~ Decoding path ~~~~~~~  #
        deconv_1 = self.deconv_1(bridge)
        skip_1 = torch.cat((deconv_1, down_4_0, down_4_1 + down_4_2), dim=1)
        up_1 = self.up_1(skip_1)

        deconv_2 = self.deconv_2(up_1)
        skip_2 = torch.cat((deconv_2, down_3_0, down_3_1, down_3_2), dim=1)
        up_2 = self.up_2(skip_2)

        deconv_3 = self.deconv_3(up_2)
        skip_3 = torch.cat((deconv_3, down_2_0, down_2_1, down_2_2), dim=1)
        up_3 = self.up_3(skip_3)

        deconv_4 = self.deconv_4(up_3)
        skip_4 = torch.cat((deconv_4, down_1_0, down_1_1, down_1_2), dim=1)
        up_4 = self.up_4(skip_4)

        out = self.out_1(up_4)
        final_out = self.out_2(out)

        return self.sigmoid(final_out)


# cpu版本测试
if __name__ == "__main__":
    batch_size = 1
    num_classes = 1
    ngf = 64

    model = Multi_Wav_UNet(input_nc=1, output_nc=num_classes, ngf=ngf).double().to('cpu')
    # print("total parameter:" + str(netSize(model)))
    MRI = torch.randn(batch_size, 3, 1024).double().to('cpu')  # bz*modal*W*H     (bz,4,64,64)=>(bz,modal,T,1)
    predict = model(MRI)
    print(predict.shape)  # (bz, 2, 64, 64)=>(bz,1,T,1)

    torchsummary.summary(model, input_size=(3, 1024), batch_size=1, device='cpu')

    print("====================================== model summary finished !!!========================================")

    modelFile = 'demo.onnx'
    torch.onnx.export(model,
                      MRI,
                      modelFile,
                      opset_version=13,  # 使用版本 10
                      export_params=True,
                      )
    onnx_model = onnx.load(modelFile)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), modelFile)

    netron.start(modelFile)
