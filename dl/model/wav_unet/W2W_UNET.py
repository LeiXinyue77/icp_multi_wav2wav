import torch
import torch.nn as nn
import torch.nn.init as init
import torchsummary
import onnx
import netron


class UNet(nn.Module):
    def __init__(self, ngf=8):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, ngf, kernel_size=3)
        self.enc2 = self.conv_block(ngf, ngf*2, kernel_size=3)
        self.enc3 = self.conv_block(ngf*2, ngf*4, kernel_size=3)
        self.enc4 = self.conv_block(ngf*4, ngf*8, kernel_size=3)
        self.center = self.conv_block(ngf*8, ngf*16, kernel_size=3)
        self.dec4 = self.conv_block(ngf*16, ngf*8, kernel_size=3)
        self.dec3 = self.conv_block(ngf*8, ngf*4, kernel_size=3)
        self.dec2 = self.conv_block(ngf*4, ngf*2, kernel_size=3)
        self.dec1 = self.conv_block(ngf*2, ngf, kernel_size=3)
        self.up_conv4 = nn.Conv1d(ngf*16, ngf*8, kernel_size=2, padding="same")
        self.up_conv3 = nn.Conv1d(ngf*8, ngf*4, kernel_size=2, padding="same")
        self.up_conv2 = nn.Conv1d(ngf*4, ngf*2, kernel_size=2, padding="same")
        self.up_conv1 = nn.Conv1d(ngf*2, ngf, kernel_size=2, padding="same")
        self.up_conv0 = nn.Conv1d(ngf, 2, kernel_size=3, padding="same")
        self.final = nn.Conv1d(2, 1, kernel_size=3, padding="same")

        # 初始化所有卷积层的权重为 He_Normal
        self.apply(self.init_weights)

    def conv_block(self, in_channels, out_channels, kernel_size):
        # padding = (kernel_size - 1) // 2  # 计算 padding 确保 'same' 填充
        conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(inplace=True)
        )
        return conv

    def init_weights(self, m):
        """ 对卷积层进行 He Normal 初始化 """
        if isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为 0

    def forward(self, x):
        _input = x.double()
        enc1 = self.enc1(_input)
        enc2 = self.enc2(nn.MaxPool1d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool1d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool1d(2)(enc3))

        drop4 = nn.Dropout(0.5)(enc4)
        center = self.center(nn.MaxPool1d(2)(drop4))
        drop_center = nn.Dropout(0.5)(center)

        # Decode with strict size matching
        up4 = self.up_conv4(nn.Upsample(scale_factor=2, mode='nearest')(drop_center))
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))

        up3 = self.up_conv3(nn.Upsample(scale_factor=2, mode='nearest')(dec4))
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))

        up2 = self.up_conv2(nn.Upsample(scale_factor=2, mode='nearest')(dec3))
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))

        up1 = self.up_conv1(nn.Upsample(scale_factor=2, mode='nearest')(dec2))
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        out = self.up_conv0(dec1)
        return torch.sigmoid(self.final(out))


# cpu版本测试
if __name__ == "__main__":
    model = UNet(ngf=8).double().to('cpu')
    MRI = torch.randn(32, 1, 1024).double().to('cpu')
    print(MRI.dtype)
    predict = model(MRI)
    print(predict.shape)  # (bz, 1, 512)
    torchsummary.summary(model, input_size=(1, 1024), batch_size=32, device='cpu')

    print("====================================== model summary finished !!!========================================")

    modelFile = 'demo.onnx'
    torch.onnx.export(model,
                      MRI,
                      modelFile,
                      opset_version=13,  # 使用版本 13
                      export_params=True,
                      )
    onnx_model = onnx.load(modelFile)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), modelFile)

    netron.start(modelFile)
