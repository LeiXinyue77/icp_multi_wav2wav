import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import onnx
import netron


class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()

        # Contraction path
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.drop5 = nn.Dropout(0.5)

        # Expansive path (反卷积替代上采样)
        self.up6 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up7 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2, padding=0)
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up8 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up9 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, padding=0)
        self.conv9 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Conv1d(64, 2, kernel_size=3, padding=1)
        self.final_conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.double()
        # Contraction path
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        # Expansive path
        up6 = self.up6(drop5)
        merge6 = torch.cat((drop4, up6), dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat((conv3, up7), dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat((conv2, up8), dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat((conv1, up9), dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)
        output = self.final_conv(conv10)
        return self.sigmoid(output)


# cpu版本测试
if __name__ == "__main__":
    model = UNet1D().double().to('cpu')
    MRI = torch.randn(1, 1, 512).double().to('cpu')
    print(MRI.dtype)
    predict = model(MRI)
    print(predict.shape)  # (bz, 1, 512)
    torchsummary.summary(model, input_size=(1, 512), batch_size=1, device='cpu')

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
