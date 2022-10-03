import torch
from torch import nn
from torchsummary import summary
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Conv/ReLu/MaxPool then flattent / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(128 * 5 * 4, 10)
        self.linear = nn.Linear(128 * 9 * 12, 10)
        self.softmax = nn.Softmax(dim=1)
    #
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        # return x
        logits = self.linear(x)
        preditions = self.softmax(logits)
        return preditions


class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        num_classes = 10
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)
    #
    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = CNN().to(device)
    # summary(model, (1, 64, 44)) 
    summary(model, (1, 128, 169)) 
    model = DenseNet().to(device)
    summary(m, (3, 224, 224))  # image net
    summary(model, (3, 128, 250))  # article 
