import torch
import torch.nn as nn

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )
        self.layer4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )
        self.fc = nn.Sequential(nn.Linear(4096, 1024),
                                nn.Dropout(0.5),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 5)
                                )

    def forward(self, x):
        x = self.layer1(x)
        # print(x.size())  # (64.8.128.128)
        x = self.layer2(x)  # [64, 16, 62, 62])
        # print(x.size())
        x = self.layer3(x)  # [64, 32, 31, 31])
        # print(x.size())
        x = self.layer4(x)  # ([64, 64, 13, 13])
        # print(x.size())
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return  x
