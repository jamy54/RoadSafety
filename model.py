import torch.nn as nn

class SafetyNet(nn.Module):
    def __init__(self):
        super(SafetyNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(50, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(4, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(50, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Linear(in_features=700, out_features=40)

        self.fc2 = nn.Linear(in_features=40, out_features=6)


    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = x.view(-1, 700)
        fc1_output = self.fc1(x)
        return self.fc2(fc1_output)