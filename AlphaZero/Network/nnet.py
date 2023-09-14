import torch.nn as nn
import torch.nn.functional as F


class TicTacToeNet(nn.Module):
    def __init__(self, in_channels, num_channels, dropout, action_size):
        super(TicTacToeNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3)
        self.bn4 = nn.BatchNorm2d(num_channels)

        # Fully connected layers
        # 4608 (5x5) or 512 (3x3)
        self.fc1 = nn.Linear(4608, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.pi = nn.Linear(512, action_size)  # probability head
        self.v = nn.Linear(512, 1)  # value head

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)

        pi = F.softmax(self.pi(x), dim=1)
        v = F.tanh(self.v(x))

        return pi, v

    def predict(self, x):
        pi, v = self.forward(x)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()


class TicTacToeNetNoNorm(nn.Module):
    def __init__(self, in_channels, num_channels, dropout, action_size):
        super(TicTacToeNetNoNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3)

        self.fc1 = nn.Linear(4608, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.dropout = nn.Dropout(dropout)

        self.pi = nn.Linear(512, action_size)  # probability head
        self.v = nn.Linear(512, 1)  # value head

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        pi = F.softmax(self.pi(x), dim=1)
        v = F.tanh(self.v(x))
        # print(th.any(th.isnan(pi)))
        # print(th.any(th.isnan(v)))

        return pi, v

    def predict(self, x):
        pi, v = self.forward(x)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()
