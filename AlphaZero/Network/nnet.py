import glob
import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import uuid


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
        # 4608 (5x5) or 512 (3x3) or 32768 (10x10) or 18432 (8x8)
        self.fc1 = nn.Linear(18432, 1024)
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

        pi = F.log_softmax(self.pi(x), dim=1)
        v = F.tanh(self.v(x))

        return pi, v

    @th.no_grad()
    def predict(self, x):
        pi, v = self.forward(x)
        pi = th.exp(pi)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()

    def to_traced_script(self, board_size: int = 10):
        return th.jit.trace(self, th.rand(1, board_size, board_size).cuda())

    def trace(self, board_size: int) -> str:
        from AlphaZero.utils import find_project_root
        traced = self.to_traced_script(board_size=board_size)
        path = f"{find_project_root()}/Checkpoints/Traces/traced{uuid.uuid4()}.pt"
        traced.save(path)
        return path

    def __del__(self):
        from AlphaZero.utils import find_project_root
        for trace_file in glob.glob(f"{find_project_root()}/Checkpoints/Traces/*.pt"):
            os.remove(trace_file)


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
