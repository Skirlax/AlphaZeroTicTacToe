import atexit
import glob
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss

from General.network import GeneralNetwork
from mem_buffer import MemBuffer


class TicTacToeNet(nn.Module, GeneralNetwork):
    def __init__(self, in_channels, num_channels, dropout, action_size):
        super(TicTacToeNet, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.dropout = dropout
        self.action_size = action_size

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
        # 4608 (5x5) or 512 (3x3) or 32768 (10x10) or 18432 (8x8) # or 18432 for atari (6x6)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.pi = nn.Linear(512, action_size)  # probability head
        self.v = nn.Linear(512, 1)  # value head
        atexit.register(self.clear_traces)

    def forward(self, x, muzero=False):
        if not muzero:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.reshape(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)

        pi = F.log_softmax(self.pi(x), dim=1)
        v = F.tanh(self.v(x))

        return pi, v

    @th.no_grad()
    def predict(self, x, muzero=False):
        pi, v = self.forward(x, muzero=muzero)
        pi = th.exp(pi)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()

    def to_traced_script(self, board_size: int = 10):
        return th.jit.trace(self, th.rand(1, board_size, board_size).cuda())

    def trace(self, board_size: int) -> str:
        traced = self.to_traced_script(board_size=board_size)
        path = "traced.pt"
        traced.save(path)
        return path

    def clear_traces(self) -> None:
        from AlphaZero.utils import find_project_root
        for trace_file in glob.glob(f"{find_project_root()}/Checkpoints/Traces/*.pt"):
            os.remove(trace_file)

    def make_fresh_instance(self):
        return TicTacToeNet(self.in_channels, self.num_channels, self.dropout, self.action_size)

    @staticmethod
    def make_from_args(args: dict):
        return TicTacToeNet(args["num_net_in_channels"], args["num_net_channels"], args["net_dropout"],
                            args["net_action_size"])

    def train_net(self, memory_buffer: MemBuffer, args: dict):
        from AlphaZero.utils import mask_invalid_actions_batch
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        losses = []
        optimizer = th.optim.Adam(self.parameters(), lr=args["lr"])
        memory_buffer.shuffle()
        for epoch in range(args["epochs"]):
            for experience_batch in memory_buffer(args["batch_size"]):
                if len(experience_batch) <= 1:
                    continue
                states, pi, v = zip(*experience_batch)
                states = th.tensor(np.array(states), dtype=th.float32, device=device)
                pi = th.tensor(np.array(pi), dtype=th.float32, device=device)
                v = th.tensor(v, dtype=th.float32, device=device).unsqueeze(1)
                pi_pred, v_pred = self.forward(states)
                masks = mask_invalid_actions_batch(states)
                loss = mse_loss(v_pred, v) + self.pi_loss(pi_pred, pi, masks)
                losses.append(loss.item())
                # self.summary_writer.add_scalar("Loss", loss.item(), i * epochs + epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return sum(losses) / len(losses)

    def pi_loss(self, y_hat, y, masks):
        masks = masks.reshape(y_hat.shape).to(self.device)
        masked_y_hat = masks * y_hat
        return -th.sum(y * masked_y_hat) / y.size()[0]

    def to_shared_memory(self):
        for param in self.parameters():
            param.share_memory_()


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
