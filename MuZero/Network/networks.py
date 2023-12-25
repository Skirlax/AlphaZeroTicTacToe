import torch as th

from AlphaZero.Network.nnet import TicTacToeNet as PredictionDynamicsNet
from MuZero.utils import add_actions_to_obs


class MuZeroNet:
    def __init__(self, input_channels: int, dropout: float, action_size: int, num_channels: int, latent_size: int):
        self.representation_network = RepresentationNet(input_channels=input_channels)
        self.dynamics_network = PredictionDynamicsNet(in_channels=256, num_channels=num_channels, dropout=dropout,
                                                      action_size=action_size)
        # prediction outputs 6x6 latent state
        self.prediction_network = PredictionDynamicsNet(in_channels=256, num_channels=num_channels, dropout=dropout,
                                                        action_size=latent_size)

    def dynamics_forward(self, x: th.Tensor, action: int):
        # concat along 3rd dimension
        action = th.full((x.shape[0], 1, x.shape[2], x.shape[3]), action, dtype=th.float32, device=x.device)
        x = add_actions_to_obs(x, action)
        x = self.dynamics_network(x, action)
        return x

    def prediction_forward(self, x: th.Tensor, predict: bool = False):
        if predict:
            return self.prediction_network.predict(x)
        return self.prediction_network(x)

    def representation_forward(self, x: th.Tensor):
        x = self.representation_network(x)
        return x


class RepresentationNet(th.nn.Module):
    def __init__(self, input_channels: int):
        super(RepresentationNet, self).__init__()
        self.conv1 = th.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.residuals1 = th.nn.ModuleList([ResidualBlock(128) for _ in range(2)])
        self.conv2 = th.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.residuals2 = th.nn.ModuleList([ResidualBlock(256) for _ in range(3)])
        self.pool1 = th.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.residuals3 = th.nn.ModuleList([ResidualBlock(256) for _ in range(3)])
        self.pool2 = th.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = th.nn.ReLU()

    def forward(self, x: th.Tensor):
        x = self.relu(self.conv1(x))
        for residual in self.residuals1:
            x = residual(x)
        x = self.relu(self.conv2(x))
        for residual in self.residuals2:
            x = residual(x)
        x = self.pool1(x)
        for residual in self.residuals3:
            x = residual(x)
        x = self.pool2(x)
        return x


class ResidualBlock(th.nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.convolution1 = th.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bnorm1 = th.nn.BatchNorm2d(channels)
        self.convolution2 = th.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bnorm2 = th.nn.BatchNorm2d(channels)
        self.relu = th.nn.ReLU()

    def forward(self, x):
        x_res = x
        x = self.relu(self.bnorm1(self.convolution1(x)))
        x = self.bnorm2(self.convolution2(x))
        x += x_res
        return self.relu(x)
