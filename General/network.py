from abc import ABC, abstractmethod

from mem_buffer import MemBuffer


class GeneralNetwork(ABC):
    @abstractmethod
    def make_fresh_instance(self):
        """
        Returns a fresh instance of the network
        """
        pass

    @staticmethod
    @abstractmethod
    def make_from_args(args: dict):
        """
        Builds the network from the given arguments dict.
        """
        pass

    @abstractmethod
    def train_net(self, memory_buffer: MemBuffer, args: dict):
        """
        Trains the network for given number of epochs
        """
        pass

    @abstractmethod
    def to_shared_memory(self):
        pass
