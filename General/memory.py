from abc import ABC, abstractmethod


class GeneralMemoryBuffer(ABC):
    def add(self, experience):
        """
        Add a single experience to the buffer.
        """
        pass

    def add_list(self, experience_list):
        """
        Add a list of experiences to the buffer.
        """
        pass

    def batch(self, batch_size):
        """
        Return a batch of experiences.
        """
        pass

    def __len__(self):
        """
        Return the length of the buffer.
        """
        pass
