import random
from collections import deque
from itertools import chain

import torch as th
from torch.utils.data import Dataset, DataLoader


class MemDataset(Dataset):
    def __init__(self, mem_buffer):
        self.mem_buffer = list(mem_buffer)

    def __len__(self):
        return len(self.mem_buffer)

    def __getitem__(self, idx):
        return self.mem_buffer[idx]


class MemBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.temp_buffer = None

    def add(self, experience):
        if not isinstance(experience, tuple):
            raise ValueError("Experience must be a tuple")
        self.buffer.append(experience)

    def add_list(self, experience_list):
        for experience in experience_list:
            self.add(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def shuffle(self):
        random.shuffle(self.buffer)  # in-place shuffle

    def make_fresh_temp_buffer(self):
        self.temp_buffer = self.buffer.copy()

    def batch(self, batch_size):
        batched_buffer = []
        buffer_len = len(self.buffer)
        for idx in range(0, buffer_len, batch_size):
            batched_buffer.append(list(self.buffer)[idx:min(idx + batch_size, buffer_len)])

        return batched_buffer

    def __call__(self, batch_size) -> list:
        return self.batch(batch_size)

    def __len__(self):
        return len(self.buffer)

    def batch_with_priorities(self, batch_size, K, alpha=1):
        if len(self.temp_buffer) == 0:
            return [], []
        priorities = self.calculate_priorities(batch_size, alpha, K)
        batch = []
        pris = []
        to_remove = []
        for idx, priority in priorities.items():
            batch.append(self.temp_buffer[idx])
            pris.append(priority)
            to_remove.append(idx)
        for idx in sorted(to_remove, reverse=True):
            del self.temp_buffer[idx]

        yield batch, pris

    def calculate_priorities(self, batch_size, alpha, K):
        ps = [(abs(self.temp_buffer[i][2][0] - self.temp_buffer[i][2][1]), i) for i in range(len(self.temp_buffer))]
        ps = [(p[0] ** alpha, p[1]) for p in ps]
        ps.sort(reverse=True, key=lambda x: x[0])
        # choices = np.random.choice([x[1] for x in ps],size=(batch_size),replace=False,p=[p[0] for p in ps]).tolist()
        # ps = [x for x in choices if x in [p[1] for p in ps]]
        ps = [ps[i:i + K] for i in range(0,int(batch_size // K),K)]
        ps = list(chain(*ps))
        sum_p = sum([p[0] for p in ps])
        return {p[1]: p[0] / sum_p for p in ps}

    def to_dataloader(self, batch_size):
        return DataLoader(MemDataset(self.buffer), batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)


class MuZeroFrameBuffer:
    def __init__(self, frame_buffer_size, noop_action: int):
        self.max_size = frame_buffer_size
        self.noop_action = noop_action
        self.buffer = deque(maxlen=frame_buffer_size)

    def add_frame(self, frame, action):
        self.buffer.append((frame, action))

    def concat_frames(self):
        frames_with_actions = [th.cat((th.tensor(frame, dtype=th.float32),
                                       th.full((frame.shape[0], frame.shape[1], 1), action, dtype=th.float32)), dim=2)
                               for frame, action in self.buffer]
        # return th.tensor(np.array(frames_with_actions), dtype=th.float32)
        return th.cat(frames_with_actions, dim=2)

    def init_buffer(self, init_state):
        for _ in range(self.max_size):
            self.add_frame(init_state, self.noop_action)

    def __len__(self):
        return len(self.buffer)
