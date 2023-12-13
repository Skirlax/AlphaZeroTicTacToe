import pickle
from AlphaZero import MemBuffer


from AlphaZero.utils import cpp_data_to_memory
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

memory = MemBuffer(1000)
cpp_data_to_memory(history, memory,10)