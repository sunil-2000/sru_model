import torch
from torch import nn
import numpy as np

# x_1 = 1; x_2 = 2; x_3 = 3...
# input data

class Data:
  def __init__(self, B=10) -> None:
    """
    Generate dataset for rnn
    """
    X = [f'x_{i}' for i in range(B)]
    Y = torch.from_numpy(np.array([i for i in range(B)], dtype=np.float32))
    unique_chars = set(''.join(X))
    self.embed_dim = len(unique_chars)
    self.char_to_int = {v:k for k, v in dict(enumerate(unique_chars)).items()}
    
    split = 6
    self.classes = B
    self.N = split

    self.Y = torch.zeros(B,B)
    for i in range(B):
      self.Y[i][i] = 1
    
    # self.x_tr, self.x_te = X[::2], X[1:len(X):2]
    # self.y_tr, self.y_te = self.Y[::2, :], self.Y[1:Y.shape[0]:2, :]    
    self.x_tr, self.x_te = X[:split], X[split:]
    self.y_tr, self.y_te = self.Y[:split], self.Y[split:]

  def to_tensor(self, raw_x):
    out = torch.zeros(len(raw_x), 1, self.embed_dim)
    for i, c in enumerate(raw_x):
      out[i][0][self.char_to_int[c]] = 1
    return out


# # sanity checks
# data = Data()
# print(data.x_tr)
# print(data.y_tr)
# print(data.x_te)
# print(data.y_te)
# print('x_1', data.to_tensor("x_1"))