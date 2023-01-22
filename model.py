import torch.nn as nn
import torch
from sru import SRU

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, sru_on=True):
    super(RNN, self).__init__()
    self.sru_on = sru_on
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    if sru_on:
      self.rnn = SRU(input_size, hidden_size, num_layers = 2)
    else:
      self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    if self.sru_on:
      out, _ = self.rnn(x)
    else:
      out, _ = self.rnn(x, hidden)
    out = out[-1, :] #  seq_len x hidden_size
    out = self.fc(out)
    return out