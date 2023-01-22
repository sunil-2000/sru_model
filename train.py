import torch
import torch.nn as nn
from clean import Data
from model import RNN


class TrainWrapper:
  # data / model
  def __init__(self, sru=True):
    hidden_size = 12
    hidden_layers = 2
    self.data = Data()
    self.model = RNN(self.data.embed_dim, hidden_size, hidden_layers, self.data.classes, sru)

    # hparams
    self.n_epochs = 1000
    self.criterion = nn.CrossEntropyLoss()
    lr = 0.001
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

  def train(self, print_steps=50):
    # training loop
    for epoch in range(self.n_epochs):
      # iterate through dataset
      for i in range(len(self.data.x_tr)):
        x = self.data.to_tensor(self.data.x_tr[i]) # tensor
        y = self.data.y_tr[i].view(1, self.data.y_tr.shape[-1])

        output = self.model(x)
        loss = self.criterion(output, y.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      if epoch % print_steps == 0:
        print(f"Epoch {epoch}; loss: {loss.item()}")
    
    return self.model

  def predict(self, model, values):
    predictions = []
    for v in values:
      x = self.data.to_tensor(v)
      out = model(x)
      predictions.append(torch.argmax(out, axis=1))
    return predictions


# trainer = TrainWrapper(sru=False)
# model = trainer.train()
# trainer.predict(model)

    