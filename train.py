import torch
import torch.nn as nn
from clean import Data
from model import RNN


# data / model
hidden_size = 12
hidden_layers = 5
data = Data()
model = RNN(data.embed_dim, hidden_size, hidden_layers, data.classes)


# hparams
n_epochs = 1000
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training loop

for epoch in range(n_epochs):
  # iterate through dataset
  for i in range(len(data.x_tr)):
    x = data.to_tensor(data.x_tr[i]) # tensor
    y = data.y_tr[i].view(1, data.y_tr.shape[-1])

    output = model(x)
    loss = criterion(output, y.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if epoch % 10 == 0:
    print(f"Epoch {epoch}")
    print(f'loss: {loss.item()}')

x = data.to_tensor("x_0")
out = model(x)
print(torch.argmax(out, axis=1))
print(out)
x = data.to_tensor("x_6")
out = model(x)
print(torch.argmax(out, axis=1))
print(out)