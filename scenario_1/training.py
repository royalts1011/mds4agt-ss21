import torch
from torch import optim
from net import SimpleNet, SimpleNet1D
from torch import nn
from utils import get_simple_data_loader, get_simple_eval_loader
import matplotlib.pyplot as plt


""" 
Inits:
"""
lr = 0.001
num_epochs = 50

# change the model here. Also change view in loading methods
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_loader = get_simple_data_loader()

loss_ls_train = []
model.train()

"""
Training process:
"""
for epoch in range(num_epochs):
    for data, labels in train_loader:

        optimizer.zero_grad()

        prediction = model(data)

        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        loss_ls_train.append(loss.data.item())

    print('===> Epoch: {} loss: {:.4f}'.format(epoch, loss.data.item()))

#plt.plot(loss_ls_train)
#plt.show()

"""
Evaluation process: At the moment simply counting the correct predictions
"""
model.eval()

eval_loader = get_simple_eval_loader()
correct_pred = 0
num_pred = 0

for data, label in eval_loader:
    pred = torch.argmax(model(data))
    num_pred += 1

    if pred.data.item() == label.data.item():
        correct_pred += 1

print('Number of correct predictions: ' + str(correct_pred))
print('Number of Predictions: ' + str(num_pred))
print('Portion: ' + str(correct_pred/num_pred))


