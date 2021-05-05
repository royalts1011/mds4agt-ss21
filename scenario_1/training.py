import torch
from torch import optim
from net import SimpleNet, SimpleNet1D
import torch.nn.functional as f
from torch import nn
from utils import get_simple_data_loader, get_simple_eval_loader
import matplotlib.pyplot as plt
from torchvision.models.mobilenet import mobilenet_v2
import os

""" 
Inits:
"""
lr = 0.001
num_epochs = 20
device = 'cuda'

# change the model here. Also change view in loading methods
model = SimpleNet()
# model = mobilenet_v2(pretrained=True)

# model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=55)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_loader = get_simple_data_loader(cuda=torch.cuda.is_available())

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

if not os.path.exists('../../models'):
    os.mkdir('../../models')

torch.save(model, '../../models/trained_model_.pt')

#plt.plot(loss_ls_train)
#plt.show()

"""
Evaluation process: At the moment simply counting the correct predictions
"""
model.eval()

eval_loader = get_simple_eval_loader(cuda=torch.cuda.is_available())
correct_pred = 0
num_pred = 0

for data, label in eval_loader:
    pred = torch.argmax(f.softmax(model(data)))
    num_pred += 1

    if pred.data.item() == label.data.item():
        correct_pred += 1

print('Number of correct predictions: ' + str(correct_pred))
print('Number of Predictions: ' + str(num_pred))
print('Portion: ' + str(correct_pred/num_pred))


"""
Saving the model:
"""
if not os.path.exists('../../models'):
    os.mkdir('../../models')

torch.save(model, '../../models/trained_model_' + str(round((correct_pred/num_pred) * 100)) + '%.pt')
