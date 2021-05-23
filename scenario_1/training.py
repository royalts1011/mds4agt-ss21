import torch
from torch import optim
from net import SimpleNet
import torch.nn.functional as f
from torch import nn
from utils import get_simple_data_loader, get_simple_eval_loader
import matplotlib.pyplot as plt
import os
from meanAveragePrecision import computeMeanAveragePrecision
from sklearn.metrics import f1_score
import numpy as np

"""
Flags and variables:
"""
load_model = True
save_model = False

path_to_model = '..' + os.sep + '..' + os.sep + 'models' + os.sep + 'trained_model_50%.pt'
""" 
Inits:
"""
lr = 0.0001
num_epochs = 165
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = SimpleNet()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_loader = get_simple_data_loader(cuda=torch.cuda.is_available())

loss_ls_train = []
model.train()


if load_model:
    model = torch.load(path_to_model)
    """
    Training process:
    """
else:
    for epoch in range(num_epochs):
        for data, labels in train_loader:

            optimizer.zero_grad()

            prediction = model(data)

            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            loss_ls_train.append(loss.data.item())

        print('===> Epoch: {} loss: {:.5f}'.format(epoch, loss.data.item()))

    # Uncomment if you want to see the training process after Training
    # plt.plot(loss_ls_train)
    # plt.show()


"""
Evaluation process:
"""

model.eval()

eval_loader = get_simple_eval_loader(cuda=torch.cuda.is_available())
correct_pred = 0
num_pred = 0
map_sum = 0

label_ls = []
pred_ls = []
estimation_ls = []

for data, label in eval_loader:
    estimation = f.softmax(model(data), dim=1)
    pred = torch.argmax(estimation)
    num_pred += 1

    if pred.data.item() == label.data.item():
        correct_pred += 1

    label_ls.append(label.data.item())
    pred_ls.append(pred.data.item())
    estimation_ls.append(estimation.cpu().detach().numpy())


print('Number of correct predictions: ' + str(correct_pred))
print('Number of Predictions: ' + str(num_pred))
print('Accuracy: ' + str(correct_pred/num_pred))
print('F1-Score: ' + str(f1_score(y_true=label_ls, y_pred=pred_ls, average='weighted')))

map, ap = computeMeanAveragePrecision(labels=label_ls, softmaxEstimations=np.array(estimation_ls).squeeze())

print('MAP: ' + str(map))


if save_model:
    if not os.path.exists('..' + os.sep + '..' + os.sep + 'models'):
        os.mkdir('..' + os.sep + '..' + os.sep + 'models')
    torch.save(model, '..' + os.sep + '..' + os.sep + 'models' + os.sep + 'trained_model_' + str(round((correct_pred /
                                                                                                        num_pred) * 100)
                                                                                                 ) + '%.pt')
