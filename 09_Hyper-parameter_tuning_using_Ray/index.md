# 09 Hyper-parameter tuning using Ray

These slides are loosely based on Kaare Mikkelsen's slides (https://www.au.dk/en/mikkelsen.kaare@ece.au.dk) <span style="color:red">*Ask for link to slides*</span>.

## Requirements

First we need to set up a container with ray installed
```bash
module load LUMI/23.03 cotainr
cotainr build minimal_pytorch.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-5.6.1.sif --conda-env=minimal_ray.yml
```
Download mnist data 
```bash
wget https://anon.erda.au.dk/share_redirect/AIYv1rmrtI -O mnist.h5
```

Set up python script for neural net
```python
#%% import packages:
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import h5py

#%% check if cuda is available:
if torch.cuda.is_available():
    print('cuda is available')
    device = torch.device("cuda:0")
else:
    print('cuda is not available')
    device = torch.device("cpu")

#%% set up data sets:
data=h5py.File('/project/project_465001063/decristo/ray_playground/data/mnist.h5')
Xtrain=np.array(data['Xtrain'])
Xtest=np.array(data['Xtest'])
ytrain=np.array(data['ytrain'])
ytest=np.array(data['ytest'])

#convert numpy arrays to torch tensors:
Xtrain=torch.from_numpy(Xtrain).float()
Xtest=torch.from_numpy(Xtest).float()
ytrain=torch.from_numpy(ytrain)
ytest=torch.from_numpy(ytest)

#set up data sets:
train_set = torch.utils.data.TensorDataset(Xtrain, ytrain)
test_set = torch.utils.data.TensorDataset(Xtest, ytest)

#set up data loaders:
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

#%% set up sequential model:

net=nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5),
        nn.Dropout2d(),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 10) 
                )

#%% set up optimizer:

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


#%% set up loss function:

loss_fn = nn.CrossEntropyLoss()


#%% train the model:

net.to(device)
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        X, labels = data


        X=X.to(device)
        labels=labels.long().to(device)
        
        optimizer.zero_grad()
        
        outputs = net(X.view(-1, 1, 28, 28))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        print(epoch+1, loss.item())

print('Finished Training')

#%% test the model:
net.to('cpu')
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs.view(-1, 1, 28, 28))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))
```
