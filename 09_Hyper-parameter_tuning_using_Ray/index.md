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


## Ray on multiple GPUs

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import ray
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch


import numpy as np


#%%

#%% import packages:
import matplotlib.pyplot as plt
import os
import h5py

#%% check resources
# 
# if cuda is available:
if torch.cuda.is_available():
    print('cuda is available')
    device = torch.device("cuda:0")
    numGPUs=torch.cuda.device_count()
else:
    print('cuda is not available')
    device = torch.device("cpu")
    numGPUs=0

print(f"numGPUs: {numGPUs}")

#number of cpus:
print('SLURM_GPUS_PER_TASK: ',os.getenv('SLURM_GPUS_PER_TASK'))
numCPUs=int(os.getenv('SLURM_CPUS_PER_TASK'))
print('numCPUs: ',numCPUs)

#print(os.listdir('/data'))


#%% set up data sets:
#data=h5py.File('/data/mnist.h5')
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



loss_fn = nn.CrossEntropyLoss()

def trainTestNet(config,train_set,test_set):    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    #set up data loaders:
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    net=nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, config['lastSize']),
            nn.ReLU(),
            nn.Linear(config['lastSize'], 10) 
                    )

    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
    
    # train the model:
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
            
        #test the model:
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

        # return 100*correct/total
        session.report({"mean_accuracy": 100*correct/total})  # Report to Tune



search_space = {"lr": tune.uniform(1e-4, 1e-2), "lastSize": tune.randint(10,100)}
algo = OptunaSearch()  

trainable_with_resources = tune.with_resources(
    tune.with_parameters(trainTestNet, train_set=train_set, test_set=test_set),
    {"cpu": 1, "gpu": 1}  # Each trial uses 1 GPU
)

tuner = tune.Tuner(  
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        metric="mean_accuracy",
        mode="max",
        search_alg=algo,
        num_samples=16,
    ),
    run_config=air.RunConfig(
        stop={"training_iteration": 1},
    ),
    param_space=search_space,
)

#if I don't limit num_cpus, ray tries to use the whole node and crashes:
ray.init( num_cpus=numCPUs,num_gpus=numGPUs, log_to_driver = False)

result_grid = tuner.fit()
print("Best config is:", result_grid.get_best_result().config,
 ' with accuracy: ', result_grid.get_best_result().metrics['mean_accuracy'])
```

Set up jobscript
```bash
#!/bin/bash
#SBATCH --job-name=helloMnist
#SBATCH --account=project_465001063
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=8
#SBATCH --output=1_tasks_56_cpu.txt

srun singularity exec --bind /pfs,/scratch,/projappl,/project,/flash,/appl ray_container.sif python ray_mnist_parallel.py
```
Note that we need to resctict the number of CPUs as Ray does not understand the concept of SLURM. This results in 
```bash
Trial status: 16 TERMINATED
Current time: 2024-05-15 09:07:54. Total running time: 1min 19s
Logical resource usage: 1.0/1 CPUs, 1.0/8 GPUs (0.0/1.0 accelerator_type:AMD-Instinct-MI250X)
Current best trial: 0b98f90c with mean_accuracy=69.33333333333333 and params={'lr': 0.0041376522033940745, 'lastSize': 41}
╭─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name              status                lr     lastSize       acc     iter     total time (s) │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ trainTestNet_b01dc912   TERMINATED   0.00359687            81   62.6           1            4.89447 │
│ trainTestNet_a13543da   TERMINATED   0.00553352            91   35.7333        1            1.3245  │
│ trainTestNet_a15d9380   TERMINATED   0.00665843            24   60.1333        1            1.22044 │
│ trainTestNet_2645f5eb   TERMINATED   0.00940528            39   27             1            1.2537  │
│ trainTestNet_17a441ec   TERMINATED   0.00930793            70   48.0667        1            1.23321 │
│ trainTestNet_0b98f90c   TERMINATED   0.00413765            41   69.3333        1            2.3021  │
│ trainTestNet_e8be5a40   TERMINATED   0.00629059            96   44.4667        1            1.25362 │
│ trainTestNet_5951f796   TERMINATED   0.0085433             17   25.7333        1            1.23503 │
│ trainTestNet_4145d119   TERMINATED   0.00252217            79   68.8667        1            1.2265  │
│ trainTestNet_c31a8a72   TERMINATED   0.00515077            73   10.2           1            1.23375 │
│ trainTestNet_8c8ce7ae   TERMINATED   0.00755736            73   53.0667        1            1.22602 │
│ trainTestNet_3802580c   TERMINATED   0.000103717           50   29.6           1            1.22028 │
│ trainTestNet_0c63e529   TERMINATED   0.000356371           52   34.1333        1            1.21622 │
│ trainTestNet_7cb0cd81   TERMINATED   0.00296936            55   37.6           1            1.22157 │
│ trainTestNet_b02c6c24   TERMINATED   0.00288709            35   48.0667        1            1.23944 │
│ trainTestNet_79a3474f   TERMINATED   0.00255889            35   15             1            1.22876 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best config is: {'lr': 0.0041376522033940745, 'lastSize': 41}  with accuracy:  69.33333333333333
```

### Ray on multiple nodes
still work in progress. based on community supported implementation https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#walkthrough-using-ray-with-slurm

