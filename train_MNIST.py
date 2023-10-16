#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-10-16 10:43:22 (ywatanabe)"

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set the random seed for reproducibility
set_random_seeds()

from PerceptronOrINN import PerceptronOrINN

# Initialize config
model_config = {
    "act_str": "intestine_simulated",
    "do_resample_act_funcs": False,
    "bs": 64,
    "n_fc_in": 784,
    "n_fc_1": 1000,
    "n_fc_2": 1000,
    "d_ratio_1": 0.5,
    "sigmoid_beta_0_mean": 1,
    "sigmoid_beta_0_var": 0,
    "sigmoid_beta_1_mean": 0,
    "sigmoid_beta_1_var": 0,
    "intestine_simulated_beta_0_mean": 3.06,
    "intestine_simulated_beta_0_var": 1.38,
    "intestine_simulated_beta_1_mean": 0,
    "intestine_simulated_beta_1_var": 3.23,
    "LABELS": list(range(10))
}

# Initialize the model, optimizer, and loss function
model = PerceptronOrINN(model_config).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Load MNIST Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training Loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item()}")

# Test Loop
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data = data.view(data.size(0), -1)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy: {(100 * correct / total):.2f}%")

"""
Epoch: 0 Batch: 0 Loss: 2.337707996368408
Epoch: 0 Batch: 100 Loss: 2.3034439086914062
Epoch: 0 Batch: 200 Loss: 2.308676242828369
Epoch: 0 Batch: 300 Loss: 2.291069269180298
Epoch: 0 Batch: 400 Loss: 2.2898669242858887
Epoch: 0 Batch: 500 Loss: 2.259193181991577
Epoch: 0 Batch: 600 Loss: 2.2262508869171143
Epoch: 0 Batch: 700 Loss: 2.2591052055358887
Epoch: 0 Batch: 800 Loss: 2.227748155593872
Epoch: 0 Batch: 900 Loss: 2.214931011199951
Epoch: 1 Batch: 0 Loss: 2.205514669418335
Epoch: 1 Batch: 100 Loss: 2.1726245880126953
Epoch: 1 Batch: 200 Loss: 2.179299831390381
Epoch: 1 Batch: 300 Loss: 2.1493451595306396
Epoch: 1 Batch: 400 Loss: 2.181206703186035
Epoch: 1 Batch: 500 Loss: 2.165060043334961
Epoch: 1 Batch: 600 Loss: 2.1518664360046387
Epoch: 1 Batch: 700 Loss: 2.107090950012207
Epoch: 1 Batch: 800 Loss: 2.0922796726226807
Epoch: 1 Batch: 900 Loss: 2.0883922576904297
Epoch: 2 Batch: 0 Loss: 2.067126750946045
Epoch: 2 Batch: 100 Loss: 2.0309972763061523
Epoch: 2 Batch: 200 Loss: 2.0337514877319336
Epoch: 2 Batch: 300 Loss: 2.038759231567383
Epoch: 2 Batch: 400 Loss: 1.9987781047821045
Epoch: 2 Batch: 500 Loss: 2.0336506366729736
Epoch: 2 Batch: 600 Loss: 1.9891915321350098
Epoch: 2 Batch: 700 Loss: 1.9620522260665894
Epoch: 2 Batch: 800 Loss: 1.9142768383026123
Epoch: 2 Batch: 900 Loss: 1.885387659072876
Epoch: 3 Batch: 0 Loss: 1.8634549379348755
Epoch: 3 Batch: 100 Loss: 1.922452449798584
Epoch: 3 Batch: 200 Loss: 1.843092918395996
Epoch: 3 Batch: 300 Loss: 1.8004738092422485
Epoch: 3 Batch: 400 Loss: 1.781233549118042
Epoch: 3 Batch: 500 Loss: 1.7627004384994507
Epoch: 3 Batch: 600 Loss: 1.6683433055877686
Epoch: 3 Batch: 700 Loss: 1.7323194742202759
Epoch: 3 Batch: 800 Loss: 1.638037919998169
Epoch: 3 Batch: 900 Loss: 1.6239663362503052
Epoch: 4 Batch: 0 Loss: 1.5985149145126343
Epoch: 4 Batch: 100 Loss: 1.5736466646194458
Epoch: 4 Batch: 200 Loss: 1.5068103075027466
Epoch: 4 Batch: 300 Loss: 1.3710484504699707
Epoch: 4 Batch: 400 Loss: 1.3616474866867065
Epoch: 4 Batch: 500 Loss: 1.401007890701294
Epoch: 4 Batch: 600 Loss: 1.426381230354309
Epoch: 4 Batch: 700 Loss: 1.313161849975586
Epoch: 4 Batch: 800 Loss: 1.3480956554412842
Epoch: 4 Batch: 900 Loss: 1.3751581907272339
Epoch: 5 Batch: 0 Loss: 1.3140125274658203
Epoch: 5 Batch: 100 Loss: 1.1920424699783325
Epoch: 5 Batch: 200 Loss: 1.2809444665908813
Epoch: 5 Batch: 300 Loss: 1.317400574684143
Epoch: 5 Batch: 400 Loss: 1.1676445007324219
Epoch: 5 Batch: 500 Loss: 1.2212748527526855
Epoch: 5 Batch: 600 Loss: 1.1691396236419678
Epoch: 5 Batch: 700 Loss: 1.1782811880111694
Epoch: 5 Batch: 800 Loss: 1.243850827217102
Epoch: 5 Batch: 900 Loss: 1.0820790529251099
Epoch: 6 Batch: 0 Loss: 1.176945686340332
Epoch: 6 Batch: 100 Loss: 1.1030327081680298
Epoch: 6 Batch: 200 Loss: 1.183580756187439
Epoch: 6 Batch: 300 Loss: 1.0883508920669556
Epoch: 6 Batch: 400 Loss: 0.9526631832122803
Epoch: 6 Batch: 500 Loss: 0.922423243522644
Epoch: 6 Batch: 600 Loss: 1.0819437503814697
Epoch: 6 Batch: 700 Loss: 0.939717173576355
Epoch: 6 Batch: 800 Loss: 1.0133917331695557
Epoch: 6 Batch: 900 Loss: 0.9692772030830383
Epoch: 7 Batch: 0 Loss: 0.9215019941329956
Epoch: 7 Batch: 100 Loss: 0.963954746723175
Epoch: 7 Batch: 200 Loss: 0.9186135530471802
Epoch: 7 Batch: 300 Loss: 0.8597159385681152
Epoch: 7 Batch: 400 Loss: 1.0357908010482788
Epoch: 7 Batch: 500 Loss: 0.9571436047554016
Epoch: 7 Batch: 600 Loss: 0.9383936524391174
Epoch: 7 Batch: 700 Loss: 0.8021243810653687
Epoch: 7 Batch: 800 Loss: 0.8582736849784851
Epoch: 7 Batch: 900 Loss: 0.8480632901191711
Epoch: 8 Batch: 0 Loss: 0.752300500869751
Epoch: 8 Batch: 100 Loss: 0.9244712591171265
Epoch: 8 Batch: 200 Loss: 0.8200180530548096
Epoch: 8 Batch: 300 Loss: 0.8038215041160583
Epoch: 8 Batch: 400 Loss: 0.8257690668106079
Epoch: 8 Batch: 500 Loss: 0.7846906185150146
Epoch: 8 Batch: 600 Loss: 0.6740760207176208
Epoch: 8 Batch: 700 Loss: 0.6744505763053894
Epoch: 8 Batch: 800 Loss: 0.6676566004753113
Epoch: 8 Batch: 900 Loss: 0.6961811184883118
Epoch: 9 Batch: 0 Loss: 0.8140008449554443
Epoch: 9 Batch: 100 Loss: 0.6493793725967407
Epoch: 9 Batch: 200 Loss: 0.6754528880119324
Epoch: 9 Batch: 300 Loss: 0.7368165254592896
Epoch: 9 Batch: 400 Loss: 0.7969047427177429
Epoch: 9 Batch: 500 Loss: 0.6278544664382935
Epoch: 9 Batch: 600 Loss: 0.6742461323738098
Epoch: 9 Batch: 700 Loss: 0.6166834235191345
Epoch: 9 Batch: 800 Loss: 0.7250018119812012
Epoch: 9 Batch: 900 Loss: 0.717006266117096
Accuracy: 83.51%
"""
