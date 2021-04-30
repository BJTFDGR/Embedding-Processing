## Use 
# coding: utf-8
import wget
import os
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES']='7'

from multiprocessing import cpu_count, Pool
from collections import Counter
import torch
import pandas as pd
import numpy as np
import torch as pt
import torchvision as ptv

## checking GPU
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print('Start')

## Read salad Data
# Load the dataset into a pandas dataframe.
df = pd.read_csv("./saladreview.csv")
salad_sentences= df['text'].values.tolist() 

length,n=len(salad_sentences),4
step=int(length/n)+1
print(length,step)
da,la=[],[]

for i in range(0,length,step):
    embedding_i_1=torch.load( 'x_salad_bert'+str(i+1)+'.pt', map_location='cpu')
    da=da+ embedding_i_1
    emb=torch.load('y_salad_bert'+str(i+1)+'.pt',map_location='cpu')
    la=la+emb

## Read Nonsalad Data
# Load the dataset into a pandas dataframe.
df = pd.read_csv("./nosaladreview.csv")
non_salad_sentences= df['text'].values.tolist() 

length=len(non_salad_sentences)
n,step=4,int(length/n)+1
print(length,step)

for i in range(0,length,step):
    embedding_i_1=torch.load( 'x_nosalad_bert'+str(i+1)+'.pt', map_location='cpu')
    da=da+ embedding_i_1
    emb=torch.load('y_nosalad_bert'+str(i+1)+'.pt',map_location='cpu')
    la=la+emb

salad_labels=[]
for item in la:
    city='salad'
    if city in item or city.lower() in item:
        salad_labels.append(1)
    else:salad_labels.append(0)

salad_data= torch.cat(da)

print('training set label distribution',pd.value_counts(salad_labels))
salad_labels=torch.tensor(salad_labels)
print(salad_data.shape)
print(salad_labels.shape)

## Combine salad and non-salad as training dataset

indices = np.arange(salad_data.shape[0])
np.random.shuffle(indices)
salad_data = salad_data[indices]
salad_labels = salad_labels[indices]
np.random.shuffle(indices)
salad_data = salad_data[indices]
salad_labels = salad_labels[indices]

# 定义全局变量
n_epochs = 250     # epoch 的数目
batch_size =48  # 决定每次读取多少图片

dataset =  torch.utils.data.TensorDataset(salad_data,salad_labels)
batch_size =48 
validation_split = 0.01
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
train_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)


## Prepare the test dataset
# Load the dataset into a pandas dataframe.
df = pd.read_csv("./airline.csv")
sentences= df['content'].values.tolist()     
city='salad'
count=0
new_sentences=[]
new_labels=[]
for item in sentences:
    if len(item)<100:continue
    if city in item[0:800] or city.lower() in item[0:800]:
        new_sentences.append(item[0:800])
        new_labels.append(1)	
        count+=1
        if count>2000: break
length=len(new_sentences)
n=12
step=int(length/n)+1
print(length,step)
da,la=[],[]
for i in range(0,length,step):
    embedding_i_1 = torch.load('x_saladairline_bert_'+str(i+1)+'__.pt', map_location='cpu')
    da = da + embedding_i_1
    emb = torch.load('y_saladairline_bert'+str(i+1)+'.pt', map_location='cpu')
    la = la+emb


# Load the dataset into a pandas dataframe.
df = pd.read_csv("./airline.csv")

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
print('Index', df.axes)
sentences = df['content'].values.tolist()
citylist = ['shanghai', 'Toronto', 'Paris', 'Rome', 'Sydney',
            'Dubai', 'bangkok', 'Singapore', 'Frankfurt', 'London']

new_sentences = []
new_labels = []
for item in sentences:
    if len(item) < 100:
        continue
    for city in citylist:
        if city in item[0:400] or city.lower() in item[0:400]:
            new_sentences.append(item[0:400])
            new_labels.append(citylist.index(city))


result = Counter(new_labels)
print(result)

cores = cpu_count()  # 4
print(cores)

length = len(new_sentences)
n = 12
step = int(length/n)+1
print(length, step)

for i in range(0, length, step):
    embedding_i_1 = torch.load(
        'x_airline_bert_'+str(i+1)+'__.pt', map_location='cpu')
    da = da + embedding_i_1
    emb = torch.load('y_airline_bert'+str(i+1)+'.pt', map_location='cpu')
    la = la+emb

labels = []
for item in la:
    city = 'salad'
    if city in item or city.lower() in item:
        labels.append(1)
    else:
        labels.append(0)

data = torch.cat(da)
print(pd.value_counts(labels))
labels = torch.tensor(labels)
print(data.shape)
print(labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

weights, x, y = [], [], []
for i in range(len(labels)):
    if labels[i] == 1 and len(x) < 500:
        weights.append(i)
        x.append(1)
    if labels[i] == 0 and len(y) < 500:
        weights.append(i)
        y.append(0)

    if len(x) >= 500 and len(y) >= 500:
        break

np.random.shuffle(weights)
data = data[weights]
labels = labels[weights]
print(labels[0:100])
print('test set distribution',len(x),len(y))


dataset =  torch.utils.data.TensorDataset(data[0:int(0.8*len(weights))],labels[0:int(0.8*len(weights))])
batch_size =48 
validation_split = .99
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
test_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)    


dataset =  torch.utils.data.TensorDataset(data[int(0.8*len(weights)):len(weights)],labels[int(0.8*len(weights)):len(weights)])
batch_size =48 
validation_split = .99
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
shadow_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)    



print('-------------Length of Dataset--------------')
print('-------------',len(train_dataset),len(test_dataset),len(shadow_dataset),'--------------')

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import datetime
import torch.nn.functional as FN
MODEL_NAME = 'DANN'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FeatureExtractor(torch.nn.Module):
    def __init__(self, n_feature=768, n_hidden=1024, n_output=256, dropout=0.3):
        super(FeatureExtractor, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden//2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden//2)

        self.hidden_3 = torch.nn.Linear(n_hidden//2, n_hidden//4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden//4)

        self.out = torch.nn.Linear(n_hidden//4, n_output)  # output layer

    def forward(self, x):
        x = FN.relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))
        x = FN.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.dropout(self.bn2(x))
        x = FN.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.dropout(self.bn3(x))
        x = self.out(x)
        return x


class Classifier(nn.Module):
    """
        Classifier
    """
    def __init__(self, input_size=256, num_classes=2):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, h):
        c = self.layer(h)
        return c

class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=256, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, h):
        y = self.layer(h)
        return y


F = FeatureExtractor().to(DEVICE)
C = Classifier().to(DEVICE)
D = Discriminator().to(DEVICE)

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
])


bce = nn.BCELoss()
xe = nn.CrossEntropyLoss()
F_opt = torch.optim.Adam(F.parameters())
C_opt = torch.optim.Adam(C.parameters())
D_opt = torch.optim.Adam(D.parameters())

max_epoch =5
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_batches = len(train_dataset)//batch_size
# lamda = 0.01
D_src = torch.ones(batch_size, 1).to(DEVICE) # Discriminator Label to real
D_tgt = torch.zeros(batch_size, 1).to(DEVICE) # Discriminator Label to fake
D_labels = torch.cat([D_src, D_tgt], dim=0)

## Training code
def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

ll_c, ll_d = [], []
acc_lst = []

shadow_set = iter(shadow_dataset)

def sample_shadow(step, n_batches):
    global shadow_set
    if step % n_batches == 0:
        shadow_set = iter(shadow_dataset)
    return shadow_set.next()


for epoch in range(1, max_epoch+1):
    for i,data in enumerate(train_dataset):
        (inputs,labels) = data
        shadow, _ = sample_shadow(step, n_batches)
        inputs = inputs.to(device).cuda()
        labels = labels.to(device).cuda()
        shadow = shadow.to(device).cuda()
        # Training Discriminator
            
        x = torch.cat([inputs, shadow], dim=0)
        h = F(x)
        y = D(h.detach())
        if y.size()!=D_labels.size():continue
        Ld = bce(y, D_labels)
        D.zero_grad()
        Ld.backward()
        D_opt.step()
        
        
        c = C(h[:batch_size])
        y = D(h)
        Lc = xe(c, labels)
        Ld = bce(y, D_labels)
        lamda = 0.1*get_lambda(epoch, max_epoch)
        Ltot = Lc -lamda*Ld
        
        
        F.zero_grad()
        C.zero_grad()
        D.zero_grad()
        
        Ltot.backward()
        
        C_opt.step()
        F_opt.step()
        
        if step % 5 == 0:
            dt = datetime.datetime.now().strftime('%H:%M:%S')
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, lambda: {:.4f} ---- {}'.format(epoch, max_epoch, step, Ld.item(), Lc.item(), lamda, dt))
            ll_c.append(Lc)
            ll_d.append(Ld)
        
        if step % 5 == 0:
            F.eval()
            C.eval()
            with torch.no_grad():                
                corrects = torch.zeros(1).to(DEVICE)
                for idx, (tgt, labels) in enumerate(test_dataset):
                    tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                    c = C(F(tgt))
                    _, preds = torch.max(c, 1)
                    corrects += (preds == labels).sum()
                acc = corrects.item() / len(test_dataset.dataset)
                print('***** Test Result: {:.4f}, Step: {}'.format(acc, step))
                acc_lst.append(acc)
                
            F.train()
            C.train()
        step += 1




# Accuracy
plt.plot(range(len(ll_c)), ll_c)
plt.savefig('XE loss.png')
plt.close()
plt.plot(range(len(ll_d)), ll_d)
plt.savefig('Discriminator loss.png')
plt.close()
plt.plot(range(len(acc_lst)), acc_lst)
plt.savefig('Accuracy.png')
print(max(acc_lst))
