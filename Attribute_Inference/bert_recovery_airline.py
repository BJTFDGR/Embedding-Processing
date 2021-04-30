

# coding: utf-8
import wget
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='3,4,5,6,7'
# If there's a GPU available...

# print('Downloading dataset...')

# # The URL for the dataset zip file.
# url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

# # Download the file (if we haven't already)
# if not os.path.exists('./cola_public_1.1.zip'):
#     wget.download(url, './cola_public_1.1.zip')

import torch

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

print('Downloading dataset...')
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./airline.csv")

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
print('Index',df.axes)
sentences= df['content'].values.tolist()     
citylist=['shanghai','Toronto','Paris','Rome','Sydney','Dubai','bangkok','Singapore','Frankfurt','London']

new_sentences=[]
new_labels=[]
for item in sentences:
    if len(item)<100:continue
    for city in citylist:
        if city in item[0:400] or city.lower() in item[0:400]:
            new_sentences.append(item[0:400])
            new_labels.append(citylist.index(city))
           


import pandas as pd
from collections import Counter
result = Counter(new_labels)
print(result)

            
from multiprocessing import Process
from multiprocessing import cpu_count, Pool
from itertools import chain
import pandas as pd
import numpy as np
cores = cpu_count() # 4
print(cores)

import os
length=len(new_sentences)
n=12
step=int(length/n)+1
print(length,step)
da,la=[],[]



for i in range(0,length,step):
    embedding_i_1=torch.load( 'x_airline_bert_'+str(i+1)+'__.pt', map_location='cpu')
    da=da+ embedding_i_1
    emb=torch.load('y_airline_bert'+str(i+1)+'.pt',map_location='cpu')
    la=la+emb

labels=[]
for item in la:
    city='London'
    if city in item or city.lower() in item:
        labels.append(1)
    else:labels.append(0)

data= torch.cat(da)
print(pd.value_counts(labels))
labels=torch.tensor(labels)
print(data.shape)
print(labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

weights,x,y=[],[],[]
for i in range(len(labels)):
    if labels[i]==1 and len(x)<2200:
        weights.append(i)
        x.append(1)
    if labels[i]==0 and len(y)<2200:
        weights.append(i)
        y.append(0)

    if len(x)>=2200 and len(y)>=2200: break

np.random.shuffle(weights)
data=data[weights]
labels=labels[weights]
print(labels[0:100])



import torch as pt
import torchvision as ptv
import numpy as np

# 定义全局变量
n_epochs = 10000     # epoch 的数目
batch_size =48  # 决定每次读取多少图片


# train_tensor = data[0:int(len(data)*0.1),:]
# train_label_tensor = labels[0:int(len(data)*0.1)]
# test_tensor = data[int(len(data)*0.1):len(data),:]
# test_label_tensor = labels[int(len(data)*0.1):len(data)]

# train_set = torch.utils.data.TensorDataset(train_tensor,train_label_tensor)
# train_dataset = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
# test_set = torch.utils.data.TensorDataset(test_tensor,test_label_tensor)
# test_dataset = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True)
        

dataset =  torch.utils.data.TensorDataset(data,labels)
batch_size =48 
validation_split = .7
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
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)    

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = torch.nn.Linear(768,180)
        self.fc2 = torch.nn.Linear(180,2)

        
    def forward(self,din):
        din = din.view(-1,768)
        dout = torch.nn.functional.sigmoid(self.fc1(din))
        dout = torch.nn.functional.sigmoid(self.fc2(dout))
#        return pt.nn.functional.sigmoid(dout,dim=1)
        return dout

# class MLP(torch.nn.Module):
#     def __init__(self):
#         super(MLP,self).__init__()
#         self.fc1 = torch.nn.Linear(768,200)
#         self.fc2 = torch.nn.Linear(200,50)
#         self.fc3 = torch.nn.Linear(50,25)
#         self.fc4 = torch.nn.Linear(25,2)

        
#     def forward(self,din):
#         din = din.view(-1,768)
#         dout = torch.nn.functional.sigmoid(self.fc1(din))
#         dout = torch.nn.functional.sigmoid(self.fc2(dout))
#         dout = torch.nn.functional.sigmoid(self.fc3(dout))
#         dout = torch.nn.functional.sigmoid(self.fc4(dout))
#         return pt.nn.functional.softmax(dout,dim=1)

model = MLP().cuda()
print(model)


weight_CE = torch.FloatTensor([2.47,2.51])
# loss func and optim
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0101, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
lossfunc = pt.nn.CrossEntropyLoss(weight=weight_CE).cuda()
#lossfunc = pt.nn.BCEWithLogitsLoss().cuda()

# accuarcy
def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
#     print(pred.shape(),label.shape())
    #test_np = (np.argmax(pred,1) == label)
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

# test accuarcy
# print(AccuarcyCompute(
#     np.array([[1,10,6],[0,2,5]],dtype=np.float32),
#     np.array([[1,2,8],[1,2,5]],dtype=np.float32)))
def train():
    for epoch in range(n_epochs):
        acc=0
        for i,data in enumerate(train_dataset):
        
            optimizer.zero_grad()
        
            (inputs,labels) = data
            inputs = pt.autograd.Variable(inputs.to(device)).cuda()
            labels = pt.autograd.Variable(labels.to(device)).cuda()
        
            outputs = model(inputs)
        
            loss = lossfunc(outputs,labels)
            loss.backward()
        
            optimizer.step()
            if (i+1) == len(train_dataset):
                acc=AccuarcyCompute(outputs,labels)
                print(i,":",acc)
            if acc>0.75:
                print(inputs,labels,loss,np.argmax(outputs.cpu().data.numpy(),1))


        accuarcy_list = []


        with torch.no_grad():  
            if epoch%20==0:  
                for i,(inputs,labels) in enumerate(test_dataset):
                    inputs = pt.autograd.Variable(inputs).cuda()
                    labels = pt.autograd.Variable(labels).cuda()
                    outputs = model(inputs)
                    accuarcy_list.append(AccuarcyCompute(outputs,labels))
                acc=sum(accuarcy_list) / len(accuarcy_list)
                print('***** Test Result: {:.4f}, Step: {}'.format(acc, epoch))
                acc_lst.append(acc)
                print(inputs,labels,np.argmax(outputs.cpu().data.numpy(),1))

def test():
    accuarcy_list = []
    for i,(inputs,labels) in enumerate(test_dataset):
        inputs = pt.autograd.Variable(inputs).cuda()
        labels = pt.autograd.Variable(labels).cuda()
        outputs = model(inputs)
        accuarcy_list.append(AccuarcyCompute(outputs,labels))
    print(sum(accuarcy_list) / len(accuarcy_list))
    print(inputs,labels,np.argmax(outputs.cpu().data.numpy(),1))


acc_lst=[]
model = MLP().cuda()
train()
test()
# Accuracy
plt.plot(range(len(acc_lst)), acc_lst)
plt.savefig('error.png')
