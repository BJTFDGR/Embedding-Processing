import sys
import os 
import torch


new_folder='/home/chenboc1/githubfolder/bookcorpus/books1/pt/'
sys.path.append(new_folder)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print('There are %d GPU(s) available.' % n_gpu)
    print('We will use the GPU:', [torch.cuda.get_device_name(i) for i in range(n_gpu)])
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


authornumber=len(os.listdir(new_folder))

# Load data here
# In white box setting, first 50 sentences are used for private training
import random
random.seed(10)

def data_split(full_list, ratio, shuffle=False):
    """
    Divide dataset: divide full_list with random ratio into sublist_1 sublist_2
    :param full_list: 
    :param ratio:     
    :param shuffle:   
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def AccuarcyComputeT3(pred,label):
    with torch.no_grad():
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
    #     print(pred.shape(),label.shape())

        count=0
        values, indices = torch.tensor(pred).topk(3, dim=1, largest=True, sorted=True)
        for i in range(indices.shape[0]):
            if label[i] in indices[i]:count+=1
        return count/len(label)

# accuarcy
def AccuarcyCompute(pred,label):
    with torch.no_grad():
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
    #     print(pred.shape(),label.shape())
        #test_np = (np.argmax(pred,1) == label)
        test_np = (np.argmax(pred,1) == label)
        test_np = np.float32(test_np)
        return np.mean(test_np)
 

# aux_data,aux_label=[],[]
# text_data,text_label=[],[]
# for i in range(300):
#     sub_data1, sub_data2 = data_split(torch.load(new_folder+'x_bert'+str(i)+'.pt', map_location=torch.device("cpu")), ratio=0.1, shuffle=True)
#     aux_data+=sub_data1
#     aux_label+=[i for j in range(int(300*0.1))]
#     text_data+=sub_data2
#     text_label+=[i for j in range(300- int(300*0.1))]


data,label=[],[]
for i in range(400):
    tmp_list=torch.load(new_folder+'x_bert'+str(i)+'.pt', map_location=torch.device("cpu"))[:300]
    data += tmp_list
    label+=[i for j in range(len(tmp_list))]
    

n_epochs = 30    
batch_size =128  

new_data=torch.cat(data)
new_label=torch.tensor(label, dtype=torch.long)

dataset =  torch.utils.data.TensorDataset(new_data,new_label) 
validation_split = .1
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
train_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)    

# class MLP(torch.nn.Module):
#     def __init__(self):
#         super(MLP,self).__init__()
#         self.fc1 = torch.nn.Linear(768,200)
#         self.fc2 = torch.nn.Linear(200,2)

        
#     def forward(self,din):
#         din = din.view(-1,768)
#         dout = torch.nn.functional.sigmoid(self.fc1(din))
#         dout = torch.nn.functional.sigmoid(self.fc2(dout))
# #        return pt.nn.functional.sigmoid(dout,dim=1)
#         return dout

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = torch.nn.Linear(768,30)

    def forward(self,din):
        din = din.view(-1,768)
        dout = torch.nn.functional.sigmoid(self.fc1(din))
#        return pt.nn.functional.sigmoid(dout,dim=1)
        return dout

model = MLP().cuda()
print(model)

# loss func and optim
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0101, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
lossfunc = torch.nn.CrossEntropyLoss().cuda()
#lossfunc = pt.nn.BCEWithLogitsLoss().cuda()

# test accuarcy
# print(AccuarcyCompute(
#     np.array([[1,10,6],[0,2,5]],dtype=np.float32),
#     np.array([[1,2,8],[1,2,5]],dtype=np.float32)))


training_data_list_sample,test_data_list_sample=[],[]
training_data_list_epoch,test_data_list_epoch=[],[]

def train():
    for epoch in range(n_epochs):
        acc=0
        accuarcy_list = []

        for i,data in enumerate(train_dataset):        
            optimizer.zero_grad()
        
            (inputs,labels) = data
            inputs = torch.autograd.Variable(inputs.to(device)).cuda()
            labels = torch.autograd.Variable(labels.to(device)).cuda()
        
            outputs = model(inputs)
        
            loss = lossfunc(outputs,labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad(): 
                if (i+1) == len(train_dataset):
                    acc=AccuarcyComputeT3(outputs,labels)
                    print(epoch,":",acc)
                if acc>0.75:
                    print(inputs,labels,loss,np.argmax(outputs.cpu().data.numpy(),1))
                
                mid=AccuarcyComputeT3(outputs,labels)
                if mid>0.5:
                    with open('5_layer_top_3_training_data.txt','a+') as f:
                        print("MID>0.5",mid,np.argmax(outputs.cpu().data.numpy(),1),labels)   
                accuarcy_list.append(mid)
                training_data_list_sample.append(mid)

        with torch.no_grad():  
            acc=sum(accuarcy_list) / len(accuarcy_list)
            training_data_list_epoch.append(acc)
        accuarcy_list = []

        with torch.no_grad():  
            for i,(inputs,labels) in enumerate(test_dataset):
                inputs = torch.autograd.Variable(inputs).cuda()
                labels = torch.autograd.Variable(labels).cuda()
                outputs = model(inputs)
                mid=AccuarcyComputeT3(outputs,labels)
                if mid>0.5:
                    with open('5_layer_top_3_test_data.txt','a+') as f:
                        print("MID>0.5",mid,np.argmax(outputs.cpu().data.numpy(),1),labels)                
                accuarcy_list.append(mid)
                test_data_list_sample.append(mid)
            acc=sum(accuarcy_list) / len(accuarcy_list)
            acc_lst.append(acc)
            test_data_list_epoch.append(acc)
            if epoch%20==0:    
                print('***** Test Result: {:.4f}, Step: {}'.format(acc, epoch))              
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

plt.plot(range(len(test_data_list_epoch)), test_data_list_epoch)
plt.savefig('5_layer_top_3_test_data_list_epoch.png')
plt.close()
plt.plot(range(len(training_data_list_epoch)), training_data_list_epoch)
plt.savefig('5_layer_top_3_training_data_list_epoch.png')
plt.close()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.hist(test_data_list_sample, range=(0,1))
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 1)
plt.title("5_layer_top_3_test_data_list_sample")
plt.savefig('5_layer_top_3_test_data_list_sample.png')
plt.close()
plt.hist(training_data_list_sample, range=(0,1))
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 1)
plt.title("5_layer_top_3_training_data_list_sample")
plt.savefig('5_layer_top_3_training_data_list_sample.png')
plt.close()
print(len(training_data_list_sample))
print(len(test_data_list_sample))
print(sum(i >0.4 for i in training_data_list_sample))
print(sum(i >0.4 for i in test_data_list_sample)) #13750 32000 780 2144

k1, k2 = [], []
print(sum(i > 0.7 for i in test_data_list_sample))
for i in training_data_list_sample:
    if i > 0.3:
        k1.append(i)
    else:
        k2.append(i)
plt.hist(k2)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 0.3)
plt.title("Binary_5_layer_top_1_test_data_list_sample")
plt.savefig('F02_5_layer_top_3_training_data_list_sample.png')
plt.close()

plt.hist(k1)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0.3, 0.9)
plt.title("Binary_5_layer_top_1_test_data_list_sample")
plt.savefig('F28_5_layer_top_3_training_data_list_sample.png')
plt.close()

k1, k2 = [], []
for i in test_data_list_sample:
    if i > 0.3:
        k1.append(i)
    else:
        k2.append(i)
plt.hist(k2)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 0.3)
plt.title("Binary_5_layer_top_1_test_data_list_sample")
plt.savefig('F02_5_layer_top_3_test_data_list_sample.png')
plt.close()

plt.hist(k1)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0.3, 0.9)
