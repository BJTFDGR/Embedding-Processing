import torch
from transformers import AutoTokenizer, AutoModel
import time,random,os
import torch as pt
import torchvision as ptv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
## If there's a GPU available...
if torch.cuda.is_available():    
    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


ARR = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
LAST = ('1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2')
pro = {0:11,1:12,2:13,3:14,4:15,5:21,6:22,7:23,8:31,9:32,10:33,11:34,12:35,13:36,14:37,15:41,16:42,17:43,18:44,19:45,20:46,21:50,22:51,23:52,24:53,25:54,26:61,27:62,28:63,29:64,30:65}

## Random Generate ID number or phone number
def createIdcard():
  #Age 1-77
  t=time.localtime()[0];
  n=random.randint(t - 77, t - 1); 
  if t-n==1:
      m = random.randint(1, time.localtime()[1]);  
      if time.localtime()[1]-m==0:
          r=random.randint(1, time.localtime()[2]); 
      else:
          r = random.randint(1, 28)
  elif t - n==77:  
      m = random.randint(time.localtime()[1], 12);  
      if time.localtime()[1] - m == 0:
          r = random.randint(time.localtime()[2], 31);  
      else:
          r = random.randint(1, 28)
  else:
      m=random.randint(1, 12);
      r = random.randint(1, 28)
  x = '%02d%02d%02d%04d%02d%02d%03d' %(pro[random.randint(0,30)],
                      random.randint(1,99),
                      random.randint(1,99),
                      n,
                      m,
                      r,
                      random.randint(1,999))

  y = 0
  for i in range(17):
      y += int(x[i]) * ARR[i]
  return '%s%s' % (x, LAST[y % 11])

def createPhone():
  prelist=["130","131","132","133","134","135","136","137","138","139","147","150","151","152","153","155","156","157","158","159","186","187","188"]
  return random.choice(prelist)+"".join(random.choice("0123456789") for i in range(8))

print(createIdcard())
print(createPhone())

## Task one: no other text, only numbers considered
idlist=[]
phonelist=[]
for i in range(1000):
  idlist.append(createIdcard())
  phonelist.append(createPhone())
'''  
## Get Embedding not in paralell mode
# Input: sentences list as input_vector
# Output: Save embeddings and sentences as x.py/y.pt
# Print: Crossentropy loss, input, output

# Parameters:
#      line 84-85: Tod-bert or ALbert as Bert family
#      line 90-91: convert_tokens_to_ids is the same with the result  
#      line 100-102: comment out if we want to store into cpu memory
'''
def get_embedding(input_vector,i):
    subbatch=[]
    for input_text in input_vector:
        if len(subbatch) %20==0: print(i,len(subbatch),input_text,len(input_vector))
        #   tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
        #   tod_bert = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tod_bert= AutoModel.from_pretrained("bert-base-uncased")
        
        #   #input_tokens = tokenizer.tokenize(input_text)
        #   #story = torch.Tensor(tokenizer.convert_tokens_to_ids(input_tokens)).long()
    
        story = tokenizer.encode(input_text, add_special_tokens=True, padding= 'max_length',max_length = 64 )
        story = torch.Tensor(story).long()

        if len(story.size()) == 1: 
            story = story.unsqueeze(0) # batch size dimension

        if torch.cuda.is_available(): 
            tod_bert = tod_bert.cuda()
            story = story.cuda()

        with torch.no_grad():
            input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
            #(one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size)

            output = tod_bert(**input_context)  
            subbatch.append(output[0][:,0,:])

    torch.save(subbatch, 'x'+str(i)+'.pt')
    torch.save(input_vector, 'y'+str(i)+'.pt')

## If embedding is not prepared, then download that

if not os.path.exists('x2.pt'):
    get_embedding(idlist,2)
embedding_i_1=torch.load('x2.pt')
idlist=torch.load('y2.pt')

## Classfiy labels and prepare data/label as dataset
labellist=[]
for i in idlist:
  num=int(i[10:12])
  labellist.append(num)
labelclassed=list(set(labellist))
label=[]
for i in labellist:
  label.append(labelclassed.index(i))
label=torch.tensor(label)
data= torch.cat(embedding_i_1)
print(label[0:100])


## TensorDataset for  train_dataset and test_dataset
# Use 30% percent shadow dataset to predict 70% private dataset
n_epochs = 100     
batch_size = 64  
dataset =  torch.utils.data.TensorDataset(data,label)
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


##Model with complex one and simple one

#class MLP(torch.nn.Module):
#    def __init__(self):
#        super(MLP,self).__init__()
#        self.fc1 = torch.nn.Linear(768,400)
#        self.fc2 = torch.nn.Linear(400,25) 
#        self.fc3 = torch.nn.Linear(25,200)
#        self.fc4 = torch.nn.Linear(200,int(max(label)+1))
        
#    def forward(self,din):
#        din = din.view(-1,768)
#        dout = torch.nn.functional.sigmoid(self.fc1(din))
#        dout = torch.nn.functional.sigmoid(self.fc2(dout))
#        dout = torch.nn.functional.sigmoid(self.fc3(dout)) 
#        return torch.nn.functional.softmax(self.fc4(dout),dim=1)
#        return self.fc4(dout)

class MLP(torch.nn.Module):
     def __init__(self):
         super(MLP,self).__init__()
         self.fc1 = torch.nn.Linear(768,50)
         self.fc2 = torch.nn.Linear(50,int(max(label))+1)
#         self.fc3 = torch.nn.Linear(200,int(max(label))+1)
        
     def forward(self,din):
         din = din.view(-1,768)
         dout = torch.nn.functional.sigmoid(self.fc1(din))
         dout = torch.nn.functional.sigmoid(self.fc2(dout))
         #dout = torch.nn.functional.sigmoid(self.fc3(dout))
         return dout       
model = MLP().cuda()
print(model)


## loss func and optim
#optimizer = pt.optim.SGD(model.parameters(),lr=0.01,momentum=0.009)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
lossfunc = pt.nn.CrossEntropyLoss().cuda()

## accuarcy
def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)


def AccuarcyComputeT3(pred,label):
    with torch.no_grad():
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
        count=0
        values, indices = torch.tensor(pred).topk(3, dim=1, largest=True, sorted=True)
        for i in range(indices.shape[0]):
            if label[i] in indices[i]:count+=1
        return count/len(label)


training_data_list_sample,test_data_list_sample=[],[]
training_data_list_epoch,test_data_list_epoch=[],[]

def train():
    for epoch in range(n_epochs):
        acc=0
        accuarcy_list = []

        for i,data in enumerate(train_dataset):
        
            optimizer.zero_grad()
        
            (inputs,labels) = data
            inputs = pt.autograd.Variable(inputs.to(device)).cuda()
            labels = pt.autograd.Variable(labels.to(device)).cuda()
        
            outputs = model(inputs)
        
            loss = lossfunc(outputs,labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad(): 
                if (i+1) == len(train_dataset):
                    acc=AccuarcyCompute(outputs,labels)
                    print(epoch,":",acc)
                if acc>0.75:
                    print(inputs,labels,loss,np.argmax(outputs.cpu().data.numpy(),1))
                
                mid=AccuarcyCompute(outputs,labels)
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
                inputs = pt.autograd.Variable(inputs).cuda()
                labels = pt.autograd.Variable(labels).cuda()
                outputs = model(inputs)
                mid=AccuarcyCompute(outputs,labels)
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

## plot
plt.plot(range(len(acc_lst)), acc_lst)
plt.savefig('error.png')

plt.plot(range(len(test_data_list_epoch)), test_data_list_epoch)
plt.savefig('Binary_Reocvermonth5_layer_top_1_test_data_list_epoch.png')
plt.close()
plt.plot(range(len(training_data_list_epoch)), training_data_list_epoch)
plt.savefig('Binary_Reocvermonth5_layer_top_1_training_data_list_epoch.png')
plt.close()
plt.hist(test_data_list_sample, range=(0,1))
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 1)
plt.title("Binary_Reocvermonth5_layer_top_1_test_data_list_sample")
plt.savefig('Binary_Reocvermonth5_layer_top_1_test_data_list_sample.png')
plt.close()
plt.hist(training_data_list_sample, range=(0,1))
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 1)
plt.title("Binary_Reocvermonth5_layer_top_3_training_data_list_sample")
plt.savefig('Binary_Reocvermonth5_layer_top_1_training_data_list_sample.png')
plt.close()
print(len(training_data_list_sample))
print(len(test_data_list_sample))
print(sum(i >0.4 for i in training_data_list_sample))
print(sum(i >0.4 for i in test_data_list_sample)) #13750 32000 780 2144

k1, k2 = [], []
print(sum(i > 0.6 for i in test_data_list_sample))
for i in training_data_list_sample:
    if i > 0.6:
        k1.append(i)
    else:
        k2.append(i)
plt.hist(k2)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 0.6)
plt.title("Binary_Reocvermonth_5_layer_top_1_test_data_list_sample")
plt.savefig('Binary_ReocvermonthF02_5_layer_top_1_training_data_list_sample.png')
plt.close()

plt.hist(k1)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0.6, 0.9)
plt.title("Binary_Reocvermonth_5_layer_top_1_test_data_list_sample")
plt.savefig('Binary_ReocvermonthF28_5_layer_top_1_training_data_list_sample.png')
plt.close()

k1, k2 = [], []
for i in test_data_list_sample:
    if i > 0.6:
        k1.append(i)
    else:
        k2.append(i)
plt.hist(k2)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0, 0.6)
plt.title("Binary_Reocvermonth_5_layer_top_1_test_data_list_sample")
plt.savefig('Binary_ReocvermonthF02_5_layer_top_1_test_data_list_sample.png')
plt.close()

plt.hist(k1)
plt.xlabel("label")
plt.ylabel("freq")
plt.xlim(0.6, 0.9)
plt.title("Binary_Reocvermonth_5_layer_top_1_test_data_list_sample")
plt.savefig('Binary_ReocvermonthF28_5_layer_top_1_test_data_list_sample.png')
plt.close()
