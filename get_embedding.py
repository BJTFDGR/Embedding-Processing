'''
We consider authorship of sentence
to be the sensitive attribute and target data to be a collection of
sentences of randomly sampled author set S from the held-out
dataset of BookCorpus, with 250 sentences per author. The goal
is to classify authorship s of sentences amongst Sgiven sentence
embeddings. For auxiliary data, we consider 10, 20, 30, 40 and 50
labeled sentences (disjoint from those in the target dataset) per
author. We also vary the size of author set |S| =100,200,400 and
800 where the inference task becomes harder as |S|increases

'''
import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

'''
Dataset: BookCorpus
250 sentences per author
consider 10, 20, 30, 40 and 50 labeled sentence
author set |S| =100,200,400,800
'''

dir_path='/home/chenboc1/githubfolder/bookcorpus/books1/newdataset'
booksentence=[]
# Here we get lens: all sentences included in list
files = os.listdir(dir_path) 
for file in files:
    if not os.path.isdir(file): 
        with open(dir_path+"/"+file,'r', encoding='UTF-8') as f:
            lines=f.readlines()
            booksentence.append(list(set(lines)))            
    else:
        '''
        path1 = path+"/"+file
        files1 = os.listdir(path1)
        for file1 in files1:
            f = open(path1+"/"+file1)
            s = [] 
            for ii in f: 
                s.append(ii)  
                list1.append(s)
        '''
for item in booksentence:item.sort(key = lambda i:len(i),reverse=True)

'''
We can use pool or process to implement the paralell computing
But the process is quite hard to set since the time usage for HPC is limited, 
if the number of process function is few, each process will run beyond time limit
if the number is many, it will take up all the resouce of HPC
So we consider use pool to automaticlly allocate the process.
Two problems occur in pool usage: 
communication in poll may break in HPC after excuting a chunk of time and 
Process assigned by pool may not go into function after a while 
This requires me to first split the data into a couple of pieces and keep that speratly 
When it finishs, use another function to combine them together.
'''  
import pandas as pd
import torch
import time
import multiprocessing
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from tqdm import tqdm

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)            
		print('new folder...') 
	else:
		print("---  There is this folder!  ---")
		

def get_embedding(sentence,i):
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tod_bert= AutoModel.from_pretrained("bert-base-uncased")    
    subbatch=[]
    new_folder='/mnt/home/chenboc1/githubfolder/bookcorpus/books1/pt/'
    for single_sentence in sentence:
        
        # 300 embeddings for each author
        if sentence.index(single_sentence)>300:break
        
        # input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
        # input_tokens = tokenizer.tokenize(input_text)
        story = tokenizer.encode(single_sentence, add_special_tokens=True)
        story = torch.Tensor(story).long()
        # input_tokens = tokenizer.tokenize(input_text)
        # story = torch.Tensor(tokenizer.convert_tokens_to_ids(input_tokens)).long()
        try:
            if len(story.size()) == 1: 
                story = story.unsqueeze(0) # batch size dimension


            with torch.no_grad():
                # input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
                # (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size)
                # tuple of (last_hidden_state, pooler_output).
                #So it's actually better to use outputs[0], which are the hidden representations of all tokens, and take the average. But what actually also works well in practice is just using the hidden representation of the [CLS] token. The reason this [CLS] token is introduced is because it can be used for classification tasks. You can see the hidden representation of the [CLS] token as a representation of the whole sequence (sentence). Since outputs[0] is of size (batch_size, seq_len, hidden_size), and we only want the vector of the [CLS] token, we can obtain it by typing outputs[0][:, 0, :]            

                output = tod_bert(story)  

                # If we use the final hidden state
                # Output : torch.Size([1, 8, 768])
                subbatch.append(output[0][:,0,:])
        except:
            print('ERROR found in', single_sentence)
    torch.save(subbatch, new_folder+'x_bert'+str(i)+'.pt')        
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (i, (end - start)))  

new_folder='/mnt/home/chenboc1/githubfolder/bookcorpus/books1/pt/'
mkdir(new_folder)

for author in booksentence:
    get_embedding(author,booksentence.index(author))
