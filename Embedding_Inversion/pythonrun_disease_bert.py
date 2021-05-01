
import torch
import pandas as pd
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


# Load the dataset into a pandas dataframe.
df = pd.read_csv("./disease.csv")

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
print('Index',df.axes)
sentences= df['sentences'].values.tolist()        
labels= df['label'].values.tolist()    
diseaselist=['leg','hand','spine','chest','ankle','head','hip','arm','face','shoulder']
print(sentences[0:20])
print(len(sentences),len(labels))
            
import torch
import time
import multiprocessing
from transformers import AutoTokenizer, AutoModel
#tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
#tod_bert = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")



def get_embedding(input_vector,i):
    torch.save(input_vector, './dataset/disease/y_disease_bert'+str(i)+'.pt')
    print('Run task %s (%s)...'%(i, os.getpid()))
    start = time.time()
    print('here1')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tod_bert= AutoModel.from_pretrained("bert-base-uncased")
    print('here2')
    lengtha=len(input_vector)
    n=10
    step=int(lengtha/n)+1
    print(i,lengtha,step)
    for j in range(0,lengtha,step):
        subbatch=[]
        if os.path.exists('./dataset/disease/x_disease_bert_'+str(i)+'_'+str(j)+'_.pt'):continue
        for input_text in input_vector[j:j+step]:
            #print('current,',i,j,len(input_vector[j:j+step]))
            if len(subbatch)%20==0: print(i,j,len(subbatch),len(input_vector))
            # Encode text 
            # input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
            # input_tokens = tokenizer.tokenize(input_text)
            story = tokenizer.encode(input_text, add_special_tokens=True, padding= 'max_length',max_length = 64 )
            story = torch.Tensor(story).long()
            print('heree')
            # input_tokens = tokenizer.tokenize(input_text)
            # story = torch.Tensor(tokenizer.convert_tokens_to_ids(input_tokens)).long()
            try:
                if len(story.size()) == 1: 
                    story = story.unsqueeze(0) # batch size dimension

            #      if torch.cuda.is_available(): 
            #          tod_bert = tod_bert.cuda()
            #          story = story.cuda()

                with torch.no_grad():
                    #input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
                    #(one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size)
                    #  tuple of (last_hidden_state, pooler_output).
                    #So it's actually better to use outputs[0], which are the hidden representations of all tokens, and take the average. But what actually also works well in practice is just using the hidden representation of the [CLS] token. The reason this [CLS] token is introduced is because it can be used for classification tasks. You can see the hidden representation of the [CLS] token as a representation of the whole sequence (sentence). Since outputs[0] is of size (batch_size, seq_len, hidden_size), and we only want the vector of the [CLS] token, we can obtain it by typing outputs[0][:, 0, :]            

                    output = tod_bert(story)  

                    # If we use the final hidden state
                    # Output : torch.Size([1, 8, 768])
                    subbatch.append(output[0][:,0,:])
            except:
                print('ERROR found in', input_text)
        print('SAVED',i,j)
        torch.save(subbatch, './dataset/disease/x_disease_bert_'+str(i)+'_'+str(j)+'_.pt')     
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (i, (end - start)))    


from multiprocessing import Process
from multiprocessing import cpu_count, Pool
from itertools import chain
import pandas as pd
import numpy as np
cores = cpu_count() # 4
print(cores)


import os
length=len(sentences)
n=36
step=int(length/n)+1
print(length,step)
da,la=[],[]
with Pool(12) as p:
  for i in range(0,length,step):
    if not os.path.exists('./dataset/disease/x_disease_bert'+str(i+1)+'.pt'):    
      p.apply_async(get_embedding, args=(sentences[i:i+step],i+1,))
  print('Waiting for all subprocesses done...')
  p.close()
  p.join()
  print('All subprocesses done.')

import pandas as pd
from random import sample


if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()



for i in range(0,length,step):
    da=da+ torch.load('./dataset/disease/x_disease_bert'+str(i)+'.pt', map_location='cpu')

torch.save(da, './dataset/disease/x_disease_bert'+'.pt')

for i in range(0,length,step):
  emb=torch.load('./dataset/disease/y_disease_bert'+str(i)+'.pt',map_location='cpu')

  la=la+emb

torch.save(la, './dataset/disease/y_disease_bert'+'.pt')



