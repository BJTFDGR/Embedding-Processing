import pandas as pd
import torch
import time
import multiprocessing
from transformers import AutoTokenizer, AutoModel
from collections import Counter

'''
File: Get embedding in paralell
Input: airline reviews
Output: Embeddings for sentences in citylist 
        'x_airline_bert_'+str(i)+'__.pt'
        original sentences
        'y_airline_bert'+str(i)+'.pt'
Total Number: Counter({9: 2281, 4: 1265, 1: 1173, 7: 1136, 2: 926, 5: 750, 3: 608, 8: 596})   8735 728
sentences/embeddings
'''

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
           

result = Counter(new_labels)
print(result)

## Make get_embedding work in paralell mode   
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
def get_embedding(input_vector,i):
    torch.save(input_vector, 'y_airline_bert'+str(i)+'.pt')
    print('Run task %s (%s)...'%(i, os.getpid()))
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tod_bert= AutoModel.from_pretrained("bert-base-uncased")

    lengtha=len(input_vector)
    n=10
    step=int(lengtha/n)+1
    print(i,lengtha,step)
    for j in range(0,lengtha,step):
        subbatch=[]
        if os.path.exists('x_airline_bert_'+str(i)+'_'+str(j)+'_.pt'):continue
        for input_text in input_vector[j:j+step]:
            #print('current,',i,j,len(input_vector[j:j+step]))
            if len(subbatch)%20==0: print(i,j,len(subbatch),len(input_vector))
            # Encode text 
            # input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
            # input_tokens = tokenizer.tokenize(input_text)
            story = tokenizer.encode(input_text, add_special_tokens=True)
            story = torch.Tensor(story).long()
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
        torch.save(subbatch, 'x_airline_bert_'+str(i)+'_'+str(j)+'_.pt')        
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (i, (end - start)))  


from multiprocessing import Process

from multiprocessing import cpu_count, Pool
from itertools import chain
import numpy as np
import os
cores = cpu_count() # 4
print(cores)



length=len(new_sentences)
n=12
step=int(length/n)+1
print(length,step)
da,la=[],[]
with Pool(8) as p:
#    print('Parent process %s.' % os.getpid())
#    p = multiprocessing.Pool(16)
    for i in range(0,length,step):
        if not os.path.exists('x_airline_bert'+str(i+1)+'.pt'):    
            p.apply_async(get_embedding, args=(new_sentences[i:i+step],i+1,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


def orgnizefile(input_vector,i):
    print('Run task %s (%s)...'%(i, os.getpid()))
    start = time.time()
    lengtha=len(input_vector)
    n=10
    step=int(lengtha/n)+1
    print(i,lengtha,step)
    subbatch=[]
    for j in range(0,lengtha,step):        
        subbatch=subbatch+ torch.load('x_airline_bert_'+str(i)+'_'+str(j)+'_.pt', map_location='cpu')

    print('SAVED',i)
    torch.save(subbatch, 'x_airline_bert_'+str(i)+'__.pt')        
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (i, (end - start)))

for i in range(0,length,step):
  orgnizefile(new_sentences[i:i+step],i+1)
>>>>>>> 3c89b31d140a946a42b330f277b4129a265c1345
