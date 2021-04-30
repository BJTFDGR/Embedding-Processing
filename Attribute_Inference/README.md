---

1. get_embedding_airline_bert.py 

File: Get embedding in paralell
Input: airline reviews
Output: Embeddings for sentences in citylist 
        'x_airline_bert_'+str(i)+'__.pt'
        original sentences
        'y_airline_bert'+str(i)+'.pt'
Total Number: Counter({9: 2281, 4: 1265, 1: 1173, 7: 1136, 2: 926, 5: 750, 3: 608, 8: 596})   8735 728
sentences/embeddings

--- 

2. get_embedding_airline_salad_bert.py

File: Get embedding in paralell
Input: airline reviews
Output: Embeddings for sentences with 'salad' in 
        'x_saladairline_bert_'+str(i)+'_'+str(j)+'_.pt'
        original sentences
        'y_saladairline_bert'+str(i)+'.pt'
Total Number: 365 sentences
