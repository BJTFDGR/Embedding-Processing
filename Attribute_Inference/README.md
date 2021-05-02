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
2. pythonrun_airline_bert_v3.py: 

file: get embeddings of sentences from airline reviews that contain 'salad'
input: airline.csv
output: 'x_saladairline_bert_'+str(i)+'__.pt'
 'y_saladairline_bert'+str(i)+'.pt'

---

3. pythonrun_nosalad_bert.py: 

file: get embeddings of sentences from food-review that not contain 'salad'
input: nosaladreview.csv
output: 'x_nosalad_bert'+str(i)+'.pt'
'y_nosalad_bert'+str(i)+'.pt'

---

4. pythonrun_salad_bert.py: 

file: get embeddings of sentences from food-review that contain 'salad'
input: saladreview.csv
output: 'x_salad_bert'+str(i+1)+'.pt'
'y_salad_bert'+str(i)+'.pt'


---

2. get_embedding_airline_salad_bert.py

File: Get embedding in paralell
Input: airline reviews
Output: Embeddings for sentences with 'salad' in 
        'x_saladairline_bert_'+str(i)+'_'+str(j)+'_.pt'
        original sentences
        'y_saladairline_bert'+str(i)+'.pt'
Total Number: 365 sentences

--- 

3. bert_recovery_airline_class_2_layer_3_top_1_white.py

Parameter:
        class_n=2/10: binary classification setting regrading all other classes as 0 and keyword class as 1
                      10 class setting aim to classify each attribute embedding into its class
        layer-n=3/5:  3 layers MLP with 80 nodes is default setting in orginal paper
                      5 layers MLP with drop out is one complex model since the simple model not performing well
        top_n=1/3:    1 means F-1 accuracy
                      3 means F-3 accuracy, which is implemented myself
File: White box setting 
Input: Only take Embeddings for sentences in citylist and original sentences as input
Label: Counter({9: 2281, 4: 1265, 1: 1173, 7: 1136, 2: 926, 5: 750, 3: 608, 8: 596})  Total 8735 embeddings
Loss func and Optimizer: CrossEntropyLoss and Adam

4. bert_recovery_airline_v2_white_salad_class_2_layer_3_top_1.py:
Parameter: 
        layer_n: 3-layers MLP or 5-layers MLP
File: classifier on whether the embeddings contains keyword 'salad'
Input: Only take Embeddings for sentences in food-review, 2000 sentences with 'salad' and 2000 sentences without 'salad'


