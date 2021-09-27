####
#### This is the explaination of twiiter
####

from hashlib import new
import os
from random import sample
dir_path='/home/chenboc1/githubfolder/bookcorpus/books1/epubtxt'

# Show number of books
print(len(os.listdir(dir_path)))

#sentences for each book
booksentence=[]

# Here we get lens: all sentences included in list
files = os.listdir(dir_path) 
for file in sample(files,3000):
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

# Get 800 bookset and its sentences
booksentence.sort(key = lambda i:len(i),reverse=True)
for item in booksentence:item.sort(key = lambda i:len(i),reverse=True)
newbooksentence=booksentence[0:800]

# Save this dataset
def save_list2txt(name,list1):
    filename = open(name, 'w',encoding='UTF-8')
    for value in list1:
        filename.write(str(value))
        filename.write('\n')
    filename.close()

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)            
		print('new folder...') 
	else:
		print("---  There is this folder!  ---")
		
new_folder='/home/chenboc1/githubfolder/bookcorpus/books1/newdataset/'
mkdir(new_folder)
for item in newbooksentence:
    save_list2txt(new_folder+str(newbooksentence.index(item))+'.txt',item)

