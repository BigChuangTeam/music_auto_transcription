#!/usr/bin/env python
# coding: utf-8
import os, sys
for i in range(1,89):
     path = str(i)
     os.makedirs(path)
        
tolist = os.listdir("/Users/bigwhite/Documents/大创/dataset/dataset")
import shutil
alllist=os.listdir("/Users/bigwhite/Documents/大创/音频特征提取/digits")
tolist = os.listdir("/Users/bigwhite/Documents/大创/dataset/dataset")
for i in alllist:
    aa,bb=i.split(".")
    for j in tolist:
        if(aa == (str(0) + j)):
            oldname= '/Users/bigwhite/Documents/大创/音频特征提取/digits/'+aa+"."+bb
            newname="/Users/bigwhite/Documents/大创/dataset/dataset"+"/"+j+'/'+aa+"."+bb
            shutil.copyfile(oldname,newname)
import librosa
import numpy as np
 
def add_noise(data):
    wn = np.random.normal(0,1,len(data))
    data_noise = np.where(data != 0.0, data.astype('float64') + 0.005 * wn, 0.0).astype(np.float32)
    return data_noise
path = '/Users/bigwhite/Documents/大创/dataset/dataset/'
for i in range(1,89):
    for j in range(1,100):
        data, fs = librosa.core.load(path + str(i) +'/'+'0'+str(i)+ '.wav')
        data_noise = add_noise(data)
        librosa.output.write_wav(path+ str(i) +'/'+ str(i)+'_' + str(j)+'.wav', data_noise, fs)
from Features import*
def wav2feature(fpath):
    fts = Features()
    maxlen = 10000
    seqlen = 200
    f = fts.MIX(fpath)
    if(f.shape[0]<seqlen):
        t = np.zeros(((seqlen-f.shape[0]),f.shape[1]))
        f = np.vstack((f,t))
    f = f.reshape(1,f.shape[0]*f.shape[1])
    return f

path =  '/Users/bigwhite/Documents/大创/dataset/dataset/'
f = wav2feature(path+ str(1) +'/'+ str(1)+'_' + str(1)+'.wav')
print(f.shape)

fts = Features()
train_x = []
train_y = []
maxlen = 10000
path =  '/Users/bigwhite/Documents/大创/dataset/dataset/'
for i in range (1,81):
    for j in range(1,100):
        f = wav2feature(path+ str(i) +'/'+ str(i)+'_' + str(j)+'.wav')
        train_x.append(f)
        train_y.append(i)
print(train_x.shape,train_y.shape)

len(train_x)

file= open('/Users/bigwhite/Documents/大创/dataset/train_x.txt', 'w')  
for fp in train_x:
        file.write(str(fp))
        file.write('\n')
file.close()
file= open('/Users/bigwhite/Documents/大创/dataset/train_y.txt', 'w')  
for fp in train_y:
        file.write(str(fp))
        file.write('\n')
file.close()

file=open('/Users/bigwhite/Documents/大创/dataset/train_x.txt', 'r')
list_read = file.readlines()
print(len(list_read))

list_read[0]



train_x2 = np.array(train_x)


train_y2 = np.array(train_y)


train_x2.dtype


train_y2.dtype

np.save('train_x.npy',train_x2)
np.save('train_y.npy',train_y2)