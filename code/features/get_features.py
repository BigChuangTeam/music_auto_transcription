#coding=gbk
import os
import numpy as np
from Features import *

fts = Features()

path = "cut"
files = os.listdir(path)
files = [path + '\\' + f for f in files if f.endswith('.wav')]

mfccs = [fts.MFCC(files[0]).flatten()]
for i in range(1,len(files)):
#for i in range(0,10):
    mfcc = fts.MFCC(files[i])
    if mfcc.shape == (130,26):
        mfccs = np.concatenate((mfccs,[mfcc.flatten()]),axis = 0)
        print(files[i] + "\n")
    else:
        print(files[i] + "\n")
    #print(mfccs)
        
FileName = "mfcc\\02val"
np.save(FileName,mfccs)
