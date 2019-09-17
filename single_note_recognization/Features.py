import librosa
import librosa.display
import numpy as np
import math
import sys
import wave
import time
'''
def __init__(self):
    self.loaded = False'''

def standardization(y):
    ans = y
    best = 0
    for smp in y:
        best = max(best, abs(smp))
    for i in range(len(y)):
        ans[i] = y[i] / best
    return ans

def energy(arr):
    ans = np.zeros(int( (len(arr) + 511) / 512 ))
    i = 0
    while 512 * i < len(arr):
        sum = 0
        for j in range(512 * i, min(512 * i + 512, len(arr))):
            sum += arr[j] * arr[j]
        if (abs(sum) < 0.00001): ans[i] = -60.0
        else: ans[i] = 10 * math.log(sum)
        i += 1
    return ans

def MFCC(fpath):
    MFCCdim = 12
    y, sr = librosa.load(fpath)
    y = standardization(y) #归一化 可不使用
    ene = energy(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCCdim).T
    mfccs = np.hstack((mfccs,ene.reshape(-1,1)))
    mfccs_delta = np.zeros(mfccs.shape)
    for i in range(mfccs.shape[0] - 1):
        mfccs_delta[i] = mfccs[i+1] - mfccs[i]
    mfccs_delta[-1] = mfccs[-1]
    mfccs = np.hstack((mfccs,mfccs_delta))
        # print(mfccs.shape)
    return mfccs

def CQT(fpath):
    y, sr = librosa.load(fpath)
    #    chroma_cq = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cq = chroma_cq.T
        #print(chroma_cq.shape)
    return chroma_cq

def STFT(fpath):
    y, sr = librosa.load(fpath)
    stft = librosa.feature.chroma_stft(y=y, sr=sr)
    #librosa.display.specshow(stft, y_axis='chroma')
    stft = stft.T
    #print(stft.shape)
    return stft

def MIX(fpath):
    y1 = MFCC(fpath)
    y2 = CQT(fpath)
    y3 = STFT(fpath)
    MIX = np.hstack((y1,y2))
    MIX = np.hstack((MIX,y3))
    #print(MIX.shape)
    return MIX
    
def wav2feature(fpath):     
    maxlen = 10000     
    seqlen = 200     
    f = MIX(fpath)     
    if(f.shape[0]<seqlen):         
        t = np.zeros(((seqlen-f.shape[0]),f.shape[1]))         
        f = np.vstack((f,t))     
    f = f.reshape(1,f.shape[0]*f.shape[1])    
    return f  
