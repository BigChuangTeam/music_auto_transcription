
# coding: utf-8

# In[1]:


import librosa
import librosa.display
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import wave
import time
import pdb


# In[35]:


#MFCC
#INPUT: The audio file path
#OUTPUT: The MFCC array(帧数,MFCCdim) 

MFCCdim = 12
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
    y, sr = librosa.load(fpath)
    y = standardization(y) #归一化 可不使用
    time = np.arange(0, len(y)) * (1.0 / sr)
    plt.subplot(311)
    plt.plot(time, y)
    ene = energy(y)
    t2 = np.arange(0, int( (len(y) + 511) / 512 ) ) * (512.0 / sr)
    plt.subplot(312)
    plt.plot(t2, ene)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCCdim).T
    #print(mfccs)
    plt.subplot(313)
    librosa.display.specshow(mfccs)
    mfccs = np.hstack((mfccs,ene.reshape(-1,1)))
    mfccs_delta = np.zeros(mfccs.shape)
    for i in range(mfccs.shape[0] - 1):
        mfccs_delta[i] = mfccs[i+1] - mfccs[i]
    mfccs_delta[-1] = mfccs[-1]
    mfccs = np.hstack((mfccs,mfccs_delta))
    print(mfccs.shape)
    #plt.show()     #use if you wanna to get the graph
    return mfccs


# In[36]:


#test mfcc

mfccs = MFCC('/Users/bigwhite/Documents/大创/钢琴88键素材/钢琴音色WAV/01-A_-大字2组.wav')


# In[53]:


#CQT
#INPUT: The audio file path
#OUTPUT: The MFCC array(帧数,CQTdim) 
def CQT(fpath):
    y, sr = librosa.load(fpath)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure()

    librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
    plt.title('chroma_cqt')
    plt.colorbar()
    plt.tight_layout()
#    plt.show()
    chroma_cq = chroma_cq.T
    print(chroma_cq.shape)
    return chroma_cq


# In[54]:


#test cqt

cqt = CQT('/Users/bigwhite/Documents/大创/钢琴88键素材/钢琴音色WAV/01-A_-大字2组.wav')


# In[62]:


#STFT
#INPUT: The audio file path
#OUTPUT: The MFCC array(帧数,STFTdim) 
def STFT(fpath):
    y, sr = librosa.load(fpath)
    stft = librosa.feature.chroma_stft(y=y, sr=sr)
# # Use an energy (magnitude) spectrum instead of power spectrogram

#     S = np.abs(librosa.stft(y))
#     stft = librosa.feature.chroma_stft(S=S, sr=sr)

# # Use a pre-computed power spectrogram with a larger frame

#     S = np.abs(librosa.stft(y, n_fft=4096))**2
#     stft = librosa.feature.chroma_stft(S=S, sr=sr)
    librosa.display.specshow(stft, y_axis='chroma')
    plt.title('chroma_stft')
    plt.colorbar()
 #   plt.show()
    stft = stft.T
    print(stft.shape)
    return stft


# In[63]:


stft = STFT('/Users/bigwhite/Documents/大创/钢琴88键素材/钢琴音色WAV/01-A_-大字2组.wav')

