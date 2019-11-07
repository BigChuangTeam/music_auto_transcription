import librosa
import librosa.display
import numpy as np
import math
import sys
import wave
import time
class Features:
    def __init__(self):
        self.loaded = False
    def standardization(self,y):
        ans = y
        best = 0
        for smp in y:
            best = max(best, abs(smp))
        for i in range(len(y)):
            ans[i] = y[i] / best
        return ans
    def energy(self,arr):
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
    def MFCC(self,fpath):
        MFCCdim = 12
        y, sr = librosa.load(fpath)
        y = self.standardization(y) #归一化 可不使用
        ene = self.energy(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCCdim).T
        mfccs = np.hstack((mfccs,ene.reshape(-1,1)))
        mfccs_delta = np.zeros(mfccs.shape)
        for i in range(mfccs.shape[0] - 1):
            mfccs_delta[i] = mfccs[i+1] - mfccs[i]
        mfccs_delta[-1] = mfccs[-1]
        mfccs = np.hstack((mfccs,mfccs_delta))
        print(mfccs.shape)
        return mfccs
    def CQT(self,fpath):
        y, sr = librosa.load(fpath)
        #    chroma_cq = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cq = chroma_cq.T
        print(chroma_cq.shape)
        return chroma_cq
    def STFT(self,fpath):
        y, sr = librosa.load(fpath)
        stft = librosa.feature.chroma_stft(y=y, sr=sr)
        librosa.display.specshow(stft, y_axis='chroma')
        stft = stft.T
        print(stft.shape)
        return stft
    def MIX(self,fpath):
        y1 = self.MFCC(fpath)
        y2 = self.CQT(fpath)
        y3 = self.STFT(fpath)
        MIX = np.hstack((y1,y2))
        MIX = np.hstack((MIX,y3))
        print(MIX.shape)
        return MIX
