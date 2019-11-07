
# coding: utf-8

# In[24]:


import librosa
import librosa.display
import numpy as np
import math
import sys
import wave
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from Features import *
EnergyThreshold = -35   # Threshold of remained frames' energy
Duration = 10           # above or blow threshold for Duration
fts = Features()
class Template :
    def __init__(self, word):
        self.word = word
        self.loaded = False
    def load(self, filename,feature_type):
        y, sr = librosa.load(filename, sr=None)
        y = fts.standardization(y)
        if feature_type =='CQT':
            ft = fts.CQT(filename)
        elif feature_type =='MFCC':
             ft = fts.MFCC(filename)
        elif feature_type =='STFT':
             ft = fts.STFT(filename)
        elif feature_type == 'MIX':
             ft  = fts.MIX(filename)
        words = cut_words(y, ft)
        self.feature = words[0]
  #      print(len(words))

    def DTWscore(self, out):
        cur = 0
        dp = np.zeros([2, len(self.feature)])
        for i in range(len(out)):
            for j in range(len(self.feature)):
                cur = i % 2
                delta = dp[1 - cur][j]
                if j > 0: delta = min(delta, dp[1 - cur][j - 1])
                if j > 1: delta = min(delta, dp[1 - cur][j - 2])
                dp[cur][j] = dis(out[i], self.feature[j]) + delta
        best = dp[cur][0]
        for i in range(len(self.feature)):
            best = min(best, dp[cur][i])
        return best
    # judge if x is above threshold (return 1) or blow threshold (return -1)
def sig(x):
    if x > EnergyThreshold: return 1.0
    else: return -1.0

def combine(seq):
    ans = []
    for i in range(len(seq)):
        if i != 0 and i != len(seq) - 1             and seq[i][1] - seq[i][0] < Duration and seq[i][2] < 0:
            seq[i][2] = 1.0
    l = r = 0
    while l < len(seq):
        f = seq[l][2]
        r = l
        while r + 1 < len(seq):
            r += 1
            t = seq[r][2]
            if (t * f < 0):
                r -= 1
                break
        r += 1
        ans.append([seq[l][0], seq[r - 1][1], f])
        l = r
    return ans


def cut_words(y, mfccs):
        # pdb.set_trace()
    words = []
    seq = []
    ene = fts.energy(y)
    l = r = 0
    while l < len(ene):
        f = sig(ene[l])
        r = l
        while r + 1 < len(ene):
            r += 1
            t = sig(ene[r])
            if (t * f < 0):
                r -= 1
                break
        r += 1
        # pdb.set_trace()
        seq.append([l, r, f])
        l = r
    seq = combine(seq)
    for i in range(len(seq)):
        # print(seq[i])
        if seq[i][2] > 0 and seq[i][1] - seq[i][0] >= Duration:
            words.append(mfccs[seq[i][0]:seq[i][1]])
    return words

def load_input(filename,feature_type):
    print('Load file: ' + filename)
    y, sr = librosa.load(filename, sr=None)
    y = fts.standardization(y)
    if feature_type =='CQT':
            ft = fts.CQT(filename)
    elif feature_type =='MFCC':
            ft = fts.MFCC(filename)
    elif feature_type =='STFT':
            ft = fts.STFT(filename)
    elif feature_type == 'MIX':
            ft = fts.MIX(filename)
    # pdb.set_trace()
    words = cut_words(y, ft)
    return words

# load templates
def initialization():
    Tems = []
    for i in range(1, 84):         # enum digit
        path = './digits/' + str(0) + str(i)  + '.wav' #your source path
            # print(path)
            print(i)
            tem = Template(str(i))
            tem.load(path,'MIX')
            Tems.append(tem)
            
    print('Initialization Finished!')
    return Tems

# find the best matching word
def best_word(Tems, word):
    ans = ''
    best = 1e10
    # for tem in Tems:
    for i in range(0, 83):         # enum digit
        val = 0
        wd = Tems[i].word
        tem = Tems[i]
            # pdb.set_trace()
        val = tem.DTWscore(word)
        if val < best:
            best = val
            ans = wd
    return ans
def dis(A, B):
    sum = 0
    for i in range(len(A)):
        d = (A[i] - B[i])
        sum += d * d
    return math.sqrt(sum)

