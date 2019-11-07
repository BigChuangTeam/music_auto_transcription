# -*- coding: utf-8 -*-
"""
@author：王建峰
Student ID: 201328013727073
"""
"""
The area of moudel
"""
import struct                            
import wave
import matplotlib
import tkFileDialog
import sys
import  varia 
from  pylab import *
import pylab as pl
from scipy.linalg import eigvals, pinv, svd, eig
from scipy import angle, arange,  mat, pi, zeros
from numpy.matlib import repmat, linspace
from Tkinter import *
from matplotlib.pyplot import figure, plot, title, subplot, ylim, stem, grid, show
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import numpy as np
###########
"""
"""
root = Tk()
root.geometry('750x550+0+0')
root.title('Music Distinguish')
"""
The area of function
"""
def Openfile():
   filename = tkFileDialog.askopenfilename(initialdir = 'E:')
   fw = wave.open(filename, 'rb')
   varia.params = fw.getparams()
   print(varia.params)
   varia.nchannels, varia.sampwidth, varia.framerate, varia.nframes = varia.params[:4]
   strData = fw.readframes(varia.nframes)
   varia.waveData = np.fromstring(strData, dtype=np.int16)
   varia.waveData = varia.waveData[::varia.div]
   fw.close()
##
def Rxx(x):
    
    m = len(x)  
    temp = mat(arange(0, m))
    indices = repmat(temp.T, 1, m) - repmat(temp, m, 1)
    acsamples=(ifft(fft(x,2*len(x)+1)*fft(x,2*len(x)+1).conjugate())).real
    return acsamples[indices + m - 1] / m
##
def Filter(dat):
    length = len(dat)
    for i in xrange(length-2):
        if (abs(dat[i+1]-dat[i])>20) & (abs(dat[i+2]-dat[i+1])>20):
            dat[i+1]=dat[i]
        if dat[i+1]>1500:
            dat[i+1]= dat[i]
    if dat[len(dat)-2]>1500:
        dat[len(dat)-2]=dat[len(dat)-3]
    if dat[len(dat)-1]>1500:
        dat[len(dat)-1]=dat[len(dat)-2]
    return dat
##
def Divide(Dat):
    Number = len(Dat)
    for j in xrange(Number):
        if  Dat[j]<=150: 
            Dat[j]=1
        if (Dat[j]>150)&(Dat[j]<=200):
            Dat[j]=2
        if (Dat[j]>200)&(Dat[j]<=250):
            Dat[j]=3
        if (Dat[j]>250)&(Dat[j]<=300):
            Dat[j]=4
        if (Dat[j]>300)&(Dat[j]<=355):
            Dat[j]=5
        if (Dat[j]>355)&(Dat[j]<=410):
            Dat[j]=6
        if (Dat[j]>410)&(Dat[j]<=465):
            Dat[j]=7
        if (Dat[j]>465)&(Dat[j]<=525):
            Dat[j]=7
        if  Dat[j]>525: 
            Dat[j]=8
    return Dat
##
def Lib_Match():
    data1 = np.fromfile("pieces of music.bin", dtype=np.float)
    flag = 3
    lib_mus =''
    Match_output = ''
    for j in range(3):
        lib_mus = varia.lib_name[j]
        data2 = np.fromfile(lib_mus, dtype=np.float)
        L1=len(data1)
        L2=len(data2)
        data11=list(data1[0:L1])
        data22=list(data2[0:L2])
        for i in range(L2-L1+1):
            dd=data22[i:i+L1]
            a=cmp(data11,dd)
            if a==0:
                flag = j
                print flag
                break
    print flag
    Match_output = varia.ture_name[flag]
    print Match_output
    output = StringVar()
    entry = Entry(root,width=25,textvariable = output)
    output.set(Match_output)
    entry.place(x = 260,y = 440,anchor = NW)
    entry['state'] = 'readonly'
        
##
def Match_Music():
    print len(varia.waveData)
    tim = np.arange(0, len(varia.waveData)) * (1.0 / varia.framerate)*varia.div
    frameSize =960
    #frameSize =500
    print 'the framesize is:',frameSize
    lappoints = 0
    frampersec = varia.params[2]/(frameSize-lappoints)      
    totframe  = int(len(varia.waveData)*(1.0 / varia.framerate)*frampersec)
    print 'the total frame:',totframe
    FRE = zeros(totframe)
    
    Image_time = f1.add_subplot(2,1,1)
    Image_time.clear()
    Image_time.plot(tim,varia.waveData)
    title('the image of time-energy domain')
    for i in range(totframe):
        idx1 = (frameSize-lappoints)*i
        idx2 = frameSize*(i+1)-i*lappoints
        r = Rxx(varia.waveData[idx1:idx2])
        U, S, Vh = svd(r,full_matrices=False)
        eigen_psd_sig = zeros(frameSize)
        for j in xrange(4):
            vsig =  U[:, j]
            eigen_psd_sig = eigen_psd_sig + (S[j])*abs(np.fft.fft(vsig, frameSize))
        eigen_psd_sig = eigen_psd_sig
        FRE[i]=np.argmax(eigen_psd_sig)*varia.framerate/(frameSize*varia.div)
##
    Filter_FRE= Filter(FRE)   
    music_freq = f1.add_subplot(2,1,2)
    music_freq.clear()
    tim1 = linspace(0,len(varia.waveData)*(1.0 / varia.framerate)*varia.div, totframe)
    music_freq.plot(tim1,Filter_FRE)
    title('the image of time-frequ domain')
    photo1.draw() 
    Divide_FRE = Divide(Filter_FRE)
    Divide_FRE.tofile("pieces of music.bin")
    FRTI = np.fromfile("pieces of music.bin", dtype=np.float)
    Lib_Match()
    print  'a piece of music is:' ,FRTI
    print " The piece of music analysis finished"
"""
"""    
"""
GUI piece 
including two buttons
one frame picture
two text
"""
#two picture
f1= figure( figsize=(14,5.6),dpi=50)
photo1 = FigureCanvasTkAgg(f1,root)
photo1.get_tk_widget().place(x = 10,y = 40,anchor = NW)

#two button
Open_file = Button(root,width=4,height=1, text = 'Open',     #button1
                   bg='yellow', anchor = 'n',command=Openfile)
Open_file.place(x = 12,y = 2,anchor = NW)
Match_music = Button(root,width=4,height=1, text = 'Match',     #button2
                   bg='yellow', anchor = 's',command=Match_Music)
Match_music.place(x = 70,y = 2,anchor = NW)
#message
Library=Message(root,text = '音乐库：\n  1.红豆 \n  2.斯塔基 \n  3.我要飞翔',
                width=250,aspect=2,bg='white')
Library.place(x = 12,y = 400,anchor = NW)
Match_outcome=Message(root,text = '乐谱识别结果：',
                width=250,aspect=2,bg='white')
Match_outcome.place(x = 260,y = 400,anchor = NW)

##output
#Match_output = '玉面小飞龙'
#output = StringVar()
#entry = Entry(root,width=25,textvariable = output)
#output.set(Match_output)
#entry.place(x = 260,y = 440,anchor = NW)
#entry['state'] = 'readonly'

"""
The end of GUI
"""
root.mainloop()



   
