
# coding: utf-8

# In[7]:


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
from Template import *


# In[ ]:


class Menu:
    def __init__(self):
        self.Tems = initialization()
        self.ops = {
            "0": self.__quit,
            "1": self.__solve,
            "2": self.__modify_debug
        }

    def __display(self):
        print(
            '-----------------------\n' +
            '0. quit\n' +
            '1. load file\n' +
            '2. turn on/off debug\n'
            '-----------------------'
        )

    def __solve(self):
        filename = input('Input filename: ')
        input_words = load_input(filename,'MIX')
        for word in input_words:
            ans = best_word(self.Tems, word)
            print(ans, end='')
            sys.stdout.flush()
        time.sleep(0.5)
        print('')

    def __modify_debug(self):
        global debug
        ori = debug
        debug = ~debug
        print('From ' + isTrue(ori) + ' to ' + isTrue(debug))

    def run(self):
        while True:
            self.__display()
            op = input('>>> ')
            op = str(op).strip()
            action = self.ops.get(str(op))
            if action:
                action()
            else:
                print('op [%s] is illegal!' % op)

    def __quit(self):
        print('Goodbye!')
        sys.exit()

def main():
    menu = Menu()
    menu.run()

if __name__ == '__main__':
    main()

