#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работы №3
# ## Проектирование цифрового фильтра и исследование его характеристик.
# ***

# ### 1. Подключение библиотек.

# In[68]:


import numpy as np
import laba1, laba2
import matplotlib.pyplot as plt
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Получение реальной ЭКГ.

# In[69]:


from scipy.misc import electrocardiogram
ECG = electrocardiogram()


# ### 3. Создание объектa рекурсивного фильтра.

# In[70]:


class RecursiveFilter:
    def __init__(self, f_c):
        x = np.exp(-2*np.pi*f_c)
        self.a0 = (1 - x)**4
        self.b1 = 4 * x
        self.b2 = -6*x ** 2 
        self.b3 = 4*x ** 3
        self.b4 = -x ** 4
        
    def filter_(self, sig):
        sig_size = sig.x1.size
        y = np.zeros(sig_size)
        x = sig.y1

        for i in range(sig_size):
            y[i] = self.a0*x[i] + self.b1*y[i - 1] + self.b2*y[i - 2] + self.b3*y[i - 3] + self.b4*y[i - 4]  
        return laba2.Signal(y)


# ### 4. Изучение влияния скользящего усредняющего фильтра на различные виды шумов.

# In[71]:


ecg = laba1.Ecg(cycles=1)
wn = laba1.WhiteNoise(deviation=0.1, size=ecg.size())
hn = laba1.HarmNoise(a_hn=0.05, f_d=5, f_hn=60, n_sin=1888)


# * #### Влияние на ECG

# In[72]:


ecg.print_()
ecg_new = laba1.AveragingFilter(3).filter_(ecg)
ecg_new.print_()


# * #### Влияние на White Noise 

# In[73]:


wn.print_()
wn_new = laba1.AveragingFilter(3).filter_(wn)
wn_new.print_()


# * #### Влияние на Harm Noise 

# In[74]:


hn.print_()
hn_new = laba1.AveragingFilter(3).filter_(hn)
hn_new.print_()


# ### 5. Изучение влияния Sinc-фильтра на различные виды шумов.

# In[75]:


sinc = laba2.sinc_func(0.05, 100)
sinc.print_()


# * #### Влияние на White Noise 

# In[76]:


conv = laba1.Convolution(sinc).signals_convolution(wn)
wn.print_()
conv.print_()


# * #### Влияние на Harm Noise 

# In[77]:


hn = laba1.HarmNoise(a_hn=0.5, f_d=5, f_hn=60, n_sin=2000)
conv = laba1.Convolution(sinc).signals_convolution(hn)
hn.print_()
conv.print_()


# ### 6. Фильтрация сигнала с помощью рекурсивного фильтра.

# In[78]:


ecg.print_()
ecg_filtered = RecursiveFilter(0.2).filter_(ecg)
ecg_filtered.print_()


# ***
# ### Вывод: Изучили особенности реализации алгоритмов цифровой обработки сигналов. Исследовали скользящий усредняющий фильтр, sinc-фильтр и рекурсивный фильтр НЧ.
