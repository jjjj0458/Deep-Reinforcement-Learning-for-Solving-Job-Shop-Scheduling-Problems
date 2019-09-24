# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:54:26 2018

@author: banana
"""
import numpy as np
import matplotlib.pyplot as plt
#ppp = timee[0:100]
#ppp.append(1270)
#ppp.append(1243)
#ppp.append(1270)
#ppp.append(1270)
#ppp.append(1222)
#
#ppp.sort(reverse = True)

plt.figure(figsize=(10,5))
plt.plot(ppp , label = "DRL")
plt.plot(np.repeat(1272,100),label = "Simple Rule")
plt.plot(np.repeat(1222,100),label = "Optimal")
plt.xlabel("Episode")
plt.ylabel("Makespan")
plt.title("Instance La11 (20x5)")
plt.legend()
plt.show()


import pickle
with open("pic.txt", "wb") as fp:   #Pickling
    pickle.dump(ppp, fp)
 
with open("pic.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

b