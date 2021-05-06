# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:26:40 2021

@author: chban
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread

my_image = imread('방창현.png')

plt.imshow(my_image)
plt.show()

