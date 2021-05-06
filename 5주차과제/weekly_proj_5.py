# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:36:51 2021

@author: chban
@description : 
디지털 논리회로에 나오는 전가산기( Full Adder) 를  퍼셉트론을 이용하여 파이썬으로 구현하고 실행결과를 보이시오
(2장에서 배운 논리회로의 구현을 참조하고 이용하시오)
"""

import numpy as np

def AND (x, y):
    z = np.array([x, y])
    a = np.array([0.5, 0.5])
    b = -0.7
    o = np.sum(a*z)+b
    if o <= 0 :
        return 0
    else :
        return 1
    
def NAND (x, y):
    z = np.array([x, y])
    a = np.array([-0.5, -0.5])
    b = 0.7
    o = np.sum(a*z) + b
    if o <= 0:
        return 0
    elif o > 0:
        return 1

def OR(x, y):
    z = np.array([x, y])
    a = np.array([0.5, 0.5])
    b = -0.2
    o = np.sum(a*z) + b
    if o <= 0:
        return 0
    elif o > 0:
        return 1

def XOR(x, y):
    out_1 = NAND(x, y)
    out_2 = OR(x, y)
    o = AND(out_1, out_2)
    return o

def FullAdder(x, y, z):
    out_xor = XOR(x, y)
    out_and = AND(x, y)
    
    SUM = XOR(out_xor, z)
    CARRY = OR(out_and, AND(out_xor, z))
    return CARRY, SUM

# 결과 출력
print('input     output')
print('(x, y, z) (c, s)')
print('----------------')
print('(0, 0, 0)', FullAdder(0, 0, 0))
print('(0, 0, 1)', FullAdder(0, 0, 1))
print('(0, 1, 0)', FullAdder(0, 1, 0))
print('(0, 1, 1)', FullAdder(0, 1, 1))
print('(1, 0, 0)', FullAdder(1, 0, 0))
print('(1, 0, 1)', FullAdder(1, 0, 1))
print('(1, 1, 0)', FullAdder(1, 1, 0))
print('(1, 1, 1)', FullAdder(1, 1, 1))