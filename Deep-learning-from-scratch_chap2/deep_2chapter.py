# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:27:40 2024

@author: Jeongmin
"""


import numpy as np

def AND(x1,x2):
    x = np.array([x1, x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1,s2)
    return y

def HALFADDER(x1, x2):
    c = AND(x1,x2)
    s = XOR(x1,x2)
    return c,s


def FULLADDER(x1, x2, ci):
    s1 = HALFADDER(x1,x2)[1]
    c1 = HALFADDER(x1,x2)[0]
    s2 = HALFADDER(s1,ci)[1]
    c2 = HALFADDER(s1,ci)[0]
    co = OR(c1,c2)
    return co,s2

print(FULLADDER(0,0,0))
print(FULLADDER(0,0,1))
print(FULLADDER(0,1,0))
print(FULLADDER(0,1,1))
print(FULLADDER(1,0,0))
print(FULLADDER(1,0,1))
print(FULLADDER(1,1,0))
print(FULLADDER(1,1,1))



























