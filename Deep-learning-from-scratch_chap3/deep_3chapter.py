# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:03:42 2024

@author: Jeongmin
"""

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=np.int32)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

"""x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
y2 = sigmoid(x)
y3 = relu(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.plot(x,y3)
plt.ylim(-0.1,5.1)
plt.show()"""

"""A = np.array([[1,2],[3,4]])
print(A.shape)
B = np.array([[5,6],[7,8]])
print(B.shape)

print(np.dot(A,B))

A = np.array([[1,2,3],[4,5,6]])
print(A.shape)
B = np.array([[1,2],[3,4],[5,6]])
print(B.shape)

print(np.dot(A,B))"""

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

A1 = np.dot(X,W1)+B1
Z1 = sigmoid(A1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

A2 = np.dot(Z1,W2)+B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3)+B3
Y = identity_function(A3)

print(A3)

















