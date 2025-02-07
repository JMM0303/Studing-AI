# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:42:22 2024

@author: Jeongmin
"""

import numpy as np

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]

def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

"""print(sum_squares_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(sum_squares_error(np.array(y), np.array(t)))"""

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

"""y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))"""

import matplotlib.pylab as plt

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

"""x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
z = numerical_diff(function_1,5)*x - numerical_diff(function_1,5)
z2 = numerical_diff(function_1,10)*x - 1

print(numerical_diff(function_1,5))
print(numerical_diff(function_1,10))

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.plot(x,z)
plt.plot(x,z2)
plt.show()"""

def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))





















