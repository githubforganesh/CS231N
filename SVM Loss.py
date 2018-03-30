# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#sri gurave namaha
import numpy as np

#non vectorized function
def L_in(x,y,W):
    delta = 1.0
    num = W.shape[0]
    scores = W.dot(x)
    loss_i = 0
    for i in range(num):
        loss_i += max(0,scores[i] - scores[y] + delta)
    return loss_i


# half vectorised function
def L_ih(x,y,W):
    delta = 1.0
    scores = W.dot(x)
    margins = np.maximum(0,scores - scores[y] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

# fully vectorised form

def L(X,y,W):
    delta = 1.0
    scores = W.dot(X)
    margins = np.maximum(0,scores - scores[y,np.arange(y.shape[0])] + delta)
    margins[y,np.arange(y.shape[0])] = 0
    loss = np.sum(margins)
    return loss

#W 3 classes and 4 dimensions
W = np.array([[11,12,13,14],
             [21,22,23,24],
             [31,32,33,34]])

# 4 dimensions and 5 images
X =  np.array([[100,200,300,400,500],
               [101,201,301,401,501],
               [102,202,302,402,502],
               [103,203,303,403,503]])

y = np.array([1,0,2,1,2])

loss_naive = 0;
for i in range(X.shape[1]):
    loss_naive += L_in(X[:,i],y[i],W)
    
loss_half = 0;
for i in range(X.shape[1]):
    loss_half += L_ih(X[:,i],y[i],W)

loss = L(X,y,W)