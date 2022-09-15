#!/usr/bin/env python
# coding: utf-8
"""
@author: FrankLawless
"""
import numpy as np

def controller(x, xVelocity, yVelocity, positionX, positionY, m, k, D): 
    
    J = np.zeros((2,2))
    J[0,0] = np.cos(x[2,:]); J[0,1] = -D*np.sin(x[2,:])
    J[1,0] = np.sin(x[2,:]); J[1,1] =  D*np.cos(x[2,:])
    
    v_feedback = np.zeros((2,1))
    
    error = np.sqrt((positionX-x[0,:])**2+(positionY-x[1,:])**2)

    if error > m:
        v_feedback[0,:] = xVelocity - m*k*(x[0,:] - positionX)/error
        v_feedback[1,:] = yVelocity - m*k*(x[1,:] - positionY)/error
        
    else:
        v_feedback[0,:] = xVelocity - k*(x[0,:] - positionX)
        v_feedback[1,:] = yVelocity - k*(x[1,:] - positionY)
    
    J = np.linalg.inv(J)
    desired_control = np.matmul(J, v_feedback)
      
    v = desired_control[0,:]
    w = desired_control[1,:]
    
    return v, w

def dynamics(x,v,w, D):
    M = np.zeros((3,2))
    M[0,0] = np.cos(x[2,:]); M[0,1] = -D*np.sin(x[2,:])
    M[1,0] = np.sin(x[2,:]); M[1,1] = D*np.cos(x[2,:])
    M[2,0] = 0;              M[2,1] = 1
    
    arr = np.zeros((2,1))
    arr[0,0] = v; arr[1,0] = w
    
    dot_x = np.matmul(M, arr)
    
    return dot_x
