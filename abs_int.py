#!/usr/bin/env python
# coding: utf-8
"""
@author: FrankLawless

"""

import numpy as np
from math import dist

def trajectoryPlanning(refPosition, velocity, interp_resolution, h):
    index = []
    position_out = np.empty((1,2))
    
    position_out[0,:] = refPosition[0,:]   
    
    for a in range(len(refPosition)):
        points = np.empty((2,2))
        if (len(refPosition) > 1):
            point1 = refPosition[0,:]
            point2 = refPosition[1,:]
            points[0,:] = point1
            points[1,:] = point2
    
            refPosition = np.delete(refPosition, 0, axis=0)
            
            points = interpolate(points,interp_resolution,h)
        else:
            points = np.delete(points, [0,1], axis=0)
            
        position_out = np.append(position_out,points,axis=0)
        
    ptr_1 = 0
    ptr_2 = 1

    while ptr_2 in range(len(position_out)-1):
        
        while dist(position_out[ptr_1,:],position_out[ptr_2,:]) < velocity*h:
            index.append(ptr_2)
            ptr_2 += 1
            if ptr_2 > (len(position_out) -1):
                break

        ptr_1 = ptr_2
        ptr_2 += 1
        
        if ptr_2 == (len(position_out)-1):
            if dist(position_out[ptr_1,:],position_out[ptr_2,:]) < velocity*h:
                index.append(ptr_2)
                

    position_out = np.delete(position_out, index, axis=0) 
    
    return position_out

def interpolate(arg_Points,interp_resolution,h):
    points = np.empty((2,2))
    points = arg_Points
    point1 = points[0,:]
    point2 = points[1,:]
    points = np.delete(points, 0, axis = 0)
    
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    
    distance = dist(point1,point2)
    step_size = distance/(interp_resolution*h)
    step_size = int(step_size)
    interval0 = np.linspace(0,distance,step_size)
    interval1 = interval0[1:len(interval0)-1]

    x_out = []
    y_out = []

    for i in interval1:
        x_n = x1 + (i/distance)*(x2 - x1)
        y_n = y1 + (i/distance)*(y2 - y1)
        x_out.append(x_n)
        y_out.append(y_n)

    column0 = np.array(x_out)
    column1 = np.array(y_out)
    column0 = np.expand_dims(column0,axis = 1)
    column1 = np.expand_dims(column1,axis = 1)
    interpolated_Points = np.concatenate((column0,column1),axis=1)
    
    new_points = np.insert(points,0,interpolated_Points,axis =0)

    return new_points
