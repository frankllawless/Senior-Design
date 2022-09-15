#!/usr/bin/env python
# coding: utf-8

from abs_int import trajectoryPlanning #int/abs function
from control import controller #linear and non-linear control
from control import dynamics #diff. drive robot 
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import sys
import rospy

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String

rospy.init_node('velocity_publisher')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)
move = Twist() 

D = .047625

# In[1]: # adjustable parameters
k = 1
m = .5
h = .1
speed = .07
interp_resolution = .001
v_max = .22
w_max = 2.84
head_int = 0
travel_distance_m = 1 # for y
# In[2]:
travel_distance_y = travel_distance_m
travel_distance_x = travel_distance_m

file = open('/home/flawless/catkin_ws/src/senior_design/src/logo.csv') # opens csv file, or any other data file like excel
position_old = np.loadtxt(file,delimiter=",") # creates positional array from input file
position_old = np.transpose(position_old/1000*travel_distance_m)

position = trajectoryPlanning(position_old, speed, interp_resolution, h)

T = len(position)
init = (position[0,0]+D*np.cos(head_int), position[0,1]+D*np.sin(head_int), head_int) # creates initial points for the base-point from input file

state = np.zeros((3,int(T))) # current state 
state[:,0] = [position[0,0], position[0,1], init[2]] # current centroid state 

xb = np.zeros((1,T))
xb[:,0] = init[0]
yb = np.zeros((1,T))
yb[:,0] = init[1]
theta = np.zeros((1,T))
theta[:,0] = init[2]

linear_vel = [] 
angular_vel = []
x_vel = []
y_vel = []
for t in range(0,T-1):
    x_vel.append((position[t+1,0] - position[t,0])/h) # finds velocity from position array on x plane
    y_vel.append((position[t+1,1]- position[t,1])/h) # finds velocity from position array on y plane
# In[3]:
rate = rospy.Rate(1/h)
# loop that brings the tracking together, finds linear and angular velocities as well as calculates the
# centroid of the robot from the pen point of the robot(positional array). This way we can control the robot
# based on a point in front of it, not the centroid itself.
for t in range(0,T-1):
    x = np.zeros((3,1))
    x[0,:] = xb[:,t]
    x[1,:] = yb[:,t]
    x[2,:] = theta[:,t]
        
    v, w = controller(x[:,:], x_vel[t], y_vel[t], position[t,0], position[t,1], m, k, D)
    
    if v > v_max:
        v = v_max
    if v < -v_max:
        v = -v_max
    if w > w_max:
        w = w_max
    if w < -w_max:
        w = -w_max

    linear_vel.append(v)
        
    angular_vel.append(w)
        
    dot_x = dynamics(x,v,w, D)
        
    xb[:,t+1] = xb[:,t] + dot_x[0]*h
    yb[:,t+1] = yb[:,t] + dot_x[1]*h
    theta[:,t+1] = theta[:,t] + dot_x[2]*h
        
    state[0,t+1] = xb[:,t+1] - D*np.cos(theta[:,t+1])
    state[1,t+1] = yb[:,t] - D*np.sin(theta[:,t+1])
    state[2,t+1] = theta[:,t+1]
    
    move.linear.x = v 
    move.angular.z = w
    pub.publish(move)
    
    print(move)
    rate.sleep()
    
move.linear.x = 0
move.angular.z = 0
pub.publish(move)
rospy.is_shutdown() 
print("done")
# In[4]:
l = []
for g in range(0,T-1):
    l.append(g*h) 
    
plt.scatter(xb, yb,color='blue',linewidth=1,label='Pen point on Robot')
plt.plot(state[0,:],state[1,:],color='red',label='robot centroid',linewidth=3)
plt.plot(position[:,0], position[:,1],color='green', label='reference trajectory',linewidth=3)
plt.legend(loc='upper right')
plt.grid()
plt.show()
  
plt.plot(l,angular_vel, )
plt.plot(l,linear_vel)
plt.title('Velocities')
plt.legend(["Angular (w)", "Linear (v)"])
plt.show()