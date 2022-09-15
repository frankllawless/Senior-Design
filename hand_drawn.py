#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:10:55 2022

"""
from abs_int import trajectoryPlanning #int/abs function
from control import controller #linear and non-linear control
from control import dynamics #diff. drive robot 
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import sys
import pygame
import pygame.camera 
import rospy

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String

rospy.init_node('velocity_publisher')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)
move = Twist() 
# In[1]: adjustable parameters
D = .047625
k = 1
m = .5
h = .1
speed = .15
interp_resolution = .001

v_max = .22
w_max = 2.84

head_int = 0

Fullscreen = False
travel_distance_y = .7
travel_distance_x = .7
# In[2]:
pygame.init()
pygame.camera.init()
infoObject = pygame.display.Info()
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
red = (255, 0, 0)

if Fullscreen== True:
	screen_x = 1400
	screen_y = 700

else:
	screen_x = 800
	screen_y = 800
	
screen = pygame.display.set_mode((screen_x, screen_y))
pygame.display.set_caption('Trajectory Planning') 

pygame.draw.line(screen, red, (screen_x/4, 5), (3*screen_x/4, 5), width = 4)
font = pygame.font.Font('freesansbold.ttf', 15)
text = font.render("{} m".format(travel_distance_x/2), True, red)
textRect = text.get_rect()
textRect.center = (screen_x/2, 19)
screen.blit(text, textRect)

pygame.draw.line(screen, red, (5, screen_y/4), (5, 3*screen_y/4), width = 4)
text = font.render("{} m".format(travel_distance_y/2), True, red)
textRect.midleft = (19, screen_y/3)
screen.blit(text, textRect)

rgb_surface = pygame.Surface((25,25))
yuv_surface = pygame.camera.colorspace(rgb_surface,"YUV")
screen.blit(yuv_surface,(10, screen_y/2-25))	

pygame.display.flip()
clock = pygame.time.Clock()
working = True
position_x = []
position_y = []

while (working == True):
    quitting = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            working = False
        elif quitting[pygame.K_SPACE]:
            pygame.quit()
            working = False
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:  # Left mouse button down.
                last_position = (event.pos[0] - event.rel[0], event.pos[1] - event.rel[1])
                position_x.append(last_position[0])
                position_y.append(-last_position[1])
                pygame.draw.line(screen, (255,255,255), last_position, event.pos, 1)
    if working == True:
        pygame.display.update()
        clock.tick(120)  # Limit the frame rate to 120 FPS.
if not position_x: 
    sys.exit("Trajectory planning was closed")
#position_y = np.flip(position_y)
# turns the 2 different x and y array's into one 2xn matrix
x_y = list(zip(position_x,position_y))
# saves matrix into csv file for use as input in trajectory traversal
np.savetxt("data.csv", x_y, delimiter = ',') 

file = open('data.csv') # opens csv file, or any other data file like excel
position = np.loadtxt(file,delimiter=",") # creates positional array from input file

position[:,0] = position[:,0]/screen_x*travel_distance_x
position[:,1] = (position[:,1]+screen_y/2)/screen_y*travel_distance_y
# In[3]:
x_vel = []
y_vel = []

position = trajectoryPlanning(position, speed, interp_resolution, h)

init = (position[0,0], position[0,1], head_int) # creates initial points from input file

T = len(position) 
for t in range(0,T-1):
    x_vel.append((position[t+1,0] - position[t,0])/h) # finds velocity from position array on x plane
    y_vel.append((position[t+1,1]- position[t,1])/h) # finds velocity from position array on y plane

TimeSpan = 20

state = np.zeros((3,int(T))) # current state 
state[:,0] = [init[0]-D*np.cos(init[2]), init[1]-D*np.sin(init[2]),init[2]] # current state (from initial conditions)
# In[4]:
xb = np.zeros((1,T))
xb[:,0] = init[0]
yb = np.zeros((1,T))
yb[:,0] = init[1]
theta = np.zeros((1,T))
theta[:,0] = init[2]
# In[5]:
linear_vel = [] 
angular_vel = []
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
# In[6]:
if Fullscreen == True:
	plt.xlim([0, travel_distance_x])
	plt.ylim([-travel_distance_y/2, travel_distance_y/2])
else:	
	plt.xlim([0, travel_distance_x])
	plt.ylim([-travel_distance_y/2, travel_distance_y/2])    
    
l = []
for g in range(0,T-1):
    l.append(g*h) 

plt.scatter(xb, yb,color='blue',linewidth=1,label='Pen point on Robot')
plt.plot(state[0,:],state[1,:],color='red',label='robot centroid',linewidth=3)
plt.plot(position[:,0], position[:,1],color='green', label='reference trajectory',linewidth=3)
plt.legend(loc='upper right')
plt.grid()
plt.show()
# In[7]:
plt.plot(l,angular_vel, )
plt.plot(l,linear_vel)
plt.title('Velocities')
plt.legend(["Angular (w)", "Linear (v)"])
plt.show()
