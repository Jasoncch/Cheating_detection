from cmath import nan
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sympy import false
import glob
import warnings

def get_movement(directory,frame_count):
    start = len(glob.glob(directory+"/"+"*.png"))-frame_count
    if start<0:
        start=0
    #print("frame=",frame_count)
    first_frame = frame = cv2.imread(directory+"/"+str(start)+".png")
    #print("first=",directory+"/"+str(start)+".png" )
    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mag = np.zeros_like(first_frame)
    sum= 0
    movement = []
    for i in range(start,len(glob.glob(directory+"/"+"*.png"))):
        if(i%2==0):
            frame = cv2.imread(directory+"/"+str(i)+".png")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = cv2.normalize(magnitude, None, 0, 100, cv2.NORM_MINMAX,dtype=cv2.CV_32F)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    avg=np.nanmean(mag)
                except RuntimeWarning:
                    avg=0
            movement.append(avg) 
            prev = gray       
    #print("movement=",movement)
    with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    ans =  np.nanmean(movement)
                except RuntimeWarning:
                    ans=0  
    #print ("total= ", np.mean(movement))
    return ans
