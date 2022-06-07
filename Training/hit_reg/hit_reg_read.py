import glob
import cv2
import numpy
import pandas as pd

def get_pixel_vector(image):
   
    pixel_vector = [] 
    pixel_count=0
    x_pos = 291;y_pos = 226
    dist=46
    len =10
    threshold=0.7
    for z in range (2):                         #top left and bot right
        x_pos += z*(dist-1)
        y_pos += z*(dist+1)
        for j in range(1,4):
            x_pos+=(j+1)%2
            y_pos-=j%2
            for i in range (len):         
                b,g,r =image[y_pos+i,x_pos+i]
                pixel_vector.append([b,g,r])       
                #if(j==2):
                    #image[y_pos+i,x_pos+i]=[255,255,255]
    x_pos = 348;y_pos = 226
    for z in range (2):                          #top right and bot left
        x_pos -= z*(dist-1)
        y_pos += z*(dist+1)
        for j in range(1,4):
            x_pos-=(j+1)%2
            y_pos-=j%2
            for i in range (len):
                b,g,r =image[y_pos+i,x_pos+i]
                pixel_vector.append([b,g,r])    
    return pixel_vector

for i in range (2):
    if i == 0 :
        directory = 'Training/hit_reg/training_data/hit_data'
    else:
        directory = 'Training/hit_reg/training_data/no_hit_data'   
    hit_frames = []
    data_vector = []
    print("reading:",directory)
    for file in glob.glob(directory + '/'+"*.png"):
        hit_frames.append(file)
    for frame in hit_frames:
        img= cv2.imread(frame)
        pixel_vector = get_pixel_vector(img)
        pixel_vector = numpy.asarray(pixel_vector)
        data_vector.append(pixel_vector.ravel())
    print(data_vector)
    data = numpy.asarray(data_vector)
    if i ==0:
        numpy.savetxt("Training/hit_reg/training_data/no_hit_data.csv", data, delimiter=",")
    else:
        numpy.savetxt("Training/hit_reg/training_data/hit_data.csv", data, delimiter=",")
        