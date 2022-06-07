import cv2
import numpy as np
import math
from joblib import dump, load


def check_red(pixel):
    b,g,r=pixel
    #print("r:",r,"g:",g,"b:",b)
    if (r>100 and g<100 and b<100 and r*1.7>b+g and math.isclose(b,g,rel_tol=0.6)):
        return True
    else:
        return False


def get_color_value(image):
    pixel_count=0
    x_pos = 276;y_pos = 210
    dist=70
    len =17
    threshold=0.6
    for z in range (2):                         #top left and bot right
        x_pos += z*(dist-1)
        y_pos += z*(dist+1)
        for j in range(1,4):
            x_pos+=(j+1)%2
            y_pos-=j%2
            for i in range (len):
                if(check_red(image[y_pos+i,x_pos+i])):
                    pixel_count+=1
                    #image[y_pos+i,x_pos+i]=[255,255,255]
    x_pos = 364
    y_pos = 210
    for z in range (2):                          #top right and bot left
        x_pos -= z*(dist-1)
        y_pos += z*(dist+1)
        for j in range(1,4):
            x_pos-=(j+1)%2
            y_pos-=j%2
            for i in range (len):
                if (check_red(image[y_pos+i,x_pos-i])):
                    pixel_count+=1
                    #image[y_pos+i,x_pos-i]=[255,255,255]
    confidence=pixel_count/(len*4*3)
    #print ("total_pixel=",len*4*3,"red_pixel=",pixel_count,"Confidence=",confidence)

    if (confidence>threshold):
        return confidence
    else:
        return False

def check_kill(image):
    svm_clf = load("Models/kill_svm.joblib") 
    if get_color_value(image):
        pixel_vector=get_pixel_vector(image)
        pixel_vector = np.asarray(pixel_vector) 
        pixel_vector = pixel_vector.ravel()
        pixel_vector = pixel_vector.reshape(1,612)
        if(svm_clf.predict(pixel_vector)==1):
            return True
        else:
            return False
    else:
        return False

def get_pixel_vector(image):
   
    pixel_count=0
    x_pos = 276;y_pos = 210
    dist=70
    len =17
    threshold=0.7
    pixel_vector = []
    for z in range (2):                         #top left and bot right
        x_pos += z*(dist-1)
        y_pos += z*(dist+1)
        for j in range(1,4):
            x_pos+=(j+1)%2
            y_pos-=j%2
            for i in range (len):
                b,g,r =image[y_pos+i,x_pos+i]
                pixel_vector.append([b,g,r]) 
                    #image[y_pos+i,x_pos+i]=[255,255,255]
    x_pos = 364
    y_pos = 210
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
 

