import cv2
import numpy as np
from joblib import dump, load

def check_hit(img):
    svm_clf = load("Models/hit_svm.joblib")
    pixel_vector= get_pixel_vector(img) 
    pixel_vector = np.asarray(pixel_vector) 
    pixel_vector = pixel_vector.ravel()
    pixel_vector = pixel_vector.reshape(1,360)
    if(svm_clf.predict(pixel_vector)==1):
        return True
    else:
        return False


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

