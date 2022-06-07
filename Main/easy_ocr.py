from math import nan
import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import re
import hit_reg
import optical_flow as of
import glob
import torch
import os.path  
import kill_feed
#IMAGE_PATH = 'OCR/threshold.png'


def get_max_hit(directory):
    frame_length =len(glob.glob(directory+"/"+"*.png"))
    no_count=0
    frame_count=0
    max_frame = 0
    #print("len=",frame_length)
    for i in range(frame_length):
        filename=directory+"/"+str(i)+".png"
        print("reading: ",filename)
        image = cv2.imread(filename)
        if(hit_reg.check_hit(image)):
            no_count=0
            frame_count+=1
            if (frame_count>max_frame):
                max_frame=frame_count
        else:
            no_count+=1
            frame_count+=1
            if (no_count>4):                #grace period
                frame_count=0
                no_count=0
    return max_frame


def get_num(result):
    index=0
    max=-1
    threshold = 0.95
    #print(result)
    for each in result:
        index+=1
        #print("Detected Number:",each[1]+" Probility = ",each[2])
        if (any(each[1].isdigit() for i in each[1]) and  each[2]>threshold):
            num_str  = re.findall("\d+", each[1])[0]
            if (int(num_str)<260):
                num = int(num_str)
                if (num>max):
                    max=num    
    return max


def read_dmg(directory):
    print("loading")
    reader = easyocr.Reader(['en'],gpu=True)
    print("reading",directory)
    
    dmg_vector=[]
  
    max_frame=get_max_hit(directory)
    max_dmg=0
    frame_length =len(glob.glob(directory+"/"+"*.png"))
    #print("frame len=",frame_length)
    start = frame_length-max_frame-20
    if start <0:
        start = 0
    for i in range(start,frame_length):
        filename=directory+"/"+str(i)+".png"
        print("reading: ",filename)
        image = cv2.imread(filename)
        cv2.imshow("Reading frame",image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        try:
            if(hit_reg.check_hit(image) or kill_feed.check_kill(image)):
                try:
                    result = reader.readtext(image,allowlist='0123456789')
                    dmg=get_num(result)
                except Exception:
                    dmg= -1
            else:
                dmg=-1
        except Exception:
            dmg= -1
        #cv2.imshow("image",image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #rect,image = cv2.threshold(image,150,255,cv2.THRESH_BINARY)
        #cv2.imwrite(directory+"/"+"t_"+str(i)+".png",image)
        if (dmg>0):
            #print("dmg=",dmg)
            if (dmg>max_dmg):
                max_dmg=dmg
            dmg_vector.append(dmg)
    #print("max_dps:",max_dmg)
    #print("max_frame:",max_frame)
    #print("dpf= ",max_dmg/max_frame)
    #print(dmg_vector)
    if(max_dmg > 0 and max_frame>0):
        movement =  of.get_movement(directory,max_frame)
        cv2.putText(img=image , text="Dmg="+str(max_dmg), org=(10, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
        cv2.putText(img=image , text="Max frames= "+str(max_frame), org=(10, 60), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
        cv2.putText(img=image , text="Movement= %.4f"%(movement), org=(10, 90), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
        cv2.imshow("Reading",image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            x=1
        return(max_dmg,max_frame,max_dmg/max_frame,movement)
        
    else:
        return (max_dmg,max_frame,-1,-1) 

           


def read_frame(kill_count,directory):
    result=[]
    result_array=[]
    final_result_array=[]
    feature_array = []
    index=0
    dps=0
    for i in range(kill_count):
        file = directory+"\\"+str(i+1)
        dps = read_dmg(file)
        print ("Resulted feature=",dps)
        result.append(dps)
    for i in range(kill_count):
        result_array.append([result[i][0],result[i][1],result[i][2],result[i][3]])
        print("Kill:%2d dmg= %3d frame=%3d dps=%2.3f movement =%2.3f"%(i+1,result[i][0],result[i][1],result[i][2],result[i][3]))
        if (result[i][0]>100 and result[i][1]>0 and result[i][2]>0 and result[i][3]!=nan): 
            final_result_array.append([result[i][0],result[i][1],result[i][2],result[i][3]])
            feature_array.append([result[i][1],result[i][2],result[i][3]])
    
    data = np.asarray(result_array)
    np.savetxt(directory+"\\"+"raw_result_data.csv", data, delimiter=",",header="dmg,frame,dps,movement")
    data = np.asarray(final_result_array)
    np.savetxt(directory+"\\"+"final_result_data.csv", data, delimiter=",",header="dmg,frame,dps,movement")
    return feature_array
