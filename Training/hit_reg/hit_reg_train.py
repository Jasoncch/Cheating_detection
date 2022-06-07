from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
import pandas as pd
import os
import glob
import cv2
import pickle 
from joblib import dump, load
import matplotlib.pyplot as plt

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

col_list = []
for i in range(360):
    col_list.append(i)

data = pd.read_csv("Training/hit_reg/training_data/ml_hit_data.csv",usecols=col_list, header=None)
print(data)
data = data.values
class_data = pd.read_csv("Training/hit_reg/training_data/ml_hit_data.csv",usecols=[360], header=None)
label= class_data.to_numpy().flatten()
#disicion tree
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(data,label)
#mlp
mlp_clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
mlp_clf.fit(data,label)  
#svm
svm_clf = svm.SVC(probability=True)
svm_clf.fit(data,label)
dump(svm_clf,'Models/New/hit_svm.joblib')
print("Tree set score: %f" % tree_clf.score(data,label))
print("MLP set score: %f" % mlp_clf.score(data,label))
print("SVM set score: %f" % svm_clf.score(data,label))



def test_true(directory,t):

    hit_frames = []
    data_vector = []
    #print("d=",directory)
    for file in glob.glob(directory+"/"+"*.png"):
        hit_frames.append(file)
    #print(str(len(hit_frames)))
    t_true=0
    m_true=0
    s_true=0
    index =0
    for file in hit_frames:
        img= cv2.imread(file)
    
        pixel_vector = get_pixel_vector(img)
        pixel_vector = np.asarray(pixel_vector) 
        pixel_vector = pixel_vector.ravel()
        test_data = pixel_vector.reshape(1,360)
        t_result = tree_clf.predict_proba(test_data)
        m_result = mlp_clf.predict_proba(test_data)
        s_result = svm_clf.predict_proba(test_data)
        #print("i=",index,"Tree=",t_result,"MLP=",m_result,"SVM=",s_result)
        if t_result[0][1] > t:
            t_true+=1
        if m_result[0][1] > t:
            m_true+=1
        if s_result[0][1] > t:
            s_true+=1   
        index+=1
    #print("total=",t_true,m_true,s_true)
    #print("Tree Accuracy=",t_true/len(hit_frames))
    #print("MLP Accuracy=",m_true/len(hit_frames))
    #print("SVM Accuracy=",s_true/len(hit_frames))
#dump(svm_clf,'kill_svm.joblib')
    return t_true,m_true,s_true,len(hit_frames)
def test_false(directory,t):
    hit_frames = []
    data_vector = []
    #print("d=",directory)
    for file in glob.glob(directory+"/"+"*.png"):
        hit_frames.append(file)
    #print(str(len(hit_frames)))
    t_true=0
    m_true=0
    s_true=0
    index =0
    for file in hit_frames:
        img= cv2.imread(file)
    
        pixel_vector = get_pixel_vector(img)
        pixel_vector = np.asarray(pixel_vector) 
        pixel_vector = pixel_vector.ravel()
        test_data = pixel_vector.reshape(1,360)
        t_result = tree_clf.predict_proba(test_data)
        m_result = mlp_clf.predict_proba(test_data)
        s_result = svm_clf.predict_proba(test_data)
        #print("i=",index,"Tree=",t_result,"MLP=",m_result,"SVM=",s_result)
        if t_result [0][0] > 1-t:
            t_true+=1
        if m_result [0][0] > 1-t:
            m_true+=1
        if s_result[0][0] > 1-t:
            s_true+=1   
        index+=1
    #print("total=",t_true,m_true,s_true)
    #print("Tree Accuracy=",t_true/len(hit_frames))
    #print("MLP Accuracy=",m_true/len(hit_frames))
    #print("SVM Accuracy=",s_true/len(hit_frames))
    return t_true,m_true,s_true,len(hit_frames)

t_count = [0,0,0]
f_count = [0,0,0]
tpr_list =[[],[],[]]
fpr_list =[[],[],[]]
acc_list =[[],[],[]]

for threhold in range (1,99):
    t = threhold/100
    directory = 'Training/hit_reg/testing_data/test_hit_data'
    t_count[0],t_count[1],t_count[2], t_total = test_true(directory,t)
    directory = 'Training/hit_reg/testing_data/test_no_hit_data'
    f_count[0],f_count[1],f_count[2], f_total = test_false(directory,t)
    print("T = ",t)

    for i in range (3):
        tpr = t_count[i]/t_total
        fpr = (f_total-f_count[i])/f_total
        accuracy =  (t_count[i]+f_count[i])/(t_total+f_total)
        
        tpr_list[i].append(tpr)
        fpr_list[i].append(fpr)
        acc_list[i].append(accuracy)
        tpr =tpr*100
        fpr =fpr*100
        accuracy = accuracy*100
        print("tpr = %.2f"%tpr+" fpr = %.2f"%fpr+" accuracy = %.2f"%accuracy)
