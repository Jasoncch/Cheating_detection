from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

data = pd.read_csv("Training/main/training_data/main_ml_training_set.csv",usecols=[1,2,3], header=0)
print(data)
data = data.values
class_data = pd.read_csv("Training/main/training_data/main_ml_training_set.csv",usecols=[4], header=0)
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
dump(svm_clf,'Models/New/main_svm.joblib')

print("Tree set score: %f" % tree_clf.score(data,label))
print("MLP set score: %f" % mlp_clf.score(data,label))
print("SVM set score: %f" % svm_clf.score(data,label))


def true_set_test(directory,threshold):

    test_set = pd.read_csv(directory,usecols=[1,2,3], header=0)
    #print (test_set.values)
    test_set=test_set.values
    t_true=0
    m_true=0
    s_true=0
    total = len(test_set)
    t = threshold
    for i in range(total):
        #print(str(i+1),test_set[i])
        test_data = [test_set[i]]
        t_result = tree_clf.predict_proba(test_data)
        m_result = mlp_clf.predict_proba(test_data)
        s_result = svm_clf.predict_proba(test_data)
        #print("Tree=",t_result,"MLP=",m_result,"SVM=",s_result)
        if t_result[0][1] > t:
            t_true+=1
        if m_result[0][1] > t:
            m_true+=1
        if s_result[0][1] > t:
            s_true+=1   
       # index+=1  
    return t_true,m_true,s_true,total

def false_set_test(directory,threshold):
    t=threshold
    test_set = pd.read_csv(directory,usecols=[1,2,3], header=0)
    #print (test_set.values)
    test_set=test_set.values
    t_true=0
    m_true=0
    s_true=0
    total = len(test_set)

    for i in range(total):
        #print(str(i+1),test_set[i])
        test_data = [test_set[i]]
        t_result = tree_clf.predict_proba(test_data)
        m_result = mlp_clf.predict_proba(test_data)
        s_result = svm_clf.predict_proba(test_data)
        #print("Tree=",t_result,"MLP=",m_result,"SVM=",s_result)
        if t_result [0][0] > 1-t:
            t_true+=1
        if m_result [0][0] > 1-t:
            m_true+=1
        if s_result[0][0] > 1-t:
            s_true+=1   
      #  index+=1 
    return t_true,m_true,s_true,total
t_count = [0,0,0]
f_count = [0,0,0]
tpr_list =[[],[],[]]
fpr_list =[[],[],[]]
acc_list =[[],[],[]]

for threshold in range (1,100):
    t = threshold/100
    directory = 'Training/main/testing_data/main_ml_true_testing_set.csv'
    t_count[0],t_count[1],t_count[2], t_total = true_set_test(directory,t)
    directory = 'Training/main/testing_data/main_ml_false_testing_set.csv'
    f_count[0],f_count[1],f_count[2], f_total = false_set_test(directory,t)
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

for i in range (3): 
    if i > 0:
        print(fpr_list[i]) 
        plt.figure()
        lw = 2
        plt.plot(
            fpr_list[i],
            tpr_list[i],
            
            color="red",
            lw=lw,
            label="ROC Curve",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        if (i==1):
            plt.title("MLP")
        if (i==2):
            plt.title("SVM")
        plt.legend(loc="lower right")
        plt.show()
