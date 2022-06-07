import feature_extractor as fe
from joblib import dump, load
import easy_ocr as ez
file = input('Enter video name: ')
svm_clf = load("Models\main_svm.joblib") 

feature_array = fe.process_video(file)

index = 0
for i in feature_array:
    index +=1
    result=svm_clf.predict_proba([[i[0],i[1],i[2]]])
    print("Kill %d has %2.3f chance of cheating"%(index,result[0][1]*100))