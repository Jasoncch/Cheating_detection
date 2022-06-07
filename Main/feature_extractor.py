import cv2
import numpy as np
import kill_feed
from time import process_time
import os
import easy_ocr as ez
import glob

def process_video(file):
  file = "Video/"+file
  cap = cv2.VideoCapture(file) 
  size = len(file) 
  temp_folder = "Temp_"+file[6:size-4] 
  if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
  directory = temp_folder
  
  files = glob.glob(temp_folder+'/*')

  #os.chdir(directory)
  #ez.read_frame()
  if (cap.isOpened()== False): 
    print("invalid file")
  y=288;x=640;h=320;w=640
  buffer=False
  buffer_count=50
  con_count=0
  frame_id=0
  frame_vector=[]
  temp_size=120
  kill_count =0
  if not os.path.exists(temp_folder+"/kill frame"):
    os.makedirs(temp_folder+"/kill frame")

  
  while(cap.isOpened()):
    
    frame_id+=1
    ret, frame = cap.read()
    if(ret):
      
      temp_frame = cv2.resize(frame,(1280,720))
      cv2.imshow("Reading",temp_frame)
      frame = cv2.resize(frame,(1920,1080))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
      crop = frame[y:y+h, x:x+w]
      image = crop
      blur = cv2.medianBlur(crop,5)
      filename =str(frame_id)+'.png'
      print(filename)
      if (len(frame_vector)<temp_size):
        frame_vector.append(crop)
      else:
        frame_vector.pop(0)
        frame_vector.append(crop)
      #kill_feed_new.get_color_value(blur)
      if (not buffer):                                  #check if in buffer
        result=kill_feed.check_kill(image)
        if(result):
          print("checked")
          cv2.putText(img=temp_frame , text='Kill detected', org=(10, 60), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255),thickness=3)
          cv2.imshow("Reading",temp_frame)
          if cv2.waitKey(25) & 0xFF == ord('q'):
            break
          con_count+=1
          if con_count==1:                              #2con frames
            #cv2.imwrite(filename,blur)
            print("kill:"+filename,"len:",len(frame_vector))
            kill_count+=1
            os.mkdir(temp_folder+"/"+str(kill_count))
          
            for i in range(len(frame_vector)):
              cv2.imwrite(temp_folder+"/"+str(kill_count)+"/"+str(i)+'.png',frame_vector[i])
              if(i==len(frame_vector)-1):
                print("kill frame")
                cv2.imwrite(temp_folder+"/kill_frame/"+str(kill_count)+'.png',frame_vector[len(frame_vector)-1])
            #os.chdir(directory)
            con_count=0
            buffer= True
      else:
        buffer_count-=1
      if (buffer_count<0):
        buffer_count=50
        buffer=False
    else:
      break

  cv2.destroyAllWindows()
  print("processing frames")
  print("Total Kills detected ",str(kill_count)+ " saved in "+ directory)
  feature_array = ez.read_frame(kill_count,directory)
  cap.release()
  cv2.destroyAllWindows()
  return feature_array
