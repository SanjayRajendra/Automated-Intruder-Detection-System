import cv2
import sys
import os
import glob

path="./data/"
cascPath = "./data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
img_wide_range=20

files = os.listdir(path)
for file_name in files:
    dir_name=path+file_name
    if os.path.isdir(dir_name):
        print("From :"+dir_name)

        files=os.listdir(dir_name)
        #print(files)
        for file in files:
            file_path=dir_name+"/"+file;
            
            if os.path.isfile(file_path):
            	s=cv2.imread(file_path)
                gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(500,500),flags=cv2.CASCADE_SCALE_IMAGE)

                if len(faces)>1:
                    print("More than one face found. "+file+" removing img from directory.")
                    os.remove(file_path)
                elif len(faces)==0:
                    print("No faces found."+file+" removing img from directory.")
                    os.remove(file_path)
                else:
                    (x, y, w, h) = faces[0]
                    img=s[y-img_wide_range:y+h+img_wide_range,x-img_wide_range:x+w+img_wide_range] 
                    img1=cv2.resize(img,(280,280), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(file_path,img1)
                    print(file)
