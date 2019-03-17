import cv2
import os

name=input("Enter your Name:")
path="./data/"+name
if not os.path.exists(path):
    os.makedirs(path)
else:
    raise "Your name is exist"


cascPath = "./data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
i=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)


    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(path+"/image"+str(i)+".jpg",frame[y-20:y+h+20,x-20:x+w+20])
            i+=1

    cv2.imshow('VideoCapture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if i>=10:
        break


video_capture.release()
cv2.destroyAllWindows()