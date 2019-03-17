import cv2
cascPath = "./data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
face=cv2.imread("./me.jpg")
gray= cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
gray,
scaleFactor=1.1,
minNeighbors=10,
minSize=(30, 30),
flags=cv2.CASCADE_SCALE_IMAGE
)
s=face[378:378+285,516:516+285,:]
for (x, y, w, h) in faces:
	cv2.rectangle(face, (x, y), (x+w, y+h), (0, 255, 0), 2)
	cv2.imshow('image'+str(x),face[y:y+h,x:x+w])

cv2.imshow("img",face)
cv2.waitKey(0)
cv2.destroyAllWindows()