import os
import cv2

def get_training_data(dir_path):
	dirs=os.listdir(dir_path)
	persons=[]
	for dir in dirs:
		dir_check=dir_path+"/"+dir
		if not os.path.isfile(dir_check):
			persons.append(dir)
		#print(y_train)

	x_train=[]
	y_train=[]
	for person in persons:
		img_path=dir_path+"/"+person
		images=os.listdir(img_path)
	
		for image in images:
			img=cv2.imread(img_path+"/"+image)
			resized=cv2.resize(img,(280,280))
			x_train.append(resized)
			y_train.append(person)
	
	return x_train,y_train

print(get_training_data("./data"))