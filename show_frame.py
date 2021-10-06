import cv2, time

path = r"C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/me.mp4"
frame_number = 21
(x,y,w,h) = (183, 294, 63, 63)

cap = cv2.VideoCapture(path)

i = 0
while(True):
	if(i == frame_number):
		break

	ret, img = cap.read()

	if img is None:
		break

	if (cv2.waitKey(1) & 0xFF == ord('q')) or ret == False : #Youssef- use ret in case no more frames in a video file. It may not be needed since we test whether img is None in each loop
		break

	i = i + 1
	
	cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1)
	img = cv2.resize(img, (960, 540))
	cv2.imshow('img1',img) #this is working fine actually

#if(i == frame_number):
#	cv2.imshow('img1',img) #not working unfortunately

input()
cap.release()