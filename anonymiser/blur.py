import cv2 

# to detect the face of the human 
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# VideoCapture is a function, to capture 
# video from the camera attached to system 
# You can pass either 0 or 1 
# 0 for laptop webcam 
# 1 for external webcam 
video_capture = cv2.VideoCapture(0) 

# a while loop to run infinite times, 
# to capture infinite number of frames for video 
# because a video is a combination of frames 
while True: 
	
	# capture the latest frame from the video 
	check, frame = video_capture.read() 

	# convert the frame into grayscale(shades of black & white) 
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

	# detect multiple faces in a captured frame 
	# scaleFactor: Parameter specify how much the 
	# image sizeis reduced at each image scale. 
	# minNeighbors: Parameter specify how many 
	# neighbours each rectangle should have to retain it. 
	# rectangle consists the detect object. 
	# Here the object is the face. 
	face = cascade.detectMultiScale( 
		gray_image, scaleFactor=2.0, minNeighbors=4) 

	for x, y, w, h in face: 

		# draw a border around the detected face. 
		# (here border color = green, and thickness = 3) 
		image = cv2.rectangle(frame, (x, y), (x+w, y+h), 
							(0, 255, 0), 3) 

		# blur the face which is in the rectangle 
		image[y:y+h, x:x+w] = cv2.medianBlur(image[y:y+h, x:x+w], 
											35) 

	# show the blurred face in the video 
	cv2.imshow('face blurred', frame) 
	key = cv2.waitKey(1) 

	# This statement just runs once per frame. 
	# Basically, if we get a key, and that key is a q, 
	if key == ord('q'): 
		break

# we will exit the while loop with a break, 
# which then runs: 
video_capture.release() 
cv2.destroyAllWindows()
