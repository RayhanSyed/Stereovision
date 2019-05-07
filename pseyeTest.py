from pseyepy import Camera, Stream, Display
import cv2
import numpy as np

c = Camera([0,1],resolution = [Camera.RES_LARGE,Camera.RES_LARGE])
#c = Camera(resolution = [Camera.RES_LARGE])
#c.exposure = 23

frame, timestamp = c.read(0)
frame2, t2 = c.read(1)

#matType = cv2.CV_8UC3; 
#test = cv2.CreateMat(480, 640, matType, frame)

test = np.array(frame,copy = True)
gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

test2 = np.array(frame2,copy = True)
gray2 = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)

cv2.imshow("test",gray)
cv2.imshow("test2",gray2)
print(test)

d = Display(c)
#s = Stream(c, file_name='example_movie.avi', display=True,codec='png') # begin saving data to files
#s.end()
#d = Display(c)
#print(frame)
#s = Stream(c, file_name='example_movie.avi', display = True, codec='png') # begin saving data to files

#s.end()

c.end()

