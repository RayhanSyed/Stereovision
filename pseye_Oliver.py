from pseyepy import Camera, Stream, Display
import cv2
import numpy as np
from cv2 import aruco
from cv2 import Rodrigues 
from math import sqrt
import matplotlib.pyplot as plt

c = Camera([0,1],resolution = [Camera.RES_LARGE,Camera.RES_LARGE])
#c = Camera(resolution = [Camera.RES_LARGE])
#c.exposure = 23

#frame, timestamp = c.read(0)
#frame2, t2 = c.read(1)

#matType = cv2.CV_8UC3; 
#test = cv2.CreateMat(480, 640, matType, frame)

#test = np.array(frame,copy = True)
#gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

#test2 = np.array(frame2,copy = True)
#gray2 = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)

#cv2.imshow("test",gray)
#cv2.imshow("test2",gray2)
#print(test)

#d = Display(c)
#s = Stream(c, file_name='example_movie.avi', display=True,codec='png') # begin saving data to files
#s.end()
#d = Display(c)
#print(frame)
#s = Stream(c, file_name='example_movie.avi', display = True, codec='png') # begin saving data to files

#s.end()



# Setup Aruco Dictionary
aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_1000 )
arucoParams = aruco.DetectorParameters_create()

# These 4 lines are used to create a picture for each of the markers 
# that are used in the code (not needed if they already exist)
# Please be sure that the printed stickers are 3 x 3 inches
# marker1 = aruco.drawMarker(aruco_dict,23,200,1)
# marker2 = aruco.drawMarker(aruco_dict,24,200,1)
# cv2.imwrite('marker1.jpg',marker1) 
# cv2.imwrite('marker2.jpg',marker2)

# Retrieve camera calibration parameters
dist = np.loadtxt('distance_coef_calibration1.txt')
mtx = np.loadtxt('matrix_calibration1.txt')
dist2 = np.loadtxt('distance_coef_calibration2.txt')
mtx2 = np.loadtxt('matrix_calibration2.txt')

vidout = True
sidel = 5.08
grphout = True
maxima = []
err = []
r = []
maxy = 2.5
#cap = cv2.VideoCapture(filename)
#tempname = filename.split('/')
#tempname1 = str(tempname[1])
#tempname2 = tempname1.split('.')
#fname = str(tempname2[0])
h = 480#c.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = 640#c.get(cv2.CAP_PROP_FRAME_WIDTH)
fname = "output"
out = cv2.VideoWriter('output_right.avi',cv2.VideoWriter_fourcc(*'MJPG'),30.0,(int(w),int(h)))
out2 = cv2.VideoWriter('output_left.avi',cv2.VideoWriter_fourcc(*'MJPG'),30.0,(int(w),int(h)))

#Preparations for the video loop
j = 0
finaldistx = []
finaldisty = []
finaldistz = []
totdistance = []
time = []
firstpass = True
offset = [0,0,0]
font = cv2.FONT_HERSHEY_SIMPLEX
i = 0
distarr = []
dist2arr = []
vecdot = [[0,0,0]]
vec2dot = [[0,0,0]]
while(i<300):
	j = 1 + j
	img, timestamp = c.read(0)
	img3, timestamp = c.read(1)
	if(True):
		corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)
		img2 = aruco.drawDetectedMarkers(img, corners, ids)
		h,w,cha = img2.shape
		rvecs, tvecs, objpoints = aruco.estimatePoseSingleMarkers(corners,sidel,mtx,dist)
		
		corners2, ids2, rejectedImgPoints2 = aruco.detectMarkers(img3, aruco_dict, parameters=arucoParams)
		img4 = aruco.drawDetectedMarkers(img3, corners2, ids2)
		h2,w2,cha2 = img4.shape
		rvecs2, tvecs2, objpoints2 = aruco.estimatePoseSingleMarkers(corners2,sidel,mtx2,dist2)
		try:
			if(ids[0] == 23):
				index23 = 0
				index24 = 1
			elif(ids[0] == 24):
				index23 = 1
				index24 = 0
			aruco.drawAxis(img2,mtx,dist,rvecs[index23,0],tvecs[index23,0],sidel/2)
			aruco.drawAxis(img2,mtx,dist,rvecs[index24,0],tvecs[index24,0],sidel/2)
			Rmat23,jacob23 = Rodrigues(rvecs[index23,0])
			Rmat24,jacob24 = Rodrigues(rvecs[index24,0])
			R23inv = np.linalg.inv(Rmat23)
			R24inv = np.linalg.inv(Rmat24)
			co24to23 = np.matmul(R23inv,tvecs[index24,0]-tvecs[index23,0])
			distance = co24to23
			#print(distance)
			# if(firstpass == True):
				# offset = distance
				# firstpass = False
			# distance = distance -offset
			tempdist = sqrt(distance[0]*distance[0] +distance[1]*distance[1]+distance[2]*distance[2])
			totdistance.append(distance[1])
			finaldistx.append(distance[0])
			finaldisty.append(distance[1])
			finaldistz.append(distance[2])
			distance2 = distance2 -offset
			#print('succ')
			tempdist2 = sqrt(distance2[0]*distance2[0] +distance2[1]*distance2[1]+distance2[2]*distance2[2])
			
		except:
			distance = np.zeros(1)
			#print('failed')	
		try:
			if(ids2[0] == 23):
				index232 = 0
				index242 = 1
			elif(ids2[0] == 24):
				index232 = 1
				index242 = 0
			aruco.drawAxis(img4,mtx2,dist2,rvecs2[index232,0],tvecs2[index232,0],sidel/2)
			aruco.drawAxis(img4,mtx2,dist2,rvecs2[index242,0],tvecs2[index242,0],sidel/2)
			Rmat232,jacob232 = Rodrigues(rvecs2[index232,0])
			Rmat242,jacob242 = Rodrigues(rvecs2[index242,0])
			R23inv2 = np.linalg.inv(Rmat232)
			R24inv2 = np.linalg.inv(Rmat242)
			co24to232 = np.matmul(R23inv2,tvecs2[index242,0]-tvecs2[index232,0])
			distance2 = co24to232
			#print(distance2)
			# if(firstpass2 == True):
				# offset2 = distance2
				# firstpass2 = False
			tempdist2 = sqrt(distance2[0]*distance2[0] +distance2[1]*distance2[1]+distance2[2]*distance2[2])
			totdistance.append(distance2[1])
			finaldistx.append(distance2[0])
			finaldisty.append(distance2[1])
			finaldistz.append(distance2[2])
			time.append((j-1)/30)
		
		except:
			distance2 = np.zeros(1)			
		#print(str(distance.size) + ', ' + str(distance2.size))
		#print(distance2)
		#print(distance2.size)
		distance = np.around(distance,decimals = 2,out = None)
		distance2 = np.around(distance2,decimals = 2,out = None)
		distarr.append(distance)
		dist2arr.append(distance2)
		if i > 4 and i % 5 == 0:
			if i % 5 == 0:
				vecdot.append((distance - distarr[i-5]) / (5/30))
				vec2dot.append((distance2 - dist2arr[i-5]) / (5/30))
			#print(str(distance) + ', ' + str(distarr[i]))
			#print(distance)
		else:
			vecdot.append(vecdot[i-1])
			vec2dot.append(vec2dot[i-1])
		#print(str(vecdot[i]) + ',  ' + str(i))	
		if(distance.size ==1) and (distance2.size == 1):
			cv2.putText(img2, 'X,Y,Z = [? ? ?]', (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img2, 'Xdot,Ydot,Zdot = [? ? ?]', (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img2, 'Distance = [? ? ?]', (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img2, 'Speed = [? ? ?]', (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			out.write(img2)
			cv2.putText(img4, 'X,Y,Z = [? ? ?]', (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img4, 'Xdot,Ydot,Zdot = [? ? ?]', (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img4, 'Distance = [? ? ?]', (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img4, 'Speed = [? ? ?]', (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			out2.write(img4)

			#print('hi')
		elif(distance.size ==3) and (distance2.size == 3):
			avdist = (distance+distance2)/2
			try:
				
				xdot = round((vecdot[i][0] + vec2dot[i][0] / 2),2)
				ydot = round((vecdot[i][1] + vec2dot[i][1] / 2),2)
				zdot = round((vecdot[i][2] + vec2dot[i][2] / 2),2)
				avspeed = [xdot,ydot,zdot]
				velFinal = (avspeed[0]**2 + avspeed[1]**2 + avspeed[2]**2)**0.5
			except:
				avspeed = 'Err'
				velFinal = ['Err','Err','Err']
			distFinal = (avdist[0]**2 + avdist[1]**2 + avdist[2]**2)**0.5
			cv2.putText(img2, 'X,Y,Z = ' + str(avdist), (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img4, 'X,Y,Z = ' + str(avdist), (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img2, 'Xdot,Ydot,Zdot = ' + str(avspeed), (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img4, 'Xdot,Ydot,Zdot = ' + str(avspeed), (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img2, 'Distance = ' + str(distFinal), (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img2, 'Speed = ' + str(velFinal), (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img4, 'Distance = ' + str(distFinal), (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img4, 'Speed = ' + str(velFinal), (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)			
			out.write(img2)
			out2.write(img4)

			#print('hi')
		else:
			if(distance.size == 3):
				avdist = distance
				avspeed = vecdot[i]
				distFinal = (avdist[0]**2 + avdist[1]**2 + avdist[2]**2)**0.5				
				try:
					avspeed[0] = round(avspeed[0],2)
					avspeed[1] = round(avspeed[1],2)
					avspeed[2] = round(avspeed[2],2)
					velFinal = (avspeed[0]**2 + avspeed[1]**2 + avspeed[2]**2)**0.5
				except:
					avspeed = 'Err'
					velFinal = ['Err','Err','Err']					
				cv2.putText(img2, 'X,Y,Z = ' + str(avdist), (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'X,Y,Z = ' + str(avdist), (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img2, 'Xdot,Ydot,Zdot = ' + str(avspeed), (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'Xdot,Ydot,Zdot = ' + str(avspeed), (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)				
				cv2.putText(img2, 'Distance = ' + str(distFinal), (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img2, 'Speed = ' + str(velFinal), (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'Distance = ' + str(distFinal), (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'Speed = ' + str(velFinal), (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)				
				out.write(img2)
				out2.write(img4)
			else:
				avdist = distance2
				avspeed = vec2dot[i]
				distFinal = (avdist[0]**2 + avdist[1]**2 + avdist[2]**2)**0.5				
				try:
					avspeed[0] = round(avspeed[0],2)
					avspeed[1] = round(avspeed[1],2)
					avspeed[2] = round(avspeed[2],2)
					velFinal = (avspeed[0]**2 + avspeed[1]**2 + avspeed[2]**2)**0.5
				except:
					avspeed = 'Err'
					velFinal = ['Err','Err','Err']
				cv2.putText(img2, 'X,Y,Z = ' + str(avdist), (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'X,Y,Z = ' + str(avdist), (0, h-50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img2, 'Xdot,Ydot,Zdot = ' + str(avspeed), (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'Xdot,Ydot,Zdot = ' + str(avspeed), (0, h-90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)								
				cv2.putText(img2, 'Distance = ' + str(distFinal), (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img2, 'Speed = ' + str(velFinal), (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'Distance = ' + str(distFinal), (0, h-130), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(img4, 'Speed = ' + str(velFinal), (0, h-170), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)				
				out.write(img2)
				out2.write(img4)				
				
			
		
	i=i+1

if(vidout == True):
	out.release()
	out2.release()
totdist = np.asarray(totdistance)
#c.release()
c.end()
sort = np.sort(totdist)
min = sort[:10]
sort = sort[::-1]
max = sort[:10]
minavg = min.mean()
maxavg = max.mean()
totavg = (minavg + maxavg)/2
sdom = max.std()
err.append(sdom)
maxima.append((maxavg - totavg)/maxy)
#omega = int(fname.split('_')[1])
#r.append(omega)
#print("Finished " + str(omega))
# Plot the results
# if(grphout ==True):
	# fig,(ax) = plt.subplots(1,1)
	# ax.plot(time,totdistance)
	# fig.savefig('output/graph/graphdist'+fname+'.png')

# disp, axdisp = plt.subplots(1,1)
# axdisp.errorbar(r,maxima,yerr=err,fmt = 'o', ecolor = 'red')
# disp.savefig('output/disptrans41.png')
# print(r)
print('\n')
print(maxima)
print('\n')
print(err)
