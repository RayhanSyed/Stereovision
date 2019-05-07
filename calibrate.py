import numpy as np
import cv2
from pseyepy import Camera, Stream, Display
import string
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints1 = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.
objpoints2 = [] # 3d point in real world space
imgpoints2 = [] # 2d points in image plane.
c = Camera([0,1],resolution = [Camera.RES_LARGE,Camera.RES_LARGE])

# frame, timestamp = c.read(0)
# #frame2, t2 = c.read(1)
# i = 0


######################################################################	
#COMMENT ME OUT WHEN YOU ARE DONE TAKING PHOTOS	
# while (i < 10):
	# frame, timestamp = c.read(0)
	# frame2, t2 = c.read(1)
	# img = np.array(frame,copy = True)
	

	# img2 = np.array(frame2,copy = True)

	
	
	# print(cv2.imwrite("Cam1calib" + str(i) + ".jpg",img))
	# print(cv2.imwrite("Cam2calib" + str(i)+ ".jpg",img2))
	
	# i+=1
#####################################################################

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibrate1/*.jpg')

for fname in images:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
	print(ret)
	
	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp)

		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)


print(objpoints)
print(imgpoints)
print(gray.shape)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.savetxt('distance_coef_calibration1.txt',dist)
np.savetxt('matrix_calibration1.txt',mtx)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibrate2/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
	
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.savetxt('distance_coef_calibration2.txt',dist)
np.savetxt('matrix_calibration2.txt',mtx)

# i=0
# while(i<10):
	# frame, timestamp = c.read(0)
	# frame2, t2 = c.read(1)
	# img = np.array(frame,copy = True)
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# img2 = np.array(frame2,copy = True)
	# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	# # Find the chess board corners
	# ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
	# ret2, corners2 = cv2.findChessboardCorners(gray2, (7,6),None)
	# # If found, add object points, image points (after refining them)
	# if ret == True:
		# objpoints1.append(objp)
		# objpoints2.append(objp)
		# corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		# corners22 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
		# imgpoints1.append(corners12)
		# imgpoints2.append(corners22)
		# # Draw and display the corners
		# img = cv2.drawChessboardCorners(img, (7,6), corners12,ret)
		# img2 = cv2.drawChessboardCorners(img2, (7,6), corners22,ret2)
	# print(i)
	# i=i+1


# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints1, imgpoints1, gray.shape[::-1],None,None)
# ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, gray2.shape[::-1],None,None)
# np.savetxt('distance_coef_calibration.txt',dist)
# np.savetxt('matrix_calibration.txt',mtx)
# np.savetxt('distance_coef_calibration2.txt',dist2)
# np.savetxt('matrix_calibration2.txt',mtx2)

