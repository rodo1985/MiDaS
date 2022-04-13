import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import os
import tqdm

# Calibrate camera using opencv and get the intrinsic matrix and distortion coefficients
def calibrate_camera(path):

    # Defining the dimensions of checkerboard
    CHECKERBOARD = (17,11)
    checker_widht_in_meters = 15
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

    # convert to meters
    objp = objp * checker_widht_in_meters

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = [f for f in os.listdir(path) if f.endswith('.jpg')]

    # start progress bar
    pbar = tqdm.tqdm(total=len(images))

    # Step through the list and search for chessboard corners
    for fname in images:
        
        # Read image
        img = cv2.imread(os.path.join(path, fname))

        # Convert image to gray scale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points
        if ret == True:
            
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

            # Append object points and image points
            objpoints.append(objp)
            imgpoints.append(corners2)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()
        
        # update progress bar
        pbar.update(1)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print('total error: ', mean_error/len(objpoints))

    return  mtx, dist, criteria, objp, CHECKERBOARD

def undistortion(img, mtx, dist):
    
    h,  w = img.shape[:2]
    newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst

def draw(img, corners, imgpts):
    try:
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, np.int0(corner), np.int0(tuple(imgpts[0].ravel())), (255,0,0), 5)
        img = cv2.line(img, np.int0(corner), np.int0(tuple(imgpts[1].ravel())), (0,255,0), 5)
        img = cv2.line(img, np.int0(corner), np.int0(tuple(imgpts[2].ravel())), (0,0,255), 5)
        
        return img
    except:
        return img

def acquire_images(mtx, dist, criteria, objp, CHECKERBOARD):
    
    # axis to plot
    axis = np.float32([[285,0,0], [0,180,0], [0,0,-60]]).reshape(-1,3)

    # define a video capture object
    vid = cv2.VideoCapture(0)

    # define a counter 
    acc = 0
    
    while(True):
    
        # Capture frame-by-frame
        ret, frame = vid.read()

        if ret:
            
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # Find the rotation and translation vectors.
                ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                frame = draw(frame, corners2, imgpts)

                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

            # Display the resulting frame
            cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Image', frame)

            # if key is pressed
            key = cv2.waitKey(1)

            # if key is pressed
            if  key & 0xFF == ord('s'):
                # write images
                cv2.imwrite('calibration_images/input_image' + str(acc).zfill(3) + '.png', frame)
                acc+=1
            elif key & 0xFF == ord('q'):
                break
    
    # After the loop release the cap object
    vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

def draw_3d_cylinder(image):
    """
    Draw a 3d cylinder using opencv
    """


mtx, dist, criteria, objp, CHECKERBOARD = calibrate_camera('calibration_images')

print(mtx, dist)

undistortion_image = undistortion(cv2.imread('calibration_images/input_image001.jpg'), mtx, dist)

plt.imshow(undistortion_image)
plt.show()


acquire_images(mtx, dist, criteria, objp, CHECKERBOARD)