import cv2

# connect with ip camera with user and password
cap = cv2.VideoCapture('rtsp://admin:@192.168.1.10:8080')


# cap = cv2.VideoCapture('http://192.168.1.10:8080')

# check if camera is connected
if cap.isOpened() == False:
    print("Error opening video stream or file")

# read frames from camera
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # show frames
        cv2.imshow('Frame', frame)

        # press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break