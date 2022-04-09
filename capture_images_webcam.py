import os
import shutil
import cv2


def main():
    
    # Empty calibration_images folder
    for filename in os.listdir("calibration_images"):
        file_path = os.path.join("calibration_images", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

     # define a video capture object
    vid = cv2.VideoCapture(0)

    # define a counter 
    acc = 0
    
    while(True):
    
        # Capture frame-by-frame
        ret, frame = vid.read()

        if ret:

            # Display the resulting frame
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', frame)

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

if __name__ == "__main__":
    main()