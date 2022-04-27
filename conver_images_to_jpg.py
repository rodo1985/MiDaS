import os
import cv2

path = 'calibration_images'
files = os.listdir(path)

# create directory
if os.path.exists(os.path.join(path, 'jpg')):
  # remove directory and files
  os.rmdir(os.path.join(path, 'jpg'))

# create directory
os.mkdir(os.path.join(path, 'jpg'))

for file in files:
  if file.endswith(".png"):
    img = cv2.imread(os.path.join(path, file))
    cv2.imwrite(os.path.join(path, 'jpg', file.replace('.png','.jpg')), img)
# # open image with opencv and save as jpg
# img = cv2.imread('calibration_images/input_image000.png')
# cv2.imwrite('calibration_images/input_image000.jpg', img)
