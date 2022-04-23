import os
import time

from matplotlib import pyplot as plt
from rembg import remove
import cv2
from PIL import Image

# Read all images in a folder
path = "./images"

for i, filename in enumerate(os.listdir(path)):
    if filename.endswith(".bmp") and i > 0:

        start_time = time.time()
        
        input_image = cv2.imread(os.path.join(path, filename))
        # input = Image.open(os.path.join(path, filename))

        output = remove(input_image)

        # convert image to gray
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # binarize image otsu
        mask = cv2.threshold(gray, 0, 254, cv2.THRESH_BINARY)[1]

        # close filter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # # add input image with output image
        # weighted_image = cv2.bitwise_and(input_image, input_image, mask= mask)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        weighted_image = cv2.addWeighted(input_image, 0.5, mask, 0.5, 0)

        print("--- %s seconds ---" % (time.time() - start_time))

        plt.subplot(141), plt.imshow(input_image)
        plt.subplot(142), plt.imshow(output)
        plt.subplot(143), plt.imshow(mask)
        plt.subplot(144), plt.imshow(weighted_image)
        plt.show()
        