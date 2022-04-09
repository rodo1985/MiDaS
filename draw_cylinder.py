import cv2
import numpy as np


def draw_ellipse():
    # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    
    # Draw an ellipse
    center_coordinates = (120, 100)
    
    axesLength = (50, 50)
    
    angle = 0
    
    startAngle = 0
    
    endAngle = 360
    
    # Red color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 5 px
    thickness = 5
    
    # Using cv2.ellipse() method
    # Draw a ellipse with red line borders of thickness of 5 px
    cv2.ellipse(img, center_coordinates, axesLength,
            angle, startAngle, endAngle, color, thickness)
   
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

draw_ellipse()