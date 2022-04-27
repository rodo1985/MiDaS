import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pyrealsense2 as rs
import os
import shutil

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

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_all_streams()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # start streaming
    profile = pipeline.start(config)

    # set settings
    profile.get_device().sensors[0].set_option(rs.option.exposure, 500)
    profile.get_device().sensors[0].set_option(rs.option.enable_auto_exposure, 0)

    # Get realsense intrinsics
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    # create camera matrix
    mtx = np.eye(3, k=0)
    mtx[0, 0] = color_intrin.fx
    mtx[1, 1] = color_intrin.fy
    mtx[0, 2] = color_intrin.ppx
    mtx[1, 2] = color_intrin.ppy
    dist= np.zeros(5)
    
    print(mtx)

    print(color_intrin)
    acc = 0

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # convert to rgb
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)

            # if key is pressed
            key = cv2.waitKey(1)

            # if key is pressed
            if key & 0xFF == ord('s'):
                # write images
                cv2.imwrite('calibration_images/input_image' + str(acc).zfill(3) + '.png', color_image)
                acc+=1
            elif key & 0xFF == ord('q'):
                break


            
            # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))

            # # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            # cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main()