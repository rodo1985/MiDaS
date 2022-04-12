from turtle import width
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pyrealsense2 as rs
import torch
import open3d as o3d




def main():

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8 , 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get realsense intrinsics
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    intrinsic = o3d.camera.PinholeCameraIntrinsic(color_intrin.width, color_intrin.height, color_intrin.fx, color_intrin.fy, color_intrin.ppx, color_intrin.ppy)

    # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Large"
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")


    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

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

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            input_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            input_batch = transform(input_image).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=input_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output_image = prediction.cpu().numpy()
            
            output_image =  ((output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image)) * 255).astype(np.uint8)

            output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

            # Show images
            plt.subplot(121), plt.imshow(color_image), plt.title('Input Image')
            plt.subplot(122), plt.imshow(output_image), plt.title('Output Image')
            plt.show() 

            image_o3d = o3d.geometry.Image(input_image)
            depth_o3d = o3d.geometry.Image(output_image)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

            o3d.visualization.draw_geometries([pc],
                                    zoom=0.7,
                                    front=[ 0.29793294460704028, 0.081657225153760046, 0.95108782880340059],
                                    lookat=[-6.2407929896882803e-05, 0.00012044991775293291, 0.00049506709056452449],
                                    up=[0.021547212598669211, -0.99665598837361413, 0.078819784751307076],
                                    width = 1080,
                                    height = 720)

            # # Show images
            # plt.subplot(131), plt.imshow(color_image), plt.title('Input Image')
            # plt.subplot(132), plt.imshow(depth_colormap), plt.title('Depth Image')
            # plt.subplot(133), plt.imshow(output_image), plt.title('Output Image')
            # plt.show() 

            # # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap, output_image))
            # # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            # cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main()
