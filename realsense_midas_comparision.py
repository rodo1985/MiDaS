import cv2
from matplotlib import pyplot as plt
import numpy as np
import pyrealsense2 as rs
import torch
import open3d as o3d
import time


def image_and_depth_to_poincloud(input_image, depth_image, intrinsic):
    """
    Converts an image and depth image to a pointcloud.
    @parameters:
        param input_image: The input image.
        param depth_image: The depth image.
        param intrinsic: The intrinsic matrix of the camera.
    return: A pointcloud.
    """

    image_o3d = o3d.geometry.Image(input_image)
    depth_o3d = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)


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
    detph_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    

    # create camera matrix
    color_intrinsic = o3d.camera.PinholeCameraIntrinsic(color_intrin.width, color_intrin.height, color_intrin.fx, color_intrin.fy, color_intrin.ppx, color_intrin.ppy)
    depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(detph_intrin.width, detph_intrin.height, detph_intrin.fx, detph_intrin.fy, detph_intrin.ppx, detph_intrin.ppy)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # pointcloud object
    pc = rs.pointcloud()

    # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Large"
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # load model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # load transformation
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")


    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            input_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            start_time = time.time()

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
            
            print("--- %s seconds ---" % (time.time() - start_time))
            
            # output_image =  ((output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image)) * 255).astype(np.uint8)

            # output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

            # Show images
            plt.subplot(131), plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)), plt.title('Input Image')
            plt.subplot(132), plt.imshow(output_image, cmap='gray'), plt.title('MiDaS Depth Image')
            plt.subplot(133), plt.imshow(depth_image, cmap='gray'), plt.title('Realsense Depth Image')
            plt.show() 

                        
            # creates pointcloud from rs depth and color images
            pcd = image_and_depth_to_poincloud(input_image, depth_image, color_intrinsic)

            o3d.visualization.draw_geometries([pcd],
                                    window_name = 'Realsense Pointcloud',
                                    zoom=0.25,
                                    front=[ 0.042882783014263515, 0.34224346163446689, -0.93863223889306557 ],
                                    lookat=[ 0.32209703152893004, 0.015038716775138822, 0.5229317858260436 ],
                                    up=[ -0.0079988736549778065, -0.93934875012201546, -0.34287015569229556 ],
                                    width = 1080,
                                    height = 720)

            # creates pointcloud from MiDaS depth and color images
            pcd = image_and_depth_to_poincloud(input_image, output_image, color_intrinsic)

            o3d.visualization.draw_geometries([pcd],
                    window_name = 'MiDaS Pointcloud',
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
