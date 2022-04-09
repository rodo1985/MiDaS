import re
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import open3d as o3d


def main():
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

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while(True):
        
        start_time = time.time()
        
        ret, frame = vid.read()

        if ret:

            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            print("Time taken: ", time.time() - start_time)

            # plt.subplot(121), plt.imshow(input_image)
            # plt.subplot(122), plt.imshow(output_image)
            # plt.show()

            # Display the resulting frame and the output image
            cv2.imshow('frame',frame)
            cv2.imshow('output_image', ((output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image)) * 255).astype(np.uint8))

            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()