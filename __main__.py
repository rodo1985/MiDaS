from tkinter import W
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

    # Read all images in a folder
    path = "./images"

    for filename in os.listdir(path):
        if filename.endswith(".bmp"):

            start_time = time.time()

            input_image = cv2.cvtColor(cv2.imread(
                os.path.join(path, filename)), cv2.COLOR_BGR2RGB)

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

            # binarize image otsu
            output_image = cv2.threshold(output_image, 90, 255, cv2.THRESH_BINARY)[1]

            # add input image with output image
            weighted_image = cv2.bitwise_and(input_image, input_image, mask= output_image)

            print("--- %s seconds ---" % (time.time() - start_time))

            plt.subplot(131), plt.imshow(input_image)
            plt.subplot(132), plt.imshow(output_image)
            plt.subplot(133), plt.imshow(weighted_image)
            plt.show()


if __name__ == "__main__":
    main()
