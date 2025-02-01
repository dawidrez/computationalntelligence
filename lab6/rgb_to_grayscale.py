import cv2
import numpy as np

folder = "media/"
paths = ["cat.png", "lena.png", "korfu.png", ]

for path in paths:
    img = cv2.imread(folder + path)
    if img is None:
        print("Error: Image not found.")
    else:
        # Convert the image to grayscale using the formula (R + G + B) / 3
        grayscale_mean_image = (img[:, :, 0]/3 + img[:, :, 1]/3 + img[:, :, 2]/3)   # Summing B, G, R channels
        grayscale_mean_image = grayscale_mean_image.astype(np.uint8)  # Convert to uint8 for proper image format
        cv2.imwrite(f'{folder}grayscale_mean_{path}', grayscale_mean_image)  # Save the result
        grayscale_image = (img[:, :, 0]*0.114 + img[:, :, 1]* 0.587 + img[:, :, 2]*0.299) / 3
        grayscale_image = grayscale_image.astype(np.uint8)
        cv2.imwrite(f'{folder}grayscale_{path}', grayscale_image)