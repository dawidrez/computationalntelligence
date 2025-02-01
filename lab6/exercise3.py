import cv2
import numpy as np
import os

files = os.listdir('bird_miniatures')
for file in files:
        img = cv2.imread('bird_miniatures/'+file)

        if img is None:
                print("Error: Image not found.")
                continue
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grayscale_image, 120, 255, cv2.THRESH_BINARY)

        # Invert the image for contour detection (since we want to detect black objects)
        inverted = cv2.bitwise_not(thresh)

        # Find contours of the birds
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove noise
        filtered_contours = []
        for contour in contours:
            filtered_contours.append(contour)

        # Count the number of objects
        object_count = len(filtered_contours)
        print(f"{file}: number of birds detected: {object_count}")

        # Optional: Draw contours on the original image
        result = img.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)  # Green contours

        # Show the results
        cv2.imshow("Original Image", img)
        cv2.imshow("Grayscale Image", grayscale_image)
        cv2.imshow("Thresholded Image", thresh)

        cv2.imshow("Detected Objects", result)

        # Wait for user input and clean up
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        cv2.waitKey(0)
