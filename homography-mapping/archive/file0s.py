import cv2
import numpy as np

# Load the image
img_path = 'dog.png'
im_src = cv2.imread(img_path)

if im_src is None:
    print("Failed to load image.")
else:
    print("Image loaded successfully.")

    # Define the source points (these points should be chosen based on your specific image)
    pts_src = np.array([
        [320, 150],
        [700, 150],
        [700, 550],
        [320, 550]
    ])

    # Define the destination points (where you want the source points to map to)
    # For examples, we map the points to a rectangular area
    pts_dst = np.array([
        [0, 0],
        [400, 0],
        [400, 600],
        [0, 600]
    ])

    # Compute the homography matrix
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Apply the homography to warp the image
    im_dst = cv2.warpPerspective(im_src, h, (400, 600))

    # Display the original and the warped image
    cv2.imshow('Source Image', im_src)
    cv2.imshow('Warped Image', im_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
