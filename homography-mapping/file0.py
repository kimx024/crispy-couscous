# This one works but doesn't do mapping
import cv2
import numpy as np

# Initialize global variables
pts_src = []


def click_event(event, x, y, flags, param):
    global pts_src
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked points
        pts_src.append((x, y))
        # Draw a circle at the clicked point
        cv2.circle(im_src, (x, y), 5, (0, 0, 255), -1)
        # Display the updated image with the point
        cv2.imshow('Source Image', im_src)
        # Print the clicked point coordinates
        print(f"Point {len(pts_src)}: ({x}, {y})")
        # If 4 points are clicked, close the window
        if len(pts_src) == 4:
            cv2.destroyAllWindows()


# Load the image
img_path = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/dataset/train/images/0005_jpg.rf.7eada4302fb9d9e7e7c8b1c591613eee.jpg"
im_src = cv2.imread(img_path)

if im_src is None:
    print("Failed to load image.")
else:
    print("Image loaded successfully.")

    # Display the image and set the mouse callback function
    cv2.imshow('Source Image', im_src)
    cv2.setMouseCallback('Source Image', click_event)
    cv2.waitKey(0)

    # Check if we have 4 points
    if len(pts_src) != 4:
        print("Error: You need to click exactly 4 points.")
    else:
        pts_src = np.array(pts_src)

        # Define the destination points (where you want the source points to map to)
        # For example, we map the points to a rectangular area
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

        # Display the warped image
        cv2.imshow('Warped Image', im_dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
