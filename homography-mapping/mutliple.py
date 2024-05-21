import cv2
import numpy as np
import os

# Define the reference coordinates of the goal corners
ref_pts = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype="float32")


def compute_homography(image_points):
    # Convert image points to a numpy array
    img_pts = np.array(image_points, dtype="float32")

    # Compute the homography matrix
    H, status = cv2.findHomography(img_pts, ref_pts)
    return H


def apply_homography(H, points):
    # Transform the points using the homography matrix
    transformed_points = cv2.perspectiveTransform(np.array([points], dtype="float32"), H)
    return transformed_points[0]


def detect_goal(frame):
    # Placeholder for goal detection logic
    # This should return the coordinates of the goal corners
    # For example: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # In practice, you would use a detection algorithm here
    detected_points = [
        [100, 100],  # Top-left corner
        [500, 100],  # Top-right corner
        [500, 300],  # Bottom-right corner
        [100, 200]  # Bottom-left corner
    ]
    return detected_points


def process_frames(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()  # Sort files if necessary

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        # Detect goal in the current frame
        goal_points = detect_goal(frame)

        if len(goal_points) == 4:
            print(f'Goal points type is {type(goal_points)}')
            # Compute the homography matrix for the current frame
            H = compute_homography(goal_points)

            # Optionally, apply the homography to verify
            mapped_points = apply_homography(H, goal_points)
            print(f'Mapped points type is {type(mapped_points)}')

            # Draw the detected goal and transformed points on the frame
            for pt in goal_points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            for pt in mapped_points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

            # Display the homography matrix (optional)
            print(f"Frame: {image_file}")
            print("Homography Matrix:\n", H)

        # Display the processed frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Example usage with a folder path
folder_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/dataset/test/images'
process_frames(folder_path)
