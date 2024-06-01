import cv2
import numpy as np
import os
import torch

# Define the reference coordinates of the goal corners
reference_points = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype="float32")


def compute_homography(image_points):
    # Convert image points to a numpy array
    img_pts = np.array(image_points, dtype="float32")

    # Compute the homography matrix
    homography_matrix, status = cv2.findHomography(img_pts, reference_points)
    return homography_matrix


def apply_homography(H, points):
    # Transform the points using the homography matrix
    transformed_points = cv2.perspectiveTransform(np.array([points], dtype="float32"), H)
    return transformed_points[0]


def yolo_to_bbox(image_shape, yolo_annotation):
    height, width = image_shape[:2]
    cls, x_center, y_center, w, h = yolo_annotation
    x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
    x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
    return int(cls), x1, y1, x2, y2

def detect_goal_with_yolo(frame, annotations):
    height, width = frame.shape[:2]
    goal_sections = {
        "goal-top-left": None,
        "goal-middle-down": None,
        "goal-top-right": None,
        "goal-bottom-left": None,
        "goal-middle-up": None,
        "goal-bottom-right": None
    }

    class_names = ["goal-top-left", "goal-middle-down", "goal-top-right",
                   "goal-bottom-left", "goal-middle-up", "goal-bottom-right",
                   "other"]

    for annotation in annotations:
        cls, x1, y1, x2, y2 = yolo_to_bbox((height, width), annotation)
        class_name = class_names[cls]
        if class_name in goal_sections:
            goal_sections[class_name] = (x1, y1, x2, y2)

    # Check if all sections are detected
    if None in goal_sections.values():
        return None

    # Combine the sections to form the goal coordinates
    goal_points = [
        (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1]),  # Top-left
        (goal_sections["goal-top-right"][2], goal_sections["goal-top-right"][1]),  # Top-right
        (goal_sections["goal-bottom-right"][2], goal_sections["goal-bottom-right"][3]),  # Bottom-right
        (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][3])  # Bottom-left
    ]
    return goal_points

def process_frames(folder_path, annotations_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()  # Sort files if necessary

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        annotation_file = os.path.splitext(image_file)[0] + ".txt"
        annotation_path = os.path.join(annotations_path, annotation_file)

        frame = cv2.imread(image_path)
        if frame is None:
            continue

        if not os.path.exists(annotation_path):
            continue

        with open(annotation_path, 'r') as f:
            annotations = [list(map(float, line.strip().split())) for line in f.readlines()]

        # Detect goal in the current frame using YOLO annotations
        goal_points = detect_goal_with_yolo(frame, annotations)

        if goal_points and len(goal_points) == 4:
            # Compute the homography matrix for the current frame
            homography_matrix = compute_homography(goal_points)

            # Optionally, apply the homography to verify
            mapped_points = apply_homography(homography_matrix, goal_points)

            # Draw the detected goal and transformed points on the frame
            for points in goal_points:
                cv2.circle(frame, (int(points[0]), int(points[1])), 5, (0, 255, 0), -1)
            for points in mapped_points:
                cv2.circle(frame, (int(points[0]), int(points[1])), 5, (255, 0, 0), -1)

            # Display the homography matrix (optional)
            print(f"Frame: {image_file}")
            print("Homography Matrix:\n", homography_matrix)

        # Display the processed frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Example usage with a folder path
frames_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/dataset/test/images'
annotations_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/dataset/test/labels'
process_frames(frames_path, annotations_path)
print("Done")
