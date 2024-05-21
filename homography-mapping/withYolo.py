import cv2
import numpy as np
import os
import torch

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


def detect_goal_yolov5(frame, model):
    results = model(frame)
    detections = results.xyxy[0]  # Get the first (and only) batch

    # Initialize dictionary to hold section coordinates
    goal_sections = {
        "goal-top-left": None,
        "goal-middle-down": None,
        "goal-top-right": None,
        "goal-bottom-left": None,
        "goal-middle-up": None,
        "goal-bottom-right": None
    }

    # Extract detected goal sections
    for *box, conf, cls in detections:
        if conf > 0.3:
            class_name = model.names[int(cls)]
            if class_name in goal_sections:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                goal_sections[class_name] = (x1, y1, x2, y2)

    # Check if all sections are detected
    if None in goal_sections.values():
        return None  # Return None if any section is missing

    # Combine the sections to form the goal coordinates
    goal_points = [
        (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1]),  # Top-left
        (goal_sections["goal-top-right"][2], goal_sections["goal-top-right"][1]),  # Top-right
        (goal_sections["goal-bottom-right"][2], goal_sections["goal-bottom-right"][3]),  # Bottom-right
        (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][3])  # Bottom-left
    ]

    print(goal_points)
    return goal_points


def process_frames(folder_path, model):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()  # Sort files if necessary

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        # Detect goal in the current frame using YOLOv5
        goal_points = detect_goal_yolov5(frame, model)
        print(goal_points)

        if goal_points and len(goal_points) == 4:
            print("I'm here")
            # Compute the homography matrix for the current frame
            H = compute_homography(goal_points)

            # Optionally, apply the homography to verify
            mapped_points = apply_homography(H, goal_points)

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


# Load YOLOv5 model locally
model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/.venv/lib/python3.10/site-packages/yolov5'  # Replace with the actual path to your local YOLOv5 repository
weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt'
model = torch.hub.load(model_path, 'custom', path=weights_path, source='local', force_reload=True)

# Load YOLOv5 model
# model = torch.hub.load('/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/yolov5m.yaml', 'custom', path='/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt', force_reload=True)

# Example usage with a folder path
folder_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/dataset/test/images'
process_frames(folder_path, model)
print("Done")
