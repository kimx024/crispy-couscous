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


def detect_goal_with_yolo(frame, model):
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

    missing_sections = [key for key, value in goal_sections.items() if value is None]

    if missing_sections:
        print(f"Missing sections: {missing_sections}")
        # Implement simple fallback logic
        if "goal-top-left" in missing_sections and goal_sections["goal-bottom-left"]:
            goal_sections["goal-top-left"] = (
            goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][1] - 50)
        if "goal-top-right" in missing_sections and goal_sections["goal-bottom-right"]:
            goal_sections["goal-top-right"] = (
            goal_sections["goal-bottom-right"][0], goal_sections["goal-bottom-right"][1] - 50)
        if "goal-bottom-left" in missing_sections and goal_sections["goal-top-left"]:
            goal_sections["goal-bottom-left"] = (
            goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1] + 50)
        if "goal-bottom-right" in missing_sections and goal_sections["goal-top-right"]:
            goal_sections["goal-bottom-right"] = (
            goal_sections["goal-top-right"][0], goal_sections["goal-top-right"][1] + 50)

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

    if missing_sections:
        print(f"Missing sections: {missing_sections}")
        # Implement simple fallback logic
        if "goal-top-left" in missing_sections and goal_sections["goal-bottom-left"]:
            goal_sections["goal-top-left"] = (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][1] - 50)
        if "goal-top-right" in missing_sections and goal_sections["goal-bottom-right"]:
            goal_sections["goal-top-right"] = (goal_sections["goal-bottom-right"][0], goal_sections["goal-bottom-right"][1] - 50)
        if "goal-bottom-left" in missing_sections and goal_sections["goal-top-left"]:
            goal_sections["goal-bottom-left"] = (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1] + 50)
        if "goal-bottom-right" in missing_sections and goal_sections["goal-top-right"]:
            goal_sections["goal-bottom-right"] = (goal_sections["goal-top-right"][0], goal_sections["goal-top-right"][1] + 50)


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
        goal_points = detect_goal_with_yolo(frame, model)

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


if __name__ == '__main__':
    # Load YOLOv5 model locally
    model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/.venv/lib/python3.10/site-packages/yolov5'
    weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt'
    yolo_model = torch.hub.load(model_path, 'custom', path=weights_path, source='local', force_reload=True)

    # Example usage with a folder path
    frames = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/dataset/train/images'
    process_frames(frames, yolo_model)
    print("Done")
