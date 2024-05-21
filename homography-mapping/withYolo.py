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


def detect_goal_yolo(frame, net, output_layers, classes):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "goal":  # Replace "goal" with your actual class label
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    goal_points = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # Top-left, Top-right, Bottom-right, Bottom-left
            goal_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            break

    return goal_points


def process_frames(folder_path, net, output_layers, classes):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()  # Sort files if necessary

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        # Detect goal in the current frame using YOLO
        goal_points = detect_goal_yolo(frame, net, output_layers, classes)

        if goal_points and len(goal_points) == 4:
            # Compute the homography matrix for the current frame
            H = compute_homography(goal_points)

            # Optionally, apply the homography to verify
            mapped_points = apply_homography(H, goal_points)

            # Draw the detected goal and transformed points on the frame
            for pt in goal_points:
                cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
            for pt in mapped_points:
                cv2.circle(frame, tuple(pt), 5, (255, 0, 0), -1)

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


# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Example usage with a folder path
folder_path = 'path/to/your/frames_folder'
process_frames(folder_path, net, output_layers, classes)
