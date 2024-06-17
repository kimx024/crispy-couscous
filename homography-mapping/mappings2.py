import numpy as np
import cv2
import torch
import os

"""
DON'T EDIT THIS FILE ANYMORE IT WORKS
Detecting: goalkeeper + ball + output generation
"""


reference_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")

def read_folder(directory):
    directory_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))]
    directory_files.sort()
    counter = len(directory_files)  # Update counter to reflect the number of files
    print(f"Folder processing completed: {counter} files are read")
    return directory_files

def get_coordinates(directory_files, model):
    counter = 0
    information_list = []
    for filename in directory_files:
        info = model(filename)
        counter += 1
        information_list.append(info)
    return information_list

def get_label_information(information_list):
    """
    This function is just a decoy function to backtrack
    :param information_list:
    :return:
    """
    if information_list:
        print("")
        for i in information_list:
            print(i.pandas().xyxy[0])
    return information_list

def establish_goal(frame, model):
    goal_sections = {
        "goal-top-left": None,
        "goal-middle-down": None,
        "goal-top-right": None,
        "goal-bottom-left": None,
        "goal-middle-up": None,
        "goal-bottom-right": None
    }
    goalkeeper_position = None

    detection = frame.xyxy[0]

    # Extract detected goal sections and goalkeeper
    for *box, confidence, class_labels in detection:
        if confidence >= 0.3:
            class_name = model.names[int(class_labels)]
            if class_name in goal_sections:
                x1, y1, x2, y2 = map(int, box)
                goal_sections[class_name] = (x1, y1, x2, y2)
            elif class_name == "goalkeeper":
                x1, y1, x2, y2 = map(int, box)
                goalkeeper_position = (x1, y1, x2, y2)

    goal_points = []
    if all(goal_sections[section] is not None for section in ["goal-bottom-left", "goal-bottom-right", "goal-top-left", "goal-top-right"]):
        goal_points = [
            (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][3]),  # Bottom-left
            (goal_sections["goal-bottom-right"][2], goal_sections["goal-bottom-right"][3]),  # Bottom-right
            (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1]),  # Top-left
            (goal_sections["goal-top-right"][2], goal_sections["goal-top-right"][1]),  # Top-right
        ]
    return goal_points, goal_sections, goalkeeper_position

def detect_ball(frame, model):
    ball_position = None
    detection = frame.xyxy[0]

    for *box, confidence, class_labels in detection:
        if confidence >= 0.3:
            class_name = model.names[int(class_labels)]
            if class_name == "football":  # Assuming the class name for the ball in the yolov5-training is 'football'
                x1, y1, x2, y2 = map(int, box)
                ball_position = (x1, y1, x2, y2)
                break  # Assuming only one ball is present, exit after finding the first one

    return ball_position

def determine_homography_goal(goal_points, ref_points) -> np.ndarray:
    goal_coordinates = np.array(goal_points, dtype="float32")
    h_goal_coordinates, projection = cv2.findHomography(goal_coordinates, ref_points)
    return h_goal_coordinates

def process_images(directory, model, ref_points):
    directory_files = read_folder(directory)
    information_list = get_coordinates(directory_files, model)

    section_colors = {
        "goal-top-left": (255, 0, 0),  # Blue
        "goal-middle-down": (0, 255, 0),  # Green
        "goal-top-right": (0, 0, 255),  # Red
        "goal-bottom-left": (255, 255, 0),  # Cyan
        "goal-middle-up": (255, 0, 255),  # Magenta
        "goal-bottom-right": (0, 255, 255)  # Yellow
    }

    goalkeeper_positions = []
    ball_positions = []

    counter = 0
    for filename in directory_files:
        frame = cv2.imread(filename)
        info = information_list[counter]
        goal_points, goal_sections, goalkeeper_position = establish_goal(info, model)

        # If not all goal sections are detected, skip the image
        if not goal_points:
            print(f"Skipping file {filename} as not all goal sections are detected.")
            counter += 1
            continue

        # Detect the ball
        ball_position = detect_ball(info, model)
        if ball_position:
            x1, y1, x2, y2 = ball_position
            ball_center = [(x1 + x2) // 2, (y1 + y2) // 2]
            ball_positions.append(ball_center)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for the ball
            print(f"Ball position: {ball_position}")

        # Draw the bounding boxes for the goal sections
        for section, box in goal_sections.items():
            if box:
                x1, y1, x2, y2 = box
                color = section_colors[section]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw the bounding box for the goalkeeper
        if goalkeeper_position:
            x1, y1, x2, y2 = goalkeeper_position
            goalkeeper_center = [(x1 + x2) // 2, (y1 + y2) // 2]
            goalkeeper_positions.append(goalkeeper_center)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for the goalkeeper
            print(f"Goalkeeper position: {goalkeeper_position}")

        print(f"Goal points: {goal_points}")

        if len(goal_points) == 4 and all(isinstance(point, (tuple, list)) and len(point) == 2 for point in goal_points):
            h_goal_coords = determine_homography_goal(goal_points, ref_points)

            goal_points_array = np.array(goal_points, dtype="float32")
            goal_points_array = np.array([goal_points_array])
            mapped_points = cv2.perspectiveTransform(goal_points_array, h_goal_coords)[0]

            for point in goal_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)
            for point in mapped_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

        cv2.imshow('Transformed Image', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        counter += 1
        print(f"Processing file: {filename}, counter: {counter}")

    cv2.destroyAllWindows()
    return goalkeeper_positions, ball_positions

def generate_output(goalkeeper_positions, ball_positions, output_file='output.txt'):
    with open(output_file, 'w') as f:
        f.write("[\n")
        for gk_pos, ball_pos in zip(goalkeeper_positions, ball_positions):
            f.write(f"(({gk_pos[0]}, {gk_pos[1]}), ({ball_pos[0]}, {ball_pos[1]})),\n")
        f.write("]\n")
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5/models/yolov5m.yaml'
    weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt'

    load_model = torch.hub.load(model_path, 'custom', path=weights_path, source='local', force_reload=True)
    file_path = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/homography-mapping/footage"

    goalkeeper_positions, ball_positions = process_images(file_path, load_model, reference_points)
    generate_output(goalkeeper_positions, ball_positions)
    print("---------------------")
    print("Done processing, this code has successfully been executed.")
