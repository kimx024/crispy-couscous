import numpy as np
import cv2
# import matplotlib.pyplot as plt
import torch
import os

"""
DON'T EDIT THIS FUCKING FILE ANYMORE IT WORKS??>?>??>
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


def establish_goal(information_list, model):
    goal_points = []
    goal_sections = {
        "goal-top-left": None,
        "goal-middle-down": None,
        "goal-top-right": None,
        "goal-bottom-left": None,
        "goal-middle-up": None,
        "goal-bottom-right": None
    }

    for index, frame in enumerate(information_list):
        detection = frame.xyxy[0]
        # Debugging: Print out the detection data for each frame
        # print(f"Frame {index + 1} detections: {detection}")

        # Extract detected goal sections
        for *box, confidence, class_labels in detection:
            if confidence >= 0.3:
                class_name = model.names[int(class_labels)]
                if class_name in goal_sections:
                    x1, x2 = int(box[0]), int(box[2])
                    y1, y2 = int(box[1]), int(box[3])
                    # Debugging: Print out the details of the detected goal sections
                    # print(f"Detected class: { class_name} in frame {index + 1}: "
                    #       f"\n (x1={x1}, y1={y1}, x2={x2}, y2={y2}) \n")
                    goal_sections[class_name] = (x1, y1, x2, y2)
                    # print(goal_sections

        # print(f"\n State of goal sections after frame {index + 1}: {goal_sections}")
        # print("---------------")

        if all(goal_sections[section] is not None for
               section in ["goal-bottom-left", "goal-bottom-right", "goal-top-left", "goal-top-right"]):
            goal_points = [
                (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][3]),  # Bottom-left
                (goal_sections["goal-bottom-right"][2], goal_sections["goal-bottom-right"][3]),  # Bottom-right
                (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1]),  # Top-left
                (goal_sections["goal-top-right"][2], goal_sections["goal-top-right"][1]),  # Top-right
            ]

            # print(f"In frame {index + 1} the corners of the goal are: {goal_points}")
        else:
            # print(f"In frame {index + 1}, not all goal sections were detected.")
            continue
    return goal_points


def determine_homography_goal(goal_points, ref_points) -> np.ndarray:
    """
    This function determines the homography of the full goal. Can be redundant perhaps when applying goal sections.
    The input of the second variable declaration are the full goal coordinates of the corners of the goal,
    calculated in the establish_goal function. The second is an array of the new or destined image.

    When projection = 1 it means that the application is valid and the points are successfully used.
    :param goal_points:
    :return: homography_goal_coordinates: np.ndarray of shape (3, 2)
    """
    goal_coordinates = np.array(goal_points, dtype="float32")
    h_goal_coordinates, projection = cv2.findHomography(goal_coordinates, ref_points)
    # print(f"Homography: {h_goal_coordinates} has type {type(h_goal_coordinates)}"
    #       f"\n Projection: {projection} has type {type(projection)}")
    return h_goal_coordinates


def apply_homography_to_image(h_goal_coords, drawn_points):
    transformation_matrix = cv2.getPerspectiveTransform(
        np.array([drawn_points], dtype="float32"), h_goal_coords)
    return transformation_matrix[0]


def process_images(directory, model, ref_points):
    # Read all the images from the folder
    directory_files = read_folder(directory)

    # Get coordinates from the model for all images
    information_list = get_coordinates(directory_files, model)

    # Establish goal points for each image
    counter = 0
    for filename in directory_files:
        frame = cv2.imread(filename)
        info = information_list[counter]
        goal_points = establish_goal([info], model)

        # Debugging: Print the goal_points to ensure they are correct
        print(f"Goal points: {goal_points}")

        if len(goal_points) == 4 and all(isinstance(point, (tuple, list)) and len(point) == 2 for point in goal_points):
            # Determine the homography for the goal
            h_goal_coords = determine_homography_goal(goal_points, ref_points)

            # Map the goal points using the homography matrix
            goal_points_array = np.array(goal_points, dtype="float32")
            goal_points_array = np.array([goal_points_array])
            mapped_points = cv2.perspectiveTransform(goal_points_array, h_goal_coords)[0]

            # Draw the detected goal and transformed points on the frame
            for point in goal_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)
            for point in mapped_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

            # Display the image with the detected and transformed points
            cv2.imshow('Transformed Image', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        counter += 1
        print(f"Processing file: {filename}, counter: {counter}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define the model path and weights path
    model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/.venv/lib/python3.10/site-packages/yolov5'
    weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt'

    # Load the custom YOLOv5 model
    load_model = torch.hub.load(model_path, 'custom', path=weights_path, source='local', force_reload=True)
    # Load the files
    file_path = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/homography-mapping/footage"

    # Functions to execute
    # all_files = read_folder(file_path)
    # coordinates = get_coordinates(all_files, load_model)
    # # label_information = get_label_information(coordinates)
    # full_goal = establish_goal(coordinates, load_model)
    # determine_homography_goal(full_goal)
    process_images(file_path, load_model, reference_points)
    print("---------------------")
    print("Done processing, this code has successfully been executed.")
