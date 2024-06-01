# import numpy as np
# import cv2
# # import matplotlib.pyplot as plt
# import torch
# import os
#
# """
# Code what uses original labelling instead of the whole model
# """
#
#
# def read_folder(image_directory, label_directory):
#     image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('png', 'jpg', 'jpeg'))]
#     label_files = [os.path.join(label_directory, os.path.splitext(f)[0] + '.txt') for f in os.listdir(image_directory) if f.endswith(('png', 'jpg', 'jpeg'))]
#     image_files.sort()
#     label_files.sort()
#     counter = len(image_files)  # Update counter to reflect the number of files
#     print(f"Folder processing completed with {counter} files")
#     return image_files, label_files
#
#
# def read_labels(label_files):
#     label_list = []
#     for label_file in label_files:
#         with open(label_file, 'r') as file:
#             labels = file.readlines()
#             parsed_labels = [list(map(float, label.strip().split())) for label in labels]
#             label_list.append(parsed_labels)
#     return label_list
#
#
# def get_coordinates(label_list):
#     information_list = []
#     for labels in label_list:
#         info = []
#         for label in labels:
#             class_id, x_center, y_center, width, height = label
#             x1 = (x_center - width / 2) * 1920  # Assuming the image width is 1920
#             y1 = (y_center - height / 2) * 1080  # Assuming the image height is 1080
#             x2 = (x_center + width / 2) * 1920
#             y2 = (y_center + height / 2) * 1080
#             confidence = 1.0  # Since we are using labels directly
#             info.append([x1, y1, x2, y2, confidence, class_id])
#         information_list.append(info)
#     return information_list
#
#
# def get_label_information(information_list):
#     """
#     This function is just a decoy function to backtrack
#     :param information_list:
#     :return:
#     """
#     if information_list:
#         print("")
#         for i in information_list:
#             print(i.pandas().xyxy[0])
#     return information_list
#
#
# def establish_goal(information_list, model):
#     goal_points = []
#     goal_sections = {
#         "goal-top-left": None,
#         "goal-middle-down": None,
#         "goal-top-right": None,
#         "goal-bottom-left": None,
#         "goal-middle-up": None,
#         "goal-bottom-right": None
#     }
#
#     for index, frame_info in enumerate(information_list):
#         detection = frame_info  # Now it's directly the list of detections
#         # Debugging: Print out the detection data for each frame
#         print(f"Frame {index + 1} detections: {detection}")
#
#         # Extract detected goal sections
#         for detection_info in detection:
#             *box, confidence, class_labels = detection_info
#             if confidence >= 0.3:
#                 class_name = model.names[int(class_labels)]
#                 if class_name in goal_sections:
#                     x1, x2 = int(box[0]), int(box[2])
#                     y1, y2 = int(box[1]), int(box[3])
#                     # Debugging: Print out the details of the detected goal sections
#                     print(f"Detected class: {class_name} in frame {index + 1}: "
#                           f"\n (x1={x1}, y1={y1}, x2={x2}, y2={y2}) \n")
#                     goal_sections[class_name] = (x1, y1, x2, y2)
#
#         print(f"\nState of goal sections after frame {index + 1}: {goal_sections}")
#         print("---------------")
#
#         if all(goal_sections[section] is not None for section in ["goal-bottom-left", "goal-bottom-right", "goal-top-left", "goal-top-right"]):
#             goal_points = [
#                 (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][3]),  # Bottom-left
#                 (goal_sections["goal-bottom-right"][2], goal_sections["goal-bottom-right"][3]),  # Bottom-right
#                 (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1]),  # Top-left
#                 (goal_sections["goal-top-right"][2], goal_sections["goal-top-right"][1]),  # Top-right
#             ]
#             print(f"Goal points identified in frame {index + 1}: {goal_points}")
#             break  # Exit loop early if all goal sections are found
#         else:
#             print(f"Not all goal sections detected in frame {index + 1}")
#
#     if not goal_points:
#         print("Goal points could not be identified.")
#
#     return goal_points
#
#
#
#
#
# def determine_homography_goal(goal_points, reference_points) -> np.ndarray:
#     """
#     This function determines the homography of the full goal.
#     :param goal_points: List of detected goal points.
#     :param reference_points: List of reference points to map the goal points to.
#     :return: homography_goal_coordinates: np.ndarray of shape (3, 3)
#     """
#     goal_coordinates = np.array(goal_points, dtype="float32")
#     h_goal_coordinates, projection = cv2.findHomography(goal_coordinates, reference_points)
#     print(f"Homography: {h_goal_coordinates} has type {type(h_goal_coordinates)}"
#           f"\nProjection: {projection} has type {type(projection)}")
#     return h_goal_coordinates
#
#
# def apply_homography_to_image(img, h_goal_coords, goal_points):
#     """
#     Apply the homography matrix to the image to get the transformed points.
#     :param img: The image to apply the homography on.
#     :param h_goal_coords: Homography matrix.
#     :param goal_points: List of detected goal points.
#     :return: Transformed points.
#     """
#     goal_points_array = np.array(goal_points, dtype="float32")
#     goal_points_array = np.array([goal_points_array])
#     mapped_points = cv2.perspectiveTransform(goal_points_array, h_goal_coords)[0]
#
#     # Draw the detected goal and transformed points on the frame
#     for point in goal_points:
#         cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
#     for point in mapped_points:
#         cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
#
#     return img
#
#
# def process_images(image_directory, label_directory, model, reference_points):
#     # Read all the images and labels from the folder
#     image_files, label_files = read_folder(image_directory, label_directory)
#
#     # Get label information from the label files
#     label_list = read_labels(label_files)
#
#     # Get coordinates from the labels for all images
#     information_list = get_coordinates(label_list)
#
#     # Establish goal points for each image
#     counter = 0
#     for filename in image_files:
#         img = cv2.imread(filename)
#         info = information_list[counter]
#
#         # Debugging: Print the info and type to understand the content
#         print(f"Processing file {filename}, counter: {counter}")
#         print(f"Info type: {type(info)}")
#
#         goal_points = establish_goal([info], model)
#
#         # Debugging: Print the goal_points to ensure they are correct
#         print(f"Goal points: {goal_points}")
#
#         if len(goal_points) == 4 and all(isinstance(point, (tuple, list)) and len(point) == 2 for point in goal_points):
#             # Determine the homography for the goal
#             h_goal_coords = determine_homography_goal(goal_points, reference_points)
#
#             # Apply the homography to the image and draw the points
#             img = apply_homography_to_image(img, h_goal_coords, goal_points)
#
#             # Display the image with the detected and transformed points
#             cv2.imshow('Transformed Image', img)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#
#         counter += 1
#         print(f"Processing file {filename}, counter: {counter}")
#     cv2.destroyAllWindows()
#
#
#
# if __name__ == "__main__":
#     # Define the model path and weights path
#     model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/.venv/lib/python3.10/site-packages/yolov5'
#     weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt'
#
#     # Load the custom YOLOv5 model
#     load_model = torch.hub.load(model_path, 'custom', path=weights_path, source='local', force_reload=True)
#     # Load the files
#     image_path = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model-dataset/test/images"
#     label_path = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model-dataset/test/labels"
#
#     # Define reference points
#     reference_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")
#
#     # Functions to execute
#     process_images(image_path, label_path, load_model, reference_points)
#     print("Done")
