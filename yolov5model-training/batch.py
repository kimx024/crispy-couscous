import os
import yolov5
from PIL import Image

# load model
model = yolov5.load('keremberke/yolov5s-football')

# set model parameters
model.conf = 0.70 # NMS threshold. Sets the model's confidence threshold for non-maximum suppression (NMS)
# NMS is a technique to eliminate redundant bounding boxes that identify the same object with lower confidence scores

model.iou = 0.35 # IoU threshold. Sets the model's Intersection over Union (IuO) threshold.
# IoU measures the overlap between two bounding boxes.
# a lower IoU threshold allows more overlap, possibly leading to multiple boxes for the same object.

model.agnostic = False # NMS class-agnostic. Specifies that NMS should be class-specific.
# setting it to False means the suppression is performed seperately for each class.

model.multi_label = False # NMS multiple labels per box.
# False indicates that each bounding box can only have one class lable.

model.max_det = 1000 # maximum number of detections per image

# directory containing multiple images
input_dir = "/yolov5model-training/trial0/part0-frames"

# directory to save results
output_dir = "/yolov5model-training/trial0/part0-results"
os.makedirs(output_dir, exist_ok=True) # create output directory if it doesn't exist

# introduce counter
counter = 0

# list all images in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):

        # construct the full file path
        img_path = os.path.join(input_dir, filename)

        # perform inference
        results = model(img_path, size=640, augment=True)

        # render and save image with detections
        rendered_img = results.render()[0]
        img = Image.fromarray(rendered_img)
        img.save(os.path.join(output_dir, filename))
        counter += 1

# Exit message
print(f'{counter} images are processed.')
