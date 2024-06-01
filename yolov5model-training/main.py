"""
File for handling single images/frames as a error-handling check.
Model from keremberke is pretrained on images from real-football matches.
Make sure to fine-tune the model on FIFA images.

```
yolov5 train --data data.yaml --img 640 --batch 16 -- weights keremberke/yolov5s-football --epochs 10
```
"""

import yolov5
from PIL import Image

# Load the pretrained model with the custom weights (best.pt)
model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/.venv/lib/python3.10/site-packages/yolov5'
weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/finalBest.pt'

model = yolov5.load(weights_path)  # loading the custom weights
print("Model loaded successfully")

# set model parameters
model.conf = 0.55   # NMS threshold. Sets the model's confidence threshold for non-maximum suppression (NMS)
# NMS is a technique to eliminate redundant bounding boxes that identify the same object with lower confidence scores

model.iou = 0.4     # IoU threshold. Sets the model's Intersection over Union (IuO) threshold.
# IoU measures the overlap between two bounding boxes.
# a lower IoU threshold allows more overlap, possibly leading to multiple boxes for the same object.

model.agnostic = False  # NMS class-agnostic. Specifies that NMS should be class-specific.
# setting it to False means the suppression is performed separately for each class.

model.multi_label = False   # NMS multiple labels per box.
# False indicates that each bounding box can only have one class label.

model.max_det = 1000    # maximum number of detections per image

# set image path (singular image)
img = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model-dataset/test/images/0007_jpg.rf.5a175221727a95b8d0fb68fc787c13eb.jpg"

# perform inference with test time augmentation
# TTA can improve the robustness of the predictions by slightly altering the input data and aggregating the results.
results = model(img, size=640, augment=True)

# pars results in form [x1, y1, x2, y2, score, class-id]
predictions = results.pred[0] # The pred attribute contains the detection outputs
# [0] means the first and typically only batch of results (yolo-standard)

boxes = predictions[:, :4] # extracts the boxes according to x1, y1 (top left), x2, y2 (bottom right)
scores = predictions[:, 4] # extracts the confidence scores attached to the bounding boxes from array (slice)
categories = predictions[:, 5] # extracts the predicted category IDs for each detection from array (slice)

# show detection bounding boxes on image
results.show()

# render detections on images
rendered_image = results.render()

# save results into 'results' folder
# each result get saved into a different folder
# results.save(save_dir='results/')
#
# # each result get saved into the same folder
# for i, img in enumerate(rendered_image):
#     img = Image.fromarray(img)
#     img.save(f'results/detected_{i}.png')
#
# # end message
# print("images are processsed")

