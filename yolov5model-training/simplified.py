from yolov5 import YOLOv5
from PIL import Image

# Specify the path to your custom trained model
model_path = "/yolov5model-training/best.pt"
model = YOLOv5(model_path)

# If you have an image to run detection on, load it with an appropriate library
# For example, using PIL:
img = Image.open("/yolov5model-training/data0s/test/images/ezgif-frame-142_jpg.rf.b8886e4b4b5085e5c43209a920386563.jpg")

# Perform inference
results = model.predict(img)

# Display or process the results
print(results)
results.show()
