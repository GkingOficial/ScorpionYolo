from ultralytics import YOLO
import cv2

model_path = "runs/detect/train/weights/best.pt"

model = YOLO(model_path)
# model = YOLO("yolov8n.pt")

img = cv2.imread("./images/pexels-sharath-g-1981542_ver_1.jpg")

results = model(img)

annotated = results[0].plot()
cv2.imshow("YOLOv8 Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
