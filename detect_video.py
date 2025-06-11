from ultralytics import YOLO
import cv2


video_path = "./videos/Scorpion_Amazing_Animals.mp4"
model_path = "runs/detect/train3/weights/best.pt"

# model = YOLO("yolov8m.pt")
model = YOLO(model_path)
cap = cv2.VideoCapture(0)

# save_output = True
# if save_output:
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     if fps == 0 or frame_width == 0 or frame_height == 0:
#         fps = 30
#         frame_width, frame_height = 360, 640
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter("saida_detectada.mp4", fourcc, fps, (frame_width, frame_height))
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)[0]

    annotated_frame = results.plot()

    cv2.imshow("YOLOv8 - Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
# if save_output:
#     out.release()
cv2.destroyAllWindows()
