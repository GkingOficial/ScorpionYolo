import argparse, yaml, torch, cv2
from pathlib import Path
from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8x.pt")
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    model.train(
        data="scorpion.yaml",
        epochs=500,
        imgsz=640,
        batch=3,
        #mosaic=False,
        device="cuda",
    )
    torch.cuda.empty_cache()

def run_inference():
    torch.cuda.empty_cache()
    device = "cuda"

    #mudar o modelo sempre depois de treinar
    model = YOLO("runs/detect/train5/weights/best.pt")

    #alterar para testar as imagens
    img = cv2.imread("./images/ug_Arizona_Bark_Scorpion_2.jpg")
    results = model(img, device=device)

    annotated = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True, help="Modo: 'train' para treinar ou 'infer' para rodar inferência")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "infer":
        run_inference()