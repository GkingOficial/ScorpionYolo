import argparse
import yaml
import torch
import cv2

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

# ─── 1) Dataset YOLO-format ────────────────────────────────────────────────────
class ScorpionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, imgsz=640):
        self.img_files   = sorted(Path(images_dir).rglob('*.*'))
        self.label_files = {p.stem: p for p in Path(labels_dir).rglob('*.txt')}
        self.imgsz       = imgsz

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Carrega imagem, converte BGR→RGB e redimensiona
        p   = self.img_files[idx]
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # C×H×W

        # Lê caixas [cls, x_center, y_center, w, h]
        lbl_path = self.label_files.get(p.stem, None)
        if lbl_path is None:
            boxes = torch.zeros((0, 5))
        else:
            data = []
            for line in open(lbl_path, 'r').read().splitlines():
                cls, x, y, w, h = map(float, line.split())
                data.append([cls, x, y, w, h])
            boxes = torch.tensor(data, dtype=torch.float32) if data else torch.zeros((0,5))
        return img, boxes

# ─── 2) collate_fn para montar o batch no formato que o YOLO loss espera ─────
def yolo_collate_fn(batch):
    imgs    = torch.stack([x[0] for x in batch], 0)  # B×3×H×W
    targets = [x[1] for x in batch]                 # lista de Tensor(N_i,5)

    batch_idx, cls, bboxes = [], [], []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        n = t.shape[0]
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        cls.append(t[:, 0].long())
        bboxes.append(t[:, 1:].float())

    if batch_idx:
        batch_idx = torch.cat(batch_idx, 0)
        cls       = torch.cat(cls, 0)
        bboxes    = torch.cat(bboxes, 0)
    else:
        batch_idx = torch.tensor([], dtype=torch.long)
        cls       = torch.tensor([], dtype=torch.long)
        bboxes    = torch.empty((0, 4), dtype=torch.float32)

    return imgs, {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

# ─── 3) Função de treino manual ────────────────────────────────────────────────
def train_manual(yaml_data="scorpion.yaml", model_path="yolov8n.pt", epochs=50):
    # 3.1) Configura device, modelo, optimizer e scaler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando device:", device)

    # carrega backbone YOLOv8 (n, s, m, l, x)
    yolo   = YOLO(model_path)
    model  = yolo.model.to(device)  # pega o nn.Module interno
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scaler = torch.amp.GradScaler()

    # 3.2) Lê YAML e monta dataloaders
    cfg = yaml.safe_load(open(yaml_data))
    base = Path(cfg["train"]).parents[1]  # .../scorpion_dataset

    train_imgs = cfg["train"]
    val_imgs   = cfg["val"]
    train_lbls = base / "labels" / "train"
    val_lbls   = base / "labels" / "val"

    # print(f'Base: {base}')
    # print(f'imgs: {train_imgs}\n{val_imgs}')
    # print(f'labels: {train_lbls}\n{val_lbls}')
    # input()

    train_ds = ScorpionDataset(train_imgs, str(train_lbls), imgsz=640)
    val_ds   = ScorpionDataset(val_imgs,   str(val_lbls),   imgsz=640)

    train_dl = DataLoader(train_ds,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          collate_fn=yolo_collate_fn)

    val_dl   = DataLoader(val_ds,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          collate_fn=yolo_collate_fn)

    # 3.3) Loop de épocas
    for epoch in range(1, epochs + 1):
        # — train —
        model.train()
        for imgs, batch in train_dl:
            imgs  = imgs.to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optim.zero_grad()
            with torch.amp.autocast(device_type=device):
                # 1) forward normal de detecção
                preds = model(imgs)

                # 2) loss, agora sim passando o batch correto
                loss, loss_items = model.loss(preds, batch)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        # — val —
        model.eval()
        with torch.no_grad():
            for imgs, batch in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                _ = model({"img": imgs, **batch})

        # libera cache CUDA
        torch.cuda.empty_cache()
        print(f"Época {epoch}/{epochs} concluída — cache liberado")

    # 3.4) salva pesos finais
    yolo.save("runs/detect/manual_train/weights/best.pt")


# ─── 4) Função de inferência ───────────────────────────────────────────────────
def run_inference(weights="runs/detect/manual_train/weights/best.pt",
                  img_path="./images/pexels-sharath-g-1981542_ver_1.jpg"):
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(weights)
    img   = cv2.imread(img_path)
    results = model(img, device=device)

    annotated = results[0].plot()
    cv2.imshow("Detecção YOLOv8", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── 5) CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train_manual()
    else:
        run_inference()
