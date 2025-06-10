import os
import cv2

images_dir = "tools/OIDv4_ToolKit/OID/Dataset/train/Scorpion"
labels_dir = "tools/OIDv4_ToolKit/OID/Dataset/train/Scorpion/Label"
output_labels_dir = "tools/OIDv4_ToolKit/OID/Dataset/train/Scorpion/YOLO_Labels"

os.makedirs(output_labels_dir, exist_ok=True)

for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    img_name = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, label_file)
    output_path = os.path.join(output_labels_dir, label_file)

    if not os.path.exists(img_path):
        print(f"Imagem não encontrada para {label_file}, pulando.")
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    with open(output_path, "w") as out:
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            _, xmin, ymin, xmax, ymax = parts
            xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])

            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            
            out.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Conversão concluída! Labels YOLO salvos em:", output_labels_dir)
