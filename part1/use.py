import cv2
import torch
import torch.nn as nn
import numpy as np


# ==========================================
# 1. Архитектура нейросети
# ==========================================
class ContourNet(nn.Module):
    def __init__(self):
        super(ContourNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(16, 1, 2, stride=2), nn.Sigmoid())

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x


def get_flow_contours(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    grad_x = cv2.Sobel(magnitude, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(magnitude, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)

    edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
    edges[edges < 0.2] = 0.0
    return edges


# ==========================================
# 3. Применение к фотографиям
# ==========================================
def apply_to_photos(img1_path, img2_path, model_path="contour_net.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = ContourNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ОШИБКА: Файл '{model_path}' не найден.")
        return

    model.eval()

    # Загрузка изображений с диска
    frame1 = cv2.imread(img1_path)
    frame2 = cv2.imread(img2_path)

    if frame1 is None or frame2 is None:
        print("ОШИБКА: Не удалось загрузить одну или обе фотографии. Проверьте пути!")
        return

    # Приводим к размеру, на котором училась сеть
    img_size = (128, 128)
    frame1_resized = cv2.resize(frame1, img_size)
    frame2_resized = cv2.resize(frame2, img_size)

    # --- ЭТАП 1: Оптический поток (требует ОБА фото) ---
    flow_contours = get_flow_contours(frame1_resized, frame2_resized)
    flow_vis = (flow_contours * 255).astype(np.uint8)

    # --- ЭТАП 2: Нейросеть (требует ТОЛЬКО ПЕРВОЕ фото) ---
    img_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
    input_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    mask = output.squeeze().cpu().numpy()
    mask_vis = (mask * 255).astype(np.uint8)
    _, nn_thresh = cv2.threshold(mask_vis, 127, 255, cv2.THRESH_BINARY)

    # --- ЭТАП 3: Визуализация ---
    # Увеличим картинки, чтобы было удобнее рассматривать
    display_size = (400, 400)
    display_orig = cv2.resize(frame1_resized, display_size)
    display_flow = cv2.resize(flow_vis, display_size)
    display_nn = cv2.resize(nn_thresh, display_size)

    # Собираем все три картинки в одну горизонтальную панель
    # (Оригинал) | (Оптический поток - 2 фото) | (Нейросеть - 1 фото)
    display_flow_bgr = cv2.cvtColor(display_flow, cv2.COLOR_GRAY2BGR)

    combined_image = np.hstack((display_orig, display_flow_bgr))

    print("Показываю результаты. Нажмите любую клавишу в окне картинки, чтобы закрыть.")
    cv2.imshow('Left: Original | Mid: Flow (2 photos) | Right: NN (1 photo)', combined_image)

    # Ждем нажатия любой клавиши
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Укажите здесь пути к вашим двум фотографиям
    # Фотографии должны быть сделаны с одной точки с небольшим смещением объекта
    apply_to_photos("DICOM\scan_022.png", "DICOM\scan_023.png")