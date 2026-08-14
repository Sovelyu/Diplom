import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader



class ContourNet(nn.Module):
    def __init__(self):
        super(ContourNet, self).__init__()
        # Энкодер (сжатие)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        # Декодер (восстановление)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(16, 1, 2, stride=2), nn.Sigmoid())

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x


# ==========================================

def get_flow_contours(frame1, frame2):
    """
    Вычисляет оптический поток между кадрами и извлекает границы движения.
    """
    # Перевод в градации серого
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Плотный оптический поток Фарнебака
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Перевод вектора потока в полярные координаты (величина и угол)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Выделение границ (градиентов) в оптическом потоке с помощью Собеля
    grad_x = cv2.Sobel(magnitude, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(magnitude, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)

    # Нормализация от 0 до 1
    edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

    # Очистка от шума (пороговая фильтрация)
    edges[edges < 0.2] = 0.0
    return edges


class VideoDataset(Dataset):
    def __init__(self, root_dir, img_size=(128, 128)):
        self.root_dir = root_dir
        self.img_size = img_size

        # Получаем список всех файлов в папке и сортируем их,
        # чтобы кадры шли строго друг за другом (frame_001, frame_002...)
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        # Количество пар будет на 1 меньше, чем всего картинок в папке
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        # Формируем полные пути к текущему и следующему кадрам
        img1_path = os.path.join(self.root_dir, self.image_files[idx])
        img2_path = os.path.join(self.root_dir, self.image_files[idx + 1])

        # Читаем изображения с диска с помощью OpenCV
        frame1 = cv2.imread(img1_path)
        frame2 = cv2.imread(img2_path)

        # Изменяем размер под требования нейросети
        frame1 = cv2.resize(frame1, self.img_size)
        frame2 = cv2.resize(frame2, self.img_size)

        # Вычисляем контуры через оптический поток (наша функция из Части 3)
        pseudo_label = get_flow_contours(frame1, frame2)

        # Превращаем в тензоры PyTorch
        frame1_tensor = torch.tensor(frame1, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label_tensor = torch.tensor(pseudo_label, dtype=torch.float32).unsqueeze(0)

        return frame1_tensor, label_tensor


# ==========================================
# 4. Основной цикл обучения
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на устройстве: {device}")

    # Инициализация сети, датасета и оптимизатора
    model = ContourNet().to(device)
    dataset = VideoDataset(root_dir=r"D:\nir\ColonCancerCT-2025 A Dataset of Abdominal CT Scans\Split_Data\Split_Data\test\Non_Cancer")  # В реальности передайте пути к вашим видео
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Предсказание сети (только по одному кадру!)
            outputs = model(inputs)

            # Сравнение предсказания с контурами из оптического потока
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Эпоха [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

    print("Обучение завершено. Теперь сеть умеет находить контуры на одиночных фото.")
    return model


if __name__ == "__main__":
    trained_model = train()
    torch.save(trained_model.state_dict(), "contour_net.pth")