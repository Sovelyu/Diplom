import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from PIL import Image
import os


# 1. Архитектура U-Net (Упрощенная)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Энкодер (сжатие)
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)

        # Декодер (разжатие)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.final = nn.Conv2d(32, 1, 1)  # 1 выходной канал (маска)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = self.pool(e1)
        e3 = torch.relu(self.enc2(e2))
        d1 = self.up(e3)
        d1 = torch.relu(self.dec1(d1))
        return torch.sigmoid(self.final(d1))


# 2. Датасет
class FrameDataset(Dataset):
    def __init__(self, root_dir):
        self.images_path = os.path.join(root_dir, 'clean_1008fps\cave_4')
        self.masks_path = os.path.join(root_dir, 'occlusions_1008fps\cave_4')
        self.filenames = os.listdir(self.images_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        # Загрузка и приведение к тензору
        image = F.to_tensor(Image.open(os.path.join(self.images_path, img_name)).convert("RGB"))
        mask = F.to_tensor(Image.open(os.path.join(self.masks_path, img_name[:3]+"_data"+img_name[3:])).convert("L"))
        return image, mask


# 3. Основной процесс обучения
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обучение на: {device}")

    # Загрузка данных
    dataset = FrameDataset('')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy для масок
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Цикл обучения
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Эпоха {epoch + 1}/{epochs}, Ошибка: {total_loss / len(loader):.4f}")

    # Сохранение модели
    torch.save(model.state_dict(), "model.pth")
    print("Обучение завершено. Модель сохранена в 'model.pth'")


if __name__ == "__main__":
    train()