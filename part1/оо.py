import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt


class MovingObjectDataset(Dataset):

    def __init__(self, num_samples=1000, height=128, width=128):
        self.num_samples = num_samples
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Создаем пустые кадры
        img1 = np.zeros((self.height, self.width), dtype=np.float32)
        img2 = np.zeros((self.height, self.width), dtype=np.float32)

        # Случайные параметры объекта (круг или квадрат)
        shape_type = np.random.choice(['circle', 'rect'])
        size = np.random.randint(15, 30)

        x1 = np.random.randint(30, self.width - 30)
        y1 = np.random.randint(30, self.height - 30)

        dx = np.random.randint(-12, 13)
        dy = np.random.randint(-12, 13)
        if dx == 0 and dy == 0: dx, dy = 5, 5

        x2, y2 = x1 + dx, y1 + dy

        if shape_type == 'circle':
            cv2.circle(img1, (x1, y1), size // 2, 0.8, -1)
            cv2.circle(img2, (x2, y2), size // 2, 0.8, -1)
        else:
            cv2.rectangle(img1, (x1 - size // 2, y1 - size // 2), (x1 + size // 2, y1 + size // 2), 0.8, -1)
            cv2.rectangle(img2, (x2 - size // 2, y2 - size // 2), (x2 + size // 2, y2 + size // 2), 0.8, -1)

        noise = np.random.rand(self.height, self.width) * 0.2
        img1 += noise
        img2 += noise

        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        target_mask = np.zeros((self.height, self.width), dtype=np.float32)
        if shape_type == 'circle':
            cv2.circle(target_mask, (x2, y2), size // 2, 1.0, -1)
        else:
            cv2.rectangle(target_mask, (x2 - size // 2, y2 - size // 2), (x2 + size // 2, y2 + size // 2), 1.0, -1)

        # Объединяем два кадра в один тензор с двумя каналами (размерность: 2, H, W)
        inputs = np.stack([img1, img2], axis=0)

        # Добавляем размерность канала для маски (размерность: 1, H, W)
        target_mask = np.expand_dims(target_mask, axis=0)

        return torch.from_numpy(inputs), torch.from_numpy(target_mask)



class MotionSegmentationNet(nn.Module):
    """Архитектура U-Net. Она принимает 2 кадра, неявно вычисляет
    оптический поток в слоях энкодера и собирает точную маску объекта в декодере."""

    def __init__(self):
        super(MotionSegmentationNet, self).__init__()

        # Энкодер (извлечение признаков движения)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)  # Выход: 64x64

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # Выход: 32x32

        # Узкое горлышко (Bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Декодер (восстановление формы объекта по вычисленному движению)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Выход: 64x64
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Выход: 128x128
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Финальный слой: 1 канал (вероятность пикселя принадлежать движущемуся объекту)
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Энкодер
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Горлышко
        b = self.bottleneck(p2)

        # Декодер
        d2 = self.up2(b)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        out = self.dec1(d1)
        return out



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Обучение будет проходить на: {device}")

# Инициализируем данные и модель
train_dataset = MovingObjectDataset(num_samples=3000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = MotionSegmentationNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 7
model.train()

print("Запуск самостоятельного обучения модели...")
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Эпоха [{epoch + 1}/{epochs}] -> Ошибка (Loss): {epoch_loss:.4f}")


model.eval()
test_dataset = MovingObjectDataset(num_samples=1)
test_inputs, test_target = test_dataset[0]

with torch.no_grad():
    # Прогоняем тестовый пример через обученную сеть
    pred = model(test_inputs.unsqueeze(0).to(device)).cpu().squeeze(0).squeeze(0).numpy()

# Переводим тензоры в обычные картинки для отображения
frame1 = test_inputs[0].numpy()
frame2 = test_inputs[1].numpy()
true_mask = test_target.squeeze(0).numpy()

# Визуализация результатов
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(frame1, cmap='gray')
axes[0].set_title("Кадр 1 (Прошлое)")
axes[0].axis('off')

axes[1].imshow(frame2, cmap='gray')
axes[1].set_title("Кадр 2 (Настоящее)")
axes[1].axis('off')

axes[2].imshow(true_mask, cmap='gray')
axes[2].set_title("Истинный контур объекта")
axes[2].axis('off')

# Порог детекции: пиксели с вероятностью > 0.5 считаем объектом
binary_pred = (pred > 0.5).astype(np.float32)
axes[3].imshow(binary_pred, cmap='jet')
axes[3].set_title("Определено нейросетью")
axes[3].axis('off')

plt.show()