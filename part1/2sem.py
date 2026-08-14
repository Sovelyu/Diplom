import cv2
import torch
import numpy as np
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.transforms.functional as F
import os
import glob


def safe_imread(path):
    """
    Читает файл как массив байтов (это обходит ограничение на кодировку путей в Windows).
    """
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Ошибка: не удалось декодировать файл {path}. Возможно, файл поврежден.")
    return img


def generate_mask_from_frames(frame1_path, frame2_path, model, device, transforms, magnitude_threshold=2.5):
    """
    Генерирует бинарную маску движущихся объектов на основе двух кадров.
    """
    # 1. Загружаем кадры через безопасную функцию
    img1 = safe_imread(frame1_path)
    img2 = safe_imread(frame2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"Не удалось загрузить кадры. Проверь пути!\n"
            f"Путь 1: '{frame1_path}'\n"
            f"Путь 2: '{frame2_path}'"
        )

    # 2. Подготовка данных для нейросети
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    tensor1 = F.to_tensor(img1_rgb).unsqueeze(0).to(device)
    tensor2 = F.to_tensor(img2_rgb).unsqueeze(0).to(device)
    batch1, batch2 = transforms(tensor1, tensor2)

    # 3. Вычисляем плотный оптический поток нейросетью RAFT
    with torch.no_grad():
        list_of_flows = model(batch1, batch2)
        predicted_flow = list_of_flows[-1][0]

        # Извлекаем компоненты смещения по осям X и Y
    flow_np = predicted_flow.cpu().numpy().transpose(1, 2, 0)
    fx, fy = flow_np[..., 0], flow_np[..., 1]

    # 4. Считаем длину вектора скорости (магнитуду) для каждого пикселя
    magnitude, _ = cv2.cartToPolar(fx, fy)

    # 5. Бинаризация: выделяем пиксели, чья скорость выше порога
    _, binary_mask = cv2.threshold(magnitude, magnitude_threshold, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    # 6. Фильтрация шумов с помощью морфологических операций
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return binary_mask, img2


def visualize_detected_objects(current_frame, mask, color_bgr=(0, 255, 0), alpha=0.6):
    """
    Накладывает на изображение полупрозрачную цветную маску в местах обнаружения объектов.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = current_frame.copy()
    cv2.drawContours(overlay, contours, -1, color_bgr, thickness=-1)

    beta = 1.0 - alpha
    output = cv2.addWeighted(current_frame, alpha, overlay, beta, 0)

    cv2.drawContours(output, contours, -1, color_bgr, thickness=2)

    return output


# =====================================================================
# Блок запуска программы
# =====================================================================
if __name__ == "__main__":
    # Настройки путей
    INPUT_FOLDER = "DICOM"
    OUTPUT_FOLDER = "results"

    # Создаем папку для результатов, если ее нет
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Ищем все PNG изображения в папке (можно добавить .jpg и др.)
    # Сортировка обязательна, чтобы кадры шли по порядку
    image_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.png")))

    if len(image_files) < 2:
        print(f"Ошибка: В папке '{INPUT_FOLDER}' должно быть как минимум 2 изображения!")
        exit()

    print(f"Найдено {len(image_files)} изображений для обработки.")

    try:
        print("Инициализация нейросети RAFT (это может занять несколько секунд)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).eval().to(device)
        transforms = weights.transforms()
        print(f"Модель загружена на {device.upper()}.\n")

        # Проходим по всем кадрам парами: (0,1), (1,2), (2,3) и т.д.
        for i in range(len(image_files) - 1):
            frame1_path = image_files[i]
            frame2_path = image_files[i + 1]

            # Получаем имя текущего файла для сохранения результатов
            base_name = os.path.basename(frame2_path)
            name_without_ext = os.path.splitext(base_name)[0]

            print(f"Обработка пары: {os.path.basename(frame1_path)} -> {base_name}...")

            # 1. Расчет оптического потока и генерация маски
            mask, current_frame = generate_mask_from_frames(
                frame1_path, frame2_path,
                model=model, device=device, transforms=transforms,
                magnitude_threshold=2.5
            )

            # 2. Создание визуализации
            result_image = visualize_detected_objects(current_frame, mask, color_bgr=(255, 0, 0), alpha=0.6)

            # 3. Сохранение результатов в отдельную папку
            mask_out_path = os.path.join(OUTPUT_FOLDER, f"mask_{name_without_ext}.png")
            vis_out_path = os.path.join(OUTPUT_FOLDER, f"vis_{name_without_ext}.png")

            cv2.imwrite(mask_out_path, mask)
            cv2.imwrite(vis_out_path, result_image)

        print(f"\nГотово! Все результаты сохранены в папку: '{OUTPUT_FOLDER}'")

    except Exception as e:
        print(f"\n[Ошибка работы скрипта]: {e}")