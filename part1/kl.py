import cv2
import torch
import numpy as np
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights
import torchvision.transforms.functional as F
import os

def generate_mask_from_frames(frame1_path, frame2_path, model, device, transforms, magnitude_threshold=2.5):
    # 1. Загружаем кадры
    img1 = cv2.imread(frame1_path)
    img2 = cv2.imread(frame2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"Не удалось загрузить кадры. Проверь пути!\n"
            f"Путь 1: '{frame1_path}'\n"
            f"Путь 2: '{frame2_path}'"
        )

    # 2. Подготовка данных для нейросети (вместо перевода в градации серого)
    # OpenCV использует BGR, а RAFT ожидает RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Переводим в тензоры PyTorch и переносим на видеокарту (если есть)
    tensor1 = F.to_tensor(img1_rgb).unsqueeze(0).to(device)
    tensor2 = F.to_tensor(img2_rgb).unsqueeze(0).to(device)
    batch1, batch2 = transforms(tensor1, tensor2)

    # 3. Вычисляем плотный оптический поток нейросетью RAFT
    with torch.no_grad():
        list_of_flows = model(batch1, batch2)
        predicted_flow = list_of_flows[-1][0] # Берем самую точную итерацию

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
    # Удаляем мелкие одиночные точки на фоне (шум камеры, шелест листьев)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # Затягиваем дыры и пустоты внутри силуэтов самих объектов
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return binary_mask, img2


def visualize_detected_objects(current_frame, mask, color_bgr=(0, 255, 0), alpha=0.6):
    """
    Накладывает на изображение полупрозрачную цветную маску в местах обнаружения объектов.
    """
    # Находим контуры объектов на маске (RETR_EXTERNAL находит только внешние границы)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем копию кадра, на которой сделаем сплошную цветную заливку объектов
    overlay = current_frame.copy()
    cv2.drawContours(overlay, contours, -1, color_bgr, thickness=-1)

    # Смешиваем оригинальный кадр и залитый слой для эффекта полупрозрачности
    beta = 1.0 - alpha  # Прозрачность цветной маски
    output = cv2.addWeighted(current_frame, alpha, overlay, beta, 0)

    # Дополнительно обводим контуры объектов тонкой яркой линией для четкости границ
    cv2.drawContours(output, contours, -1, color_bgr, thickness=2)

    return output


def safe_imread(path):
    # Читаем файл как массив байтов (это обходит ограничение на кодировку)
    stream = np.fromfile(path, dtype=np.uint8)
    # Декодируем массив в изображение
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Ошибка: не удалось декодировать файл {path}. Возможно, файл поврежден.")
    return img


# =====================================================================
# Блок запуска программы
# =====================================================================
if __name__ == "__main__":
    FRAME_1 = "DICOM\scan_001.png"
    FRAME_2 = "DICOM\scan_002.png"

    try:
        print("Инициализация нейросети RAFT (это может занять несколько секунд)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #weights = Raft_Small_Weights.DEFAULT
        #model = raft_small(weights=weights, progress=False).eval().to(device)
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).eval().to(device)
        transforms = weights.transforms()
        print(f"Модель загружена на {device.upper()}.")

        print("Шаг 1: Расчет оптического потока и генерация маски...")
        # Передаем модель и трансформации внутрь функции
        mask, current_frame = generate_mask_from_frames(
            FRAME_1, FRAME_2,
            model=model, device=device, transforms=transforms,
            magnitude_threshold=2.5
        )

        print("Шаг 2: Создание цветной полупрозрачной визуализации...")
        # Выберем красивый синий цвет для выделения (255, 0, 0) или зеленый (0, 255, 0)
        result_image = visualize_detected_objects(current_frame, mask, color_bgr=(255, 0, 0), alpha=0.6)

        # Сохраняем полученные файлы на диск
        cv2.imwrite("extracted_binary_mask.png", mask)
        cv2.imwrite("final_colored_detection.png", result_image)
        print("Готово! Результаты сохранены в папку проекта.")

        # Выводим окна с результатами на экран
        cv2.imshow("1. Original Current Frame", current_frame)
        cv2.imshow("2. Generated Binary Mask", mask)
        cv2.imshow("3. Detected Objects (Colored)", result_image)

        print("\nНажми любую клавишу в окне изображения, чтобы закрыть программу...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n[Ошибка работы скрипта]: {e}")
        print("Пожалуйста, проверьте правильность путей к файлам FRAME_1 и FRAME_2.")