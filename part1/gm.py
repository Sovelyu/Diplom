import cv2
import numpy as np
from hs import computeHS_from_frames


def generate_mask_from_frames(frame1_path, frame2_path, magnitude_threshold=2.5):
    """
    Генерирует бинарную маску движущихся объектов на основе двух кадров
    с помощью расчета плотного оптического потока Фарнебека.

    :param frame1_path: Путь к первому (прошлому) кадру
    :param frame2_path: Путь ко второму (текущему) кадру
    :param magnitude_threshold: Порог скорости для отсечения неподвижного фона
    :return: Бинарная маска (np.uint8) и оригинальный текущий кадр
    """
    # 1. Загружаем кадры
    img1 = cv2.imread(frame1_path)
    img2 = cv2.imread(frame2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"Не удалось загрузить кадры. Проверь пути!\n"
            f"Путь 1: '{frame1_path}'\n"
            f"Путь 2: '{frame2_path}'"
        )

    # 2. Переводим кадры в градации серого
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. Вычисляем плотный оптический поток
    fx, fy = computeHS_from_frames(gray1, gray2, alpha=15, delta=1e-1)

    # Извлекаем компоненты смещения по осям X и Y
    #fx, fy = flow[..., 0], flow[..., 1]
    magnitude, _ = cv2.cartToPolar(fx, fy)

    # 4. Считаем длину вектора скорости (магнитуду) для каждого пикселя
    #magnitude, _ = cv2.cartToPolar(fx, fy)

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

    :param current_frame: Оригинальный текущий кадр
    :param mask: Бинарная маска объектов (0 и 255)
    :param color_bgr: Цвет заливки в формате BGR (по умолчанию зеленый)
    :param alpha: Прозрачность оригинального кадра (от 0 до 1)
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


# =====================================================================
# Блок запуска программы
# =====================================================================
if __name__ == "__main__":
    FRAME_1 = "DICOM\scan_001.png"
    FRAME_2 = "DICOM\scan_002.png"

    try:
        print("Шаг 1: Расчет оптического потока и генерация маски...")
        # Если объекты движутся медленно или камера шумит, отрегулируй порог (magnitude_threshold)
        mask, current_frame = generate_mask_from_frames(FRAME_1, FRAME_2, magnitude_threshold=2.5)

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