import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO


def segment_objects_with_flow(frames_dir):
    model = YOLO("yolov8n-seg.pt")

    extensions = ('*.jpg', '*.jpeg', '*.png')
    frame_paths = []
    for ext in extensions:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    frame_paths.sort()

    if len(frame_paths) < 2:
        print(f"Ошибка: Нужно минимум 2 кадра в папке '{frames_dir}'")
        return

    prev_frame = cv2.imread(frame_paths[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Flow", cv2.WINDOW_NORMAL)

    for i in range(1, len(frame_paths)):
        current_frame = cv2.imread(frame_paths[i])
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Создаем копию кадра для отрисовки прозрачных масок
        overlay = current_frame.copy()

        # 2. Расчет оптического потока
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Пороговая маска движения
        motion_mask = np.zeros_like(magnitude, dtype=np.uint8)
        motion_mask[magnitude > 2.0] = 255

        # 3. Запуск сегментации нейросетью
        results = model(current_frame, verbose=False)[0]

        # Проверяем, нашла ли сеть хоть какие-то маски объектов
        if results.masks is not None:
            # Получаем маски в формате numpy, смасштабированные под размер картинки
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes

            for mask, box in zip(masks, boxes):
                # Приводим маску объекта к размеру исходного кадра и бинаризуем
                object_mask = cv2.resize(mask, (current_frame.shape[1], current_frame.shape[0]))
                object_mask = (object_mask > 0.5).astype(np.uint8) * 255

                # Ищем пересечение формы объекта с картой движения
                # (смотрим только на движение внутри контура объекта)
                motion_inside_object = cv2.bitwise_and(motion_mask, object_mask)

                # Считаем процент движущихся пикселей внутри точной формы объекта
                total_object_pixels = np.sum(object_mask == 255)
                if total_object_pixels == 0:
                    continue

                motion_ratio = np.sum(motion_inside_object == 255) / total_object_pixels

                # Определяем статус и цвет в зависимости от движения
                if motion_ratio > 0.10:  # Более 10% объекта движется
                    color = (0, 255, 0)  # Зеленый для движущихся
                    status = "MOVING"
                else:
                    color = (0, 0, 255)  # Красный для статичных
                    status = "STILL"

                # 4. Отрисовка точной формы (заливка силуэта)
                overlay[object_mask == 255] = color

                # Дополнительно находим контур формы, чтобы красиво его обвести
                contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(current_frame, contours, -1, color, 2)


        # Смешиваем полупрозрачную заливку масок с оригинальным кадром, где прорисованы контуры
        alpha = 0.4  # Прозрачность заливки формы
        result_frame = cv2.addWeighted(overlay, alpha, current_frame, 1 - alpha, 0)

        # 5. Вывод на экран
        cv2.imshow("Flow", result_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev_gray = current_gray

    cv2.destroyAllWindows()


if __name__ == "__main__":
    segment_objects_with_flow(r"D:\nir\ColonCancerCT-2025 A Dataset of Abdominal CT Scans\Split_Data\Split_Data\train\Non_Cancer")
