import cv2
import numpy as np
from scipy.ndimage.filters import convolve as filter2
import os


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def get_magnitude(u, v):
    scale = 3
    s = 0.0
    cnt = 0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dx = u[i, j] * scale
            dy = v[i, j] * scale
            s += np.sqrt(dx * dx + dy * dy)
            cnt += 1

    return s / cnt


def draw_quiver_on_frame(u, v, frame):
    """
    Рисует векторы оптического потока прямо на кадре
    """
    scale = 3
    magnitudeAvg = get_magnitude(u, v)

    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dx = int(u[i, j] * scale)
            dy = int(v[i, j] * scale)
            mag = np.sqrt(dx * dx + dy * dy)

            if mag > magnitudeAvg:
                cv2.arrowedLine(
                    out,
                    (j, i),
                    (j + dx, i + dy),
                    (0, 0, 255),
                    1,
                    tipLength=0.3
                )

    return out


def get_derivatives(img1, img2):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1, x_kernel) + filter2(img2, x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return fx, fy, ft


# =========================
# HORN–SCHUNCK
# =========================

def computeHS_from_frames(frame1, frame2, alpha, delta=1e-1):
    frame1 = frame1.astype(float)
    frame2 = frame2.astype(float)

    frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)
    frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)

    u = np.zeros(frame1.shape)
    v = np.zeros(frame1.shape)

    fx, fy, ft = get_derivatives(frame1, frame2)

    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                           [1 / 6, 0, 1 / 6],
                           [1 / 12, 1 / 6, 1 / 12]])

    for _ in range(300):
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)

        p = fx * u_avg + fy * v_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2

        u_new = u_avg - fx * (p / d)
        v_new = v_avg - fy * (p / d)

        if np.linalg.norm(u_new - u) < delta:
            break

        u, v = u_new, v_new

    return u, v


# =========================
# ОБРАБОТКА ВИДЕО
# =========================

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось прочитать видео")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        u, v = computeHS_from_frames(prev_gray, gray)
        vis = draw_quiver_on_frame(u, v, prev_gray)

        out.write(vis)

        prev_gray = gray

    cap.release()
    out.release()
    print("Готово! Видео сохранено:", output_video)


import cv2


def process_images(image1_path, image2_path, output_image_path):
    """
    Вычисляет оптический поток между двумя изображениями и сохраняет результат,
    накладывая визуализацию на второе изображение.
    """
    # 1. Загружаем оба изображения
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        raise RuntimeError("Не удалось прочитать изображения. Проверьте правильность путей.")

    # 2. Переводим в градации серого (алгоритм Horn-Schunck обычно работает с ч/б)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. Вычисляем оптический поток от первого кадра ко второму
    u, v = computeHS_from_frames(gray1, gray2)

    # 4. Накладываем векторы (quiver) на ВТОРОЕ изображение
    # Если функция draw_quiver_on_frame ожидает одноканальное изображение,
    # замените img2 на gray2
    vis = draw_quiver_on_frame(u, v, gray2)

    # 5. Сохраняем итоговое изображение
    cv2.imwrite(output_image_path, vis)
    print("Готово! Изображение сохранено:", output_image_path)


# =====================================================================
# Пример использования
# =====================================================================
"""if __name__ == "__main__":
    # Предполагается, что функции computeHS_from_frames и draw_quiver_on_frame
    # уже определены в твоем коде выше.

    FRAME_1 = "DICOM\\scan_001.png"
    FRAME_2 = "DICOM\\scan_002.png"
    OUTPUT = "result_flow.png"

    process_images(FRAME_1, FRAME_2, OUTPUT)

    #process_video("video.mp4", "output_floww.mp4")
"""
