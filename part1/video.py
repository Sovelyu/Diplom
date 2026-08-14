import cv2
import numpy as np
import eje

def optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)


    u, v = eje.horn_schunck(prev_gray, curr_gray, alpha=1.0, num_iter=200)

    flow = np.stack((u, v), axis=-1)
    return flow


input_video = "vvideo.mp4"
output_video = "output.avi"

# Загружаем видео
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Ошибка: не удалось открыть видео.")
    exit()

# Получаем параметры видео
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Считываем первый кадр
ret, prev_frame = cap.read()
if not ret:
    print("Ошибка: не удалось прочитать первый кадр.")
    cap.release()
    exit()

frames = []  # сюда будут складываться обработанные кадры

print("Обработка кадров")

# Основной цикл обработки всех кадров
while True:
    ret, curr_frame = cap.read()
    if not ret:
        print("Обработка завершена.")
        break

    flow = optical_flow(prev_frame, curr_frame)
    display_frame = curr_frame.copy()

    # Визуализация стрелками
    h, w = flow.shape[:2]
    step = 10  # шаг сетки стрелок
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            if abs(fx) > 0.5 or abs(fy) > 0.5:
                cv2.arrowedLine(display_frame, (x, y),
                                (x + int(fx), y + int(fy)),
                                (0, 255, 0), 1, tipLength=0.3)

    # Добавляем текст
    cv2.putText(display_frame, "Optical Flow (Horn-Schunck)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    frames.append(display_frame)
    prev_frame = curr_frame


cap.release()

output_video = "output.mp4"  # или любой другой путь

# используем кодек для mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, size)

for frame in frames:
    out.write(frame)

out.release()