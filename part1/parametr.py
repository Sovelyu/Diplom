import numpy as np
import glob
from hs import computeHS_from_frames
import os
import cv2



def read_flow_file(file_path):
    """ Читает файл .flo (формат Middlebury / Sintel) """
    with open(file_path, 'rb') as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if tag != 202021.25:
            raise ValueError(f"Неверный тег в файле: {file_path}")

        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=width * height * 2)

    flow = data.reshape((height, width, 2))

    # Маскируем неизвестные значения (unknown flow) согласно спецификации Sintel
    unknown_mask = (np.abs(flow[..., 0]) > 1e9) | (np.abs(flow[..., 1]) > 1e9)
    flow[unknown_mask] = np.nan
    return flow


def calculate_metrics(u_hs, v_hs, u_true, v_true):
    """
    Векторный расчет Mean EPE и Mean AE по всему изображению.
    Игнорирует пиксели со значениями NaN.
    """
    # Маска валидных пикселей (где истинный поток известен)
    valid_mask = ~np.isnan(u_true) & ~np.isnan(v_true)

    if not np.any(valid_mask):
        return 0.0, 0.0

    # Вырезаем только валидные пиксели в виде одномерных векторов
    u_h = u_hs[valid_mask]
    v_h = v_hs[valid_mask]
    u_t = u_true[valid_mask]
    v_t = v_true[valid_mask]

    # --- End-Point Error (EPE) ---

    epe_map = np.sqrt((u_h - u_t) ** 2 + (v_h - v_t) ** 2)
    mean_epe = np.mean(epe_map)

    # --- Angular Error (AE) ---
    eps = 1e-10  # Защита от деления на ноль
    numerator = 1 + u_h * u_t + v_h * v_t
    denominator = np.sqrt(1 + u_h ** 2 + v_h ** 2) * np.sqrt(1 + u_t ** 2 + v_t ** 2)

    # Ограничиваем аргумент arccos интервалом [-1.0, 1.0] для защиты от погрешностей float
    arg = np.clip(numerator / (denominator + eps), -1.0, 1.0)
    ae_map = np.arccos(arg)
    mean_ae = np.mean(ae_map)

    return mean_ae, mean_epe


if __name__ == "__main__":
    frames_dir = r"D:\nir\flow_dataset\training\final\temple_3"
    flow_dir = r"D:\nir\flow_dataset\training\flow\temple_3"

    # 1. Поиск и сортировка путей
    flo_paths = glob.glob(os.path.join(flow_dir, "*.flo"))
    flo_paths.sort()
    print(f"Найдено файлов потока (.flo): {len(flo_paths)}")

    extensions = ('*.jpg', '*.jpeg', '*.png')
    frame_paths = []
    for ext in extensions:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    frame_paths.sort()
    print(f"Найдено кадров: {len(frame_paths)}")

    # Инициализируем список и загружаем ВСЕ кадры в него
    frames = []
    for path in frame_paths:
        image_bgr = cv2.imread(path)
        if image_bgr is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

    num_pairs = min(len(frames) - 1, len(flo_paths))
    print(f"Будет обработано пар кадров: {num_pairs}\n")

    # 2. Цикл перебора значений alpha
    for alpha in range(1, 21):
        total_ae = 0.0
        total_epe = 0.0

        for i in range(num_pairs):
            # ВНИМАНИЕ: Проверьте возвращаемые значения!
            # Если ваша функция возвращает (u, v), оставьте u_hs, v_hs.
            # Если (v, u), поменяйте переменные местами: v_hs, u_hs = ...
            u_hs, v_hs = computeHS_from_frames(frames[i], frames[i + 1], alpha)

            # Читаем истинный поток из файла
            flow_gt = read_flow_file(flo_paths[i])
            u_true = flow_gt[..., 0]
            v_true = flow_gt[..., 1]

            # Вычисляем средние метрики для текущего кадра без циклов
            ae, epe = calculate_metrics(u_hs, v_hs, u_true, v_true)

            total_ae += ae
            total_epe += epe

        # Находим среднее значение ошибок по всей последовательности для текущего alpha
        mean_sequence_ae = total_ae / num_pairs
        mean_sequence_epe = total_epe / num_pairs

        print(f"===========================")
        print(f"Значение alpha: {alpha}")
        print(f"Mean Angular error: {mean_sequence_ae:.4f} рад")
        print(f"Mean End point error: {mean_sequence_epe:.4f} пикс\n")