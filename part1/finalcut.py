import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

#определение поля скоростей
def H_S(I1, I2, alpha, iter_times):
    # Приводим изображения к типу float
    I1 = I1.astype(np.float32)
    I2 = I2.astype(np.float32)

    I1_p = np.pad(I1, ((1, 1), (1, 1)), mode='edge')
    I2_p = np.pad(I2, ((1, 1), (1, 1)), mode='edge')

    Ix = np.zeros_like(I1, dtype=np.float32)
    Iy = np.zeros_like(I1, dtype=np.float32)
    It = np.zeros_like(I1, dtype=np.float32)

    #вычисляем производные
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            # смещаем индексы на +1 из-за паддинга
            ip = i + 1
            jp = j + 1

            Ix[i, j] = 0.25*(I1_p[ip, jp+1] - I1_p[ip, jp] + I1_p[ip+1, jp+1] - I1_p[ip+1, jp]+
                             I2_p[ip, jp+1]-I2_p[ip, jp]+I2_p[ip+1, jp+1]-I2_p[ip+1, jp])
            Iy[i, j] = 0.25*(I1_p[ip+1, jp] - I1_p[ip, jp] + I1_p[ip+1, jp+1] - I1_p[ip, jp+1]-
                             I2_p[ip, jp+1]-I2_p[ip, jp]+I2_p[ip+1, jp+1] + I2_p[ip+1, jp])
            It[i, j] = 0.25*((-1) * I1_p[ip+1, jp] - I1_p[ip, jp] - I1_p[ip+1, jp+1] - I1_p[ip, jp+1]+
                             I2_p[ip, jp+1] + I2_p[ip, jp] + I2_p[ip+1, jp+1] + I2_p[ip+1, jp])

    u = np.zeros_like(I1, dtype=np.float32)
    v = np.zeros_like(I1, dtype=np.float32)
    a = (alpha ** 2 + Ix ** 2 + Iy ** 2)

    for i in range(iter_times):

        # Паддинг для свертки
        u_pad = np.pad(u, ((1, 1), (1, 1)), mode='edge')
        v_pad = np.pad(v, ((1, 1), (1, 1)), mode='edge')

        # Вычисляем средние с помощью свертки (вручную для простоты)
        u_avg = np.zeros_like(u_pad)
        v_avg = np.zeros_like(v_pad)

        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                u_avg[i, j] = ((u_pad[i-1, j] + u_pad[i, j+1] + u_pad[i+1, j] + u_pad[i, j-1])/6 +
                               (u_pad[i-1, j-1] + u_pad[i-1, j+1] + u_pad[i+1, j+1] + u_pad[i+1, j-1]) /12)
                v_avg[i, j] = ((v_pad[i-1, j] + v_pad[i, j+1] + v_pad[i+1, j] + v_pad[i, j-1])/6 +
                               (v_pad[i-1, j-1] + v_pad[i-1, j+1] + v_pad[i+1, j+1] + v_pad[i+1, j-1]) /12)
                u_pad = u_avg
                v_pad = v_avg

    Ix = np.pad(Ix, ((1, 1), (1, 1)), mode='edge')
    Iy = np.pad(Iy, ((1, 1), (1, 1)), mode='edge')
    It = np.pad(It, ((1, 1), (1, 1)), mode='edge')
    a = np.pad(a, ((1, 1), (1, 1)), mode='edge')

    u = u_avg - Ix * (Ix*u_avg + Iy*v_avg + It) / a
    v = v_avg - Iy * (Ix * u_avg + Iy * v_avg + It) / a

    return u, v

def to_grayscale(img):
    if img.ndim == 3:
        if img.shape[2] == 4:   # RGBA
            img = img[..., :3]
        return np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return img  # уже ч/б



img11 = imread("test images/table1.jpg").copy()
img22 = imread("test images/table2.jpg").copy()

img1 = to_grayscale(img11)
img2 = to_grayscale(img22)
#img1 = np.zeros((h, w), dtype=np.float32)
#img2 = np.zeros((h, w), dtype=np.float32)


alpha = 5.0
num_iter = 20
u, v = H_S(img1, img2, alpha=alpha, iter_times=num_iter)

if __name__ == "__main__":
    # 3) Наложим поле векторов (quiver) поверх первого кадра для наглядности
    plt.plot()
    plt.imshow(img22, cmap='gray')
    plt.axis('off')

    step = 20

    # Размеры исходного изображения
    h, w = img1.shape[:2]

    # Сетка координат
    Y, X = np.mgrid[0:h:step, 0:w:step]

    # Подвыборка потока под ту же сетку
    U = u[0:Y.shape[0]*step:step, 0:X.shape[1]*step:step]
    V = v[0:Y.shape[0]*step:step, 0:X.shape[1]*step:step]

    # На случай, если всё равно остаётся 1 строка/столбец лишняя
    U = U[:Y.shape[0], :Y.shape[1]]
    V = V[:Y.shape[0], :Y.shape[1]]

    # Теперь размеры гарантированно совпадают
    plt.quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=1)

    plt.axis('off')

    plt.tight_layout()
    plt.show()
