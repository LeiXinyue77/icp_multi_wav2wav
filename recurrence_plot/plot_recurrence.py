import os
import numpy as np
import pylab as plt
from scipy.spatial.distance import pdist, squareform

def rec_plot(s, eps=0.01, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

from scipy.ndimage import zoom


def resize_image(Z, target_size=(128, 128)):
    # 使用scipy的zoom函数调整图像大小，确保递归图为目标大小
    return zoom(Z, (target_size[0] / Z.shape[0], target_size[1] / Z.shape[1]))


if __name__ == "__main__":
    root_dir = '../dl/data'
    folders = ['folder2', 'folder3', 'folder4', 'folder5']
    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        sig = np.load(file_path)
                        # 第 0 列信号是否有大于 20 的值
                        if np.any(sig[:, 0] > 30):
                            print(f"sig {file_path} has values greater than 30")
                            for i in range(sig.shape[1]):
                                rec = rec_plot(sig[:, i])
                                plt.imshow(rec)
                                # 调整递归图的大小
                                rec_resized = resize_image(rec, target_size=(128, 128))  # 调整为128x128
                                plt.imshow(rec_resized, cmap='inferno', origin='lower', aspect='auto')
                                plt.colorbar()  # 添加颜色条
                                plt.show()
                                print("sig %i - recurrence plot" % i)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
