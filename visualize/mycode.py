import os
from pathlib import Path

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import file_io

if __name__ == '__main__':
    cb_shrink = 0.7
    fig, axes = plt.subplots(6, 4)

    counter = 0
    mean = 0.0
    std = 0.0
    root_dir = Path('../../datasets/HCI_Benchmark/full_data/')
    dirs_name = ['training', 'additional', 'stratified']
    for dir_name in dirs_name:
        dir_path = root_dir / dir_name
        sub_dirs_name = os.listdir(str(dir_path))
        for sub_dir_name in sub_dirs_name:
            sub_dir_path = dir_path / sub_dir_name
            if sub_dir_path.is_dir():
                disp_map = file_io.read_disparity(sub_dir_path, highres=False)

                num_pixels = disp_map.size
                disp_map_mean = np.mean(disp_map)
                disp_map_std = np.std(disp_map)

                mean += disp_map_mean
                std += disp_map_std

                row = counter // 4
                col = counter % 4
                axes[row, col].imshow(disp_map, cmap=cm.viridis, interpolation="none")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                axes[row, col].set_title(f'mean: {disp_map_mean: .2f}, std: {disp_map_std:.2f}', fontsize=10)

                counter += 1

                # cc = axes[row, col].imshow(disp_map, cmap=cm.viridis, interpolation="none")
                # plt.colorbar(cc, shrink=cb_shrink)
    mean /= 24
    std /= 24
    print(f'mean: {mean:.2f}, std: {std:.2f}')
    plt.show()
