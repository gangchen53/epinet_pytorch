from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_disparity_jet(data: np.ndarray, save_path: str):
    max_val = np.nanmax(data[data != np.inf])
    min_val = np.nanmin(data[data != np.inf])
    data_norm = (data - min_val) / (max_val - min_val)

    img = (data_norm * 255.0).astype(np.uint8)  # normalized to [0,1] and then multiplied by 255
    img = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    cv2.imwrite(save_path, img)


def plot_training_curve(data: List[List[float]], save_path: str):
    train_loss_list = data[0]
    test_loss_list = data[1]
    test_mae_list = data[2]
    test_mse_x100_list = data[3]
    test_bp_ratio_list = data[4]
    test_epoch_list = data[5]

    train_epoch = range(len(train_loss_list))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    axes[0, 0].plot(train_epoch, train_loss_list, color='r', label='Train Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')

    axes[0, 0].scatter(test_epoch_list, test_loss_list, color='b', label='Test Loss')
    axes[0, 0].legend()

    axes[0, 1].scatter(test_epoch_list, test_mae_list, color='b', label='Test MAE')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test MAE')

    axes[1, 0].scatter(test_epoch_list, test_mse_x100_list, color='b', label='Test MSE_x100')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test MES_x100')

    axes[1, 1].scatter(test_epoch_list, test_bp_ratio_list, color='b', label='Test Bad Pixel Ratio')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Bad Pixel Ratio')

    plt.savefig(save_path)
    plt.close()
