import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as data

from utils.datasets import HCIDataset
from utils.plot import save_disparity_jet
from utils.loss_function import LossFunction
from models.epi_net import EPINet


def test(save_dir: str,
         checkpoint: str,
         angular_views: Optional[List[int]] = None,
         test_margin: int = 11,
         device: str = 'cuda',
         loss_type: str = 'mae',
         ):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    dataset_root_dir = Path('../datasets/HCI_Benchmark/full_data')
    test_images_dir = [
        'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
        'training/boxes',        'training/cotton', 'training/dino',       'training/sideboard',
    ]

    if angular_views is None:
        angular_views = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    test_dataset = HCIDataset(
        dataset_root_dir=str(dataset_root_dir),
        images_dir=test_images_dir,
        angular_views=angular_views,
        is_train=False,
        margin=test_margin,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        # num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    print(f'------> Finish loading the testing set')

    model = EPINet(
        in_channels=len(angular_views),
        multistream_layer_channels=70,
        multistream_layer_depth=3,
        merge_layer_depth=7,
    ).to(device)
    model.eval()

    if loss_type is None:
        loss_type = 'mae'
    loss_function = LossFunction(loss_type=loss_type).to(device)
    loss_function_info = repr(loss_function)

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])

    start_time = end_time = 0
    test_batch_loss = 0
    mean_absolute_error_batch = 0
    mean_squared_error_x100_batch = 0
    bad_pixel_ratio_batch = 0
    num_test_loader = len(test_loader)
    for batch_id, (images_90d, images_0d, images_45d, images_m45d, labels) in enumerate(test_loader):
        images_90d = images_90d.to(device)
        images_0d = images_0d.to(device)
        images_45d = images_45d.to(device)
        images_m45d = images_m45d.to(device)  # (1, C, 512, 512)
        labels = labels.to(device)  # (1, 512, 512)

        inputs = (images_90d, images_0d, images_45d, images_m45d)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)  # (1, 1, 512 - 22, 512 - 22), (1, 1, 490, 490)
        end_time = time.time()

        outputs = outputs.squeeze()  # (1, 1, 490, 490) -> (490, 490)

        disparity = labels[0]

        loss = loss_function(predictions=outputs, labels=disparity)
        test_batch_loss += loss.item()

        disparity = disparity.cpu().numpy()
        outputs = outputs.cpu().numpy()
        diff = np.abs(outputs - disparity)

        bp = (diff >= 0.07)
        mean_absolute_error = np.average(diff)
        mean_squared_error_x100 = 100 * np.average(np.square(diff))
        bad_pixel_ratio = 100 * np.average(bp)

        mean_absolute_error_batch += mean_absolute_error
        mean_squared_error_x100_batch += mean_squared_error_x100
        bad_pixel_ratio_batch += bad_pixel_ratio

        error_save_path = save_dir / f'{batch_id}_error.jpg'
        save_disparity_jet(data=diff, save_path=str(error_save_path))

        prediction_save_path = save_dir / f'{batch_id}_prediction.jpg'
        save_disparity_jet(data=outputs, save_path=str(prediction_save_path))

        gt_save_path = save_dir / f'{batch_id}_gt.jpg'
        save_disparity_jet(data=disparity, save_path=str(gt_save_path))

    inference_time = end_time - start_time
    test_batch_average_loss = test_batch_loss / num_test_loader
    mean_absolute_error_average = mean_absolute_error_batch / num_test_loader
    mean_squared_error_x100_average = mean_squared_error_x100_batch / num_test_loader
    bad_pixel_ratio_average = bad_pixel_ratio_batch / num_test_loader

    save_txt = save_dir / 'final_test.txt'
    with save_txt.open('w') as f:
        lines = [
            f'Loss Function: {loss_function_info}',
            f'Loss: {test_batch_average_loss}',
            f'MAE: {mean_absolute_error_average}',
            f'MSE_x100: {mean_squared_error_x100_average}',
            f'BP Ratio: {bad_pixel_ratio_average}',
            f'Inference Time: {inference_time} second'
        ]
        for line in lines:
            f.write(line + '\n')
    print(f'------> Final Test! Results save at {save_dir}')


if __name__ == '__main__':
    test(
        save_dir='results/2022_08_27_18_55_45/final_test',
        checkpoint='results/2022_08_27_18_55_45/best_model.tar',
        angular_views=[0, 1, 2, 3, 4, 5, 6],
        test_margin=11,
        device='cuda',
        loss_type='mae',
    )
