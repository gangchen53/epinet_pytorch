import os
import random
import time
import argparse
from pathlib import Path
from typing import List

import numpy as np
import imageio
import torch
import torch.optim as optim
import torch.utils.data as data

from utils.logger import mylogger
from utils.plot import save_disparity_jet, plot_training_curve
from utils.early_stopping import EarlyStopping
from utils.datasets import HCIDataset
from utils.loss_function import LossFunction
from models.epi_net import EPINet


class Trainer:
    def __init__(self, opt):
        dataset_root_dir = opt.dataset_root_dir
        results_save_dir = opt.results_save_dir
        num_angular_views = opt.num_angular_views
        resume = opt.resume
        checkpoint = opt.checkpoint
        device = opt.device
        start_epoch = opt.start_epoch
        early_stopping_patience = opt.early_stopping_patience
        train_patch_size = opt.train_patch_size
        learning_rate = opt.learning_rate
        batch_size = opt.batch_size
        task = opt.task
        loss_type = opt.loss_type
        steps = opt.steps
        gamma = opt.gamma
        test_frequency = opt.test_frequency
        plot_frequency = opt.plot_frequency
        seed = opt.seed
        test_margin = opt.test_margin
        train_warm_start = opt.train_warm_start

        assert plot_frequency >= test_frequency, 'plot_frequency must be greater than test_frequency!'
        if resume:
            assert checkpoint, 'If resume is False, checkpoint must not be None!'

        # setting random seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        dataset_root_dir = Path(dataset_root_dir)

        # Early stopping mechanism for stop training, because training is an infinite loop.
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, is_minimize=True)

        angular_views = list(range(num_angular_views))

        """
            We use totally 16 LF images,
            Since some images(4, 6, 15) have a reflection region (kitchen, museum, vinyl),
            We decrease frequency of occurrence for them. 
            Details in epinet paper.
        """
        train_images_dir = [
            'additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
            'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',
            'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table',
            'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl',
        ]

        test_images_dir = [
            'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
            'training/boxes',        'training/cotton', 'training/dino',       'training/sideboard',
        ]

        # load invalid regions from training data (ex. reflective region)
        bool_mask_img4 = imageio.v2.imread(
            str(dataset_root_dir / 'additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
        )
        bool_mask_img6 = imageio.v2.imread(
            str(dataset_root_dir / 'additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
        )
        bool_mask_img15 = imageio.v2.imread(
            str(dataset_root_dir / 'additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')
        )
        bool_mask_img4 = 1.0 * bool_mask_img4[:, :, 3] > 0
        bool_mask_img6 = 1.0 * bool_mask_img6[:, :, 3] > 0
        bool_mask_img15 = 1.0 * bool_mask_img15[:, :, 3] > 0

        train_dataset = HCIDataset(
            dataset_root_dir=str(dataset_root_dir),
            train_patch_size=train_patch_size,
            images_dir=train_images_dir,
            angular_views=angular_views,
            bool_mask_img4=bool_mask_img4,
            bool_mask_img6=bool_mask_img6,
            bool_mask_img15=bool_mask_img15,
            is_train=True,
            dataset_length=16 * 4 * 10,
            # dataset_length=None,
        )
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            # num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        print(f'------> Finish loading the training set')

        test_dataset = HCIDataset(
            dataset_root_dir=str(dataset_root_dir),
            images_dir=test_images_dir,
            angular_views=angular_views,
            is_train=False,
            margin=test_margin,
        )
        self.test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            # num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        print(f'------> Finish loading the testing set')

        self.device = device
        self.model = EPINet(
            in_channels=len(angular_views),
            multistream_layer_channels=70,
            multistream_layer_depth=3,
            merge_layer_depth=7,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer_info = repr(self.optimizer)

        if loss_type is None:
            loss_type = 'mae'
        self.loss_function = LossFunction(loss_type=loss_type).to(self.device)
        loss_function_info = repr(self.loss_function)

        self.current_epoch = start_epoch

        # plot training curve
        self.train_loss_list = []
        self.test_loss_list = []
        self.test_mae_list = []
        self.test_mse_x100_list = []
        self.test_bp_ratio_list = []

        self.best_mse_x100 = float('inf')

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=steps, gamma=gamma)
        lr_scheduler_info = f'lr_scheduler: MultiStepLR, milestones={steps}, gamma={gamma}'

        self.test_epoch_list = []
        self.train_total_time = 0
        if resume:
            self.results_save_dir = Path(checkpoint).parents[0]

            training_logger_save_path = self.results_save_dir / 'training_logger.txt'
            testing_logger_save_path = self.results_save_dir / 'testing_logger.txt'
            self.training_logger = mylogger(name='training_logger', logger_save_path=str(training_logger_save_path))
            self.testing_logger = mylogger(name='testing_logger', logger_save_path=str(testing_logger_save_path))

            self.training_logger.info(f'resume, loading checkpoint: {checkpoint}')

            # last model checkpoint is broken, we load second last model checkpoint.
            try:
                state_dict = torch.load(checkpoint)
                self.model.load_state_dict(state_dict['model'])
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

                self.best_mse_x100 = state_dict['best_mse_x100']
                self.current_epoch = state_dict['epoch'] + 1
                self.train_total_time = state_dict['train_total_time']

                self.early_stopping.counter = state_dict['early_stopping_counter']
                self.early_stopping.best_score = self.best_mse_x100

                self.train_loss_list = state_dict['train_loss_list']
                self.test_loss_list = state_dict['test_loss_list']
                self.test_mae_list = state_dict['test_mae_list']
                self.test_mse_x100_list = state_dict['test_mse_x100_list']
                self.test_bp_ratio_list = state_dict['test_bp_ratio_list']
                self.test_epoch_list = state_dict['test_epoch_list']
            except KeyError:
                print('------> Because loading current model has key error, we will load second_last_model!')

                second_last_state_dict_path = self.results_save_dir / 'second_last_model.tar'
                state_dict = torch.load(second_last_state_dict_path)

                self.model.load_state_dict(state_dict['model'])
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

                self.best_mse_x100 = state_dict['best_mse_x100']
                self.current_epoch = state_dict['epoch'] + 1
                self.train_total_time = state_dict['train_total_time']

                self.early_stopping.counter = state_dict['early_stopping_counter']
                self.early_stopping.best_score = self.best_mse_x100

                self.train_loss_list = state_dict['train_loss_list']
                self.test_loss_list = state_dict['test_loss_list']
                self.test_mae_list = state_dict['test_mae_list']
                self.test_mse_x100_list = state_dict['test_mse_x100_list']
                self.test_bp_ratio_list = state_dict['test_bp_ratio_list']
                self.test_epoch_list = state_dict['test_epoch_list']
        else:
            results_save_dir = Path(results_save_dir) / time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            results_save_dir.mkdir(exist_ok=True, parents=True)
            self.results_save_dir = results_save_dir

            training_logger_save_path = self.results_save_dir / 'training_logger.txt'
            testing_logger_save_path = self.results_save_dir / 'testing_logger.txt'
            self.training_logger = mylogger(name='training_logger', logger_save_path=str(training_logger_save_path))
            self.testing_logger = mylogger(name='testing_logger', logger_save_path=str(testing_logger_save_path))

        self.visualize_save_dir = self.results_save_dir / 'visualize_output'
        self.visualize_save_dir.mkdir(exist_ok=True, parents=True)

        self.best_model_save_path = self.results_save_dir / 'best_model.tar'
        self.second_last_model_save_path = self.results_save_dir / 'second_last_model.tar'
        self.current_model_save_path = self.results_save_dir / 'current_model.tar'

        if self.current_epoch == 1:
            self.training_logger.info(f'Batch_Size: {batch_size}')
            self.training_logger.info(f'Angular_Views: {angular_views}')
            self.training_logger.info(f'Early_Stopping_Patience: {early_stopping_patience}')
            self.training_logger.info(f'Train_Patch_Size: {train_patch_size}')
            self.training_logger.info(f'Optimizer: {optimizer_info}')
            self.training_logger.info(f'Loss_Function: {loss_function_info}')
            self.training_logger.info(f'LR_Scheduler: {lr_scheduler_info}')
        self.test_frequency = test_frequency
        self.plot_frequency = plot_frequency
        self.train_warm_start = train_warm_start
        self.learning_rate = learning_rate

        self.task = task
        print(f'------> Complete the model initialization setup')

    def evaluation(self):
        self.model.eval()

        test_batch_loss = 0
        mean_absolute_error_batch = 0
        mean_squared_error_x100_batch = 0
        bad_pixel_ratio_batch = 0
        num_test_loader = len(self.test_loader)
        for batch_id, (images_90d, images_0d, images_45d, images_m45d, labels) in enumerate(self.test_loader):
            images_90d = images_90d.to(self.device)
            images_0d = images_0d.to(self.device)
            images_45d = images_45d.to(self.device)
            images_m45d = images_m45d.to(self.device)  # (1, C, 512, 512)
            labels = labels.to(self.device)  # (1, 512, 512)

            inputs = (images_90d, images_0d, images_45d, images_m45d)

            with torch.no_grad():
                outputs = self.model(inputs)  # (1, 1, 512 - 22, 512 - 22), (1, 1, 490, 490)
            outputs = outputs.squeeze()  # (1, 1, 490, 490) -> (490, 490)

            disparity = labels[0]

            loss = self.loss_function(predictions=outputs, labels=disparity)
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

            if self.current_epoch % 100 == 0:
                save_dir = self.visualize_save_dir / str(self.current_epoch)
                save_dir.mkdir(exist_ok=True, parents=True)
                save_path = save_dir / f'{batch_id}_error.jpg'

                save_disparity_jet(data=diff, save_path=str(save_path))

        test_batch_average_loss = test_batch_loss / num_test_loader
        mean_absolute_error_average = mean_absolute_error_batch / num_test_loader
        mean_squared_error_x100_average = mean_squared_error_x100_batch / num_test_loader
        bad_pixel_ratio_average = bad_pixel_ratio_batch / num_test_loader

        self.test_loss_list.append(test_batch_average_loss)
        self.test_mae_list.append(mean_absolute_error_average)
        self.test_mse_x100_list.append(mean_squared_error_x100_average)
        self.test_bp_ratio_list.append(bad_pixel_ratio_average)
        self.test_epoch_list.append(self.current_epoch)

        if self.best_mse_x100 > mean_squared_error_x100_average:
            self.best_mse_x100 = mean_squared_error_x100_average
            torch.save({
                'model': self.model.state_dict(),
                'best_mse_x100': self.best_mse_x100,
                'epoch': self.current_epoch,
            }, str(self.best_model_save_path))

        self.testing_logger.info(
            f'{"Test":<5s} | Epoch: {self.current_epoch}, '
            f'Loss: {test_batch_average_loss:.5f}, '
            f'MAE: {mean_absolute_error_average:.5f}, '
            f'MSE_x100: {mean_squared_error_x100_average:.5f}, '
            f'Bad_Pixel_Ratio: {bad_pixel_ratio_average:.5f}, '
            f'Best_MSE_x100: {self.best_mse_x100:.5f}, '
            f'Early_Stopping_Counter: {self.early_stopping.counter}'
        )

        self.early_stopping(current_score=mean_squared_error_x100_average)

    def train(self):
        self.model.train()

        start_time = time.time()

        train_batch_loss = 0
        mean_absolute_error_batch = 0
        mean_squared_error_x100_batch = 0
        bad_pixel_ratio_batch = 0
        num_train_loader = len(self.train_loader)
        for batch_id, (images_90d, images_0d, images_45d, images_m45d, labels) in enumerate(self.train_loader):
            images_90d = images_90d.to(self.device)
            images_0d = images_0d.to(self.device)
            images_45d = images_45d.to(self.device)
            images_m45d = images_m45d.to(self.device)  # (B, C, H, W)
            labels = labels.to(self.device)  # (B, H, W)

            inputs = (images_90d, images_0d, images_45d, images_m45d)

            if self.train_warm_start:
                if self.current_epoch <= 10:
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.learning_rate * float(self.current_epoch) / 10.0
            if self.current_epoch >= 5000:
                for g in self.optimizer.param_groups:
                    g['lr'] = 1e-6
            if self.current_epoch >= 6000:
                for g in self.optimizer.param_groups:
                    g['lr'] = 1e-7

            self.optimizer.zero_grad()

            outputs = self.model(inputs)  # (B, 1, H, W)
            outputs = outputs.squeeze(1)  # (B, H, W)

            outputs_scale = 1.0 * outputs
            labels_scale = 1.0 * labels

            loss = self.loss_function(predictions=outputs_scale, labels=labels_scale)

            loss.backward()
            self.optimizer.step()

            train_batch_loss += loss.item()

            # calculate training batch evaluation metric
            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            diff = np.abs(outputs - labels)
            bp = (diff >= 0.07)
            mean_absolute_error = np.average(diff)
            mean_squared_error_x100 = 100 * np.average(np.square(diff))
            bad_pixel_ratio = 100 * np.average(bp)

            mean_absolute_error_batch += mean_absolute_error
            mean_squared_error_x100_batch += mean_squared_error_x100
            bad_pixel_ratio_batch += bad_pixel_ratio

        self.lr_scheduler.step()

        train_batch_average_loss = train_batch_loss / num_train_loader
        self.train_loss_list.append(train_batch_average_loss)

        mean_absolute_error_average = mean_absolute_error_batch / num_train_loader
        mean_squared_error_x100_average = mean_squared_error_x100_batch / num_train_loader
        bad_pixel_ratio_average = bad_pixel_ratio_batch / num_train_loader

        end_time = time.time()
        training_time = (end_time - start_time) / 3600

        self.train_total_time += training_time

        self.training_logger.info(
            f'{"Train":<5s} | Epoch: {self.current_epoch}, '
            f'Loss: {train_batch_average_loss:.5f}, '
            f'MAE: {mean_absolute_error_average:.5f}, '
            f'MSE_x100: {mean_squared_error_x100_average:.5f}, '
            f'Bad_Pixel_Ratio: {bad_pixel_ratio_average:.5f}, '
            f'Learning_Rate: {self.optimizer.param_groups[0]["lr"]}, '
            f'Training_Time: {training_time:.5f} Hours'
        )

        if self.current_model_save_path.exists():
            if self.second_last_model_save_path.exists():
                os.remove(self.second_last_model_save_path)
            os.rename(src=self.current_model_save_path, dst=self.second_last_model_save_path)

        torch.save({
            'model': self.model.state_dict(),
            'best_mse_x100': self.best_mse_x100,
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'train_total_time': self.train_total_time,
            'early_stopping_counter': self.early_stopping.counter,
            'train_loss_list': self.train_loss_list,
            'test_loss_list': self.test_loss_list,
            'test_mae_list': self.test_mae_list,
            'test_mse_x100_list': self.test_mse_x100_list,
            'test_bp_ratio_list': self.test_bp_ratio_list,
            'test_epoch_list': self.test_epoch_list,
        }, str(self.current_model_save_path))

    def run(self):
        print(f'------> Model starts running')
        if self.task == 'train':
            while True:
                self.train()

                if self.current_epoch < 5000:
                    if self.current_epoch % self.test_frequency == 0:
                        self.evaluation()
                else:
                    self.evaluation()

                if self.current_epoch % self.plot_frequency == 0:
                    save_path = self.results_save_dir / 'training_curve.jpg'
                    plot_training_curve(
                       data=[
                            self.train_loss_list,
                            self.test_loss_list,
                            self.test_mae_list,
                            self.test_mse_x100_list,
                            self.test_bp_ratio_list,
                            self.test_epoch_list,
                       ],
                       save_path=str(save_path),
                    )

                if self.early_stopping.early_stop:
                    self.training_logger.info(f'training_total_time: {self.train_total_time:.3f}')
                    print(f'EarlyStopping, best score over {self.early_stopping.counter} rounds without a drop')
                    break

                self.current_epoch += 1
        else:
            self.evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root_dir', type=str, default='../datasets/HCI_Benchmark/full_data',
                        help='dataset_root_dir')
    parser.add_argument('--results_save_dir', type=str, default='results', help='results_save_dir')
    parser.add_argument('--loss_type', type=str, default='mae', help='results_save_dir')
    parser.add_argument('--num_angular_views', type=int, default=9, help='angular_views')
    parser.add_argument('--resume', action='store_true', help='whether to resume')
    parser.add_argument('--checkpoint', type=str, default=None, help='model weight path')
    parser.add_argument('--device', type=str, default='cuda', help='device, cpu or cuda')
    parser.add_argument('--start_epoch', type=int, default=1, help='start_epoch')
    parser.add_argument('--test_margin', type=int, default=11, help='start_epoch')
    parser.add_argument('--seed', type=int, default=53, help='random seed')
    parser.add_argument('--test_frequency', type=int, default=10, help='start_epoch')
    parser.add_argument('--plot_frequency', type=int, default=10, help='plot_frequency')
    parser.add_argument('--early_stopping_patience', type=int, default=2000, help='start_epoch')
    parser.add_argument('--train_patch_size', type=int, default=96, help='train_patch_size (>= 23)')
    parser.add_argument('--task', type=str, default='train', help='train or evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='start_epoch')
    parser.add_argument('--train_warm_start', action='store_true', help='train_warm_start')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--steps', default=[1000, 2000, 3000, 4000], help='learning rate step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='learning rate decay parameter: Gamma')

    current_opt = parser.parse_args()

    trainer = Trainer(current_opt)
    trainer.run()
