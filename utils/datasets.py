import random
from pathlib import Path
from typing import List, Optional

import torch
import imageio
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data


def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    with open(fpath, 'rb') as f:
        # header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception(
                'Could not parse dimensions: "%s". Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception(
                'Could not parse max value / endianess information: "%s". Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))  # (H, W)
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data


class HCIDataset(data.Dataset):
    def __init__(self,
                 dataset_root_dir: str,
                 images_dir: List[str],
                 angular_views: List[int],
                 bool_mask_img4: Optional[np.ndarray] = None,
                 bool_mask_img6: Optional[np.ndarray] = None,
                 bool_mask_img15: Optional[np.ndarray] = None,
                 train_patch_size: Optional[int] = None,
                 margin: Optional[int] = None,
                 is_train: bool = True,
                 dataset_length: Optional[int] = None,
                 ):
        super(HCIDataset, self).__init__()
        if is_train:
            assert train_patch_size, 'if is_train is True, train_patch_size must not be None!'
        else:
            assert margin, 'if is_train is False, margin must not be None!'

        self.train_patch_size = train_patch_size
        self.margin = margin

        self.bool_mask_img4 = bool_mask_img4
        self.bool_mask_img6 = bool_mask_img6
        self.bool_mask_img15 = bool_mask_img15

        self.is_train = is_train

        self.dataset_root_dir = dataset_root_dir
        self.images_dir = images_dir

        self.angular_views = angular_views

        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of
        # shape(C x H x W) in the range[0.0, 1.0] if the PIL Image belongs to one of the
        # modes(L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8.
        self.transforms = transforms.ToTensor()

        self.base_shift_dict = {
            9: 0,
            7: 1,
            5: 2,
            3: 3
        }
        self.base_shift = self.base_shift_dict[len(self.angular_views)]

        # cache dataset into dict
        images_dir_filter = list(set(images_dir))
        self.dataset_dict = {}
        for img_dir in images_dir_filter:
            images, label = self.load_data(
                dataset_root_dir=self.dataset_root_dir,
                image_dir=img_dir,
            )
            self.dataset_dict[img_dir] = {
                'images': images,
                'label': label,
            }
        self.dataset_length = dataset_length

    def __getitem__(self, index):
        if self.dataset_length is not None:
            index = index % len(self.images_dir)

        class_name = self.images_dir[index].split('/')[1]
        images = self.dataset_dict[self.images_dir[index]]['images']
        label = self.dataset_dict[self.images_dir[index]]['label']

        if self.is_train:
            images_90d, images_0d, images_45d, images_m45d, label = self.generate_patch_size_data(
                images=images,
                label=label,
                class_name=class_name,
                image_size=self.train_patch_size,
                label_size=self.train_patch_size - 22,
            )

            images_90d, images_0d, images_45d, images_m45d, label = \
                self.data_augmentation_for_train(
                    images_90d=images_90d,
                    images_0d=images_0d,
                    images_45d=images_45d,
                    images_m45d=images_m45d,
                    label=label,
                )
        else:
            images_90d, images_0d, images_45d, images_m45d, label = self.generate_full_size_data(
                images=images,
                label=label,
                margin=self.margin,
            )

        # images: (H, W, C) -> (C, H, W)
        # labels: (H, W)
        images_90d = self.transforms(np.ascontiguousarray(images_90d))
        images_0d = self.transforms(np.ascontiguousarray(images_0d))
        images_45d = self.transforms(np.ascontiguousarray(images_45d))
        images_m45d = self.transforms(np.ascontiguousarray(images_m45d))
        label = torch.from_numpy(np.ascontiguousarray(label))

        return images_90d, images_0d, images_45d, images_m45d, label

    def __len__(self):
        if self.dataset_length is not None:
            return self.dataset_length
        else:
            return len(self.images_dir)

    def load_data(self, dataset_root_dir: str, image_dir: str):
        """
        :param dataset_root_dir:
        :param image_dir:
        :return:
            images: np.ndarray, (H, W, num_angular_views, num_angular_views, 3)
            labels: np.ndarray, (H, W)
        """
        images = np.zeros((512, 512, 9, 9, 3), dtype=np.uint8)

        for i in range(81):
            img_path = Path(dataset_root_dir) / image_dir / f'input_Cam{i:03d}.png'
            if img_path.exists():
                img = imageio.v2.imread(str(img_path))  # (H, W, C)
                img = img.astype(np.float32)
            else:
                raise FileNotFoundError(f'{str(img_path)} does not exist!')

            # note: i - 9 * (i // 9) = i % 9, i // 9 represent row, i - 9 * (i // 9) represent column
            # images_batch shape is (B, H, W, row, col, C)
            images[:, :, i // 9, i % 9, :] = img

        lbl_path = Path(dataset_root_dir) / image_dir / 'gt_disp_lowres.pfm'
        if lbl_path.exists():
            label = read_pfm(str(lbl_path))  # lbl shape is (H, W), default (512, 512)
        else:
            raise FileNotFoundError(f'{str(lbl_path)} does not exist!')
        return images, label

    def generate_patch_size_data(self,
                                 images: np.ndarray,
                                 label: np.ndarray,
                                 class_name: str,
                                 image_size: int,
                                 label_size: int,
                                 ):
        """ The order in which the images are stored is as follows:
             45d       90d     m45d
                0       0       0
                   1    1     1
                     2  2   2
                      3 3 3
            0d  0 1 2 3 4 5 6 7 8
                      5 5 5
                    6   6   6
                  7     7     7
                8       8       8

        :param images:
        :param label:
        :param class_name:
        :param image_size:
        :param label_size:
        :return:
            images_90d: (H, W, C)
            images_0d: (H, W, C)
            images_45d: (H, W, C)
            images_m45d: (H, W, C)
            disparity: (H, W)
        """
        num_angular_views = len(self.angular_views)
        images_90d = np.zeros((image_size, image_size, num_angular_views), dtype=np.float32)
        images_0d = np.zeros((image_size, image_size, num_angular_views), dtype=np.float32)
        images_45d = np.zeros((image_size, image_size, num_angular_views), dtype=np.float32)
        images_m45d = np.zeros((image_size, image_size, num_angular_views), dtype=np.float32)

        disparity = np.zeros((label_size, label_size), dtype=np.float32)

        crop_half = int(0.5 * (image_size - label_size))

        sum_diff = 0.0
        valid = True
        while sum_diff < 0.01 * image_size * image_size or not valid:
            valid = True

            rand_3color = 0.05 + np.random.rand(3)
            rand_3color = rand_3color / np.sum(rand_3color)
            r, g, b = rand_3color

            if len(self.angular_views) == 9:
                x_shift = 0
                y_shift = 0
            elif len(self.angular_views) == 7:
                x_shift = np.random.randint(0, 3) - 1  # (0, 1, 2) - 1 = (-1, 0, 1)
                y_shift = np.random.randint(0, 3) - 1  # (0, 1, 2) - 1 = (-1, 0, 1)
            elif len(self.angular_views) == 5:
                x_shift = np.random.randint(0, 5) - 2  # (0, 1, 2, 3, 4) - 2 = (-2, -1, 0, 1, 2)
                y_shift = np.random.randint(0, 5) - 2  # (0, 1, 2, 3, 4) - 2 = (-2, -1, 0, 1, 2)
            elif len(self.angular_views) == 3:
                x_shift = np.random.randint(0, 7) - 3  # (-3, -2, -1, 0, 1, 2, 3)
                y_shift = np.random.randint(0, 7) - 3  # (-3, -2, -1, 0, 1, 2, 3)

            k = np.random.randint(17)
            if k < 8:
                scale = 1
            elif k < 14:
                scale = 2
            else:
                scale = 3

            # return a random integer from [low, high), low = 0, high = 512 - scale * image_size
            x_start = np.random.randint(0, 512 - scale * image_size)
            y_start = np.random.randint(0, 512 - scale * image_size)
            if class_name in ['kitchen', 'museum', 'vinyl']:
                if class_name == 'kitchen':
                    mask = self.bool_mask_img4
                if class_name == 'museum':
                    mask = self.bool_mask_img6
                else:  # if class_name == 'vinyl':
                    mask = self.bool_mask_img15
                # If the mask covers the image area or the label area, then let valid be False.
                flag_1 = np.sum(mask[
                    x_start + scale * crop_half: x_start + scale * crop_half + scale * label_size: scale,
                    y_start + scale * crop_half: y_start + scale * crop_half + scale * label_size: scale]
                ) > 0
                flag_2 = np.sum(mask[
                    x_start: x_start + scale * image_size: scale, y_start: y_start + scale * image_size: scale]
                ) > 0
                if flag_1 and flag_2:
                    valid = False

            if valid:
                image_center = (1 / 255.0) * (
                    r * images[x_start: x_start + scale * image_size: scale,
                               y_start: y_start + scale * image_size: scale,
                               4 + x_shift, 4 + y_shift, 0].astype(np.float32) +
                    g * images[x_start: x_start + scale * image_size: scale,
                               y_start: y_start + scale * image_size: scale,
                               4 + x_shift, 4 + y_shift, 1].astype(np.float32) +
                    b * images[x_start: x_start + scale * image_size: scale,
                               y_start: y_start + scale * image_size: scale,
                               4 + x_shift, 4 + y_shift, 2].astype(np.float32)
                )
                # I don't know why we should calculate sum_diff ?
                sum_diff = np.sum(np.abs(image_center - image_center[int(0.5 * image_size), int(0.5 * image_size)]))

                for i in self.angular_views:  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    images_90d[:, :, i] = \
                        r * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + x_shift + i, 4 + y_shift, 0].astype(np.float32) + \
                        g * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + x_shift + i, 4 + y_shift, 1].astype(np.float32) + \
                        b * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + x_shift + i, 4 + y_shift, 2].astype(np.float32)

                    images_0d[:, :, i] = \
                        r * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   4 + x_shift, self.base_shift + y_shift + i, 0].astype(np.float32) + \
                        g * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   4 + x_shift, self.base_shift + y_shift + i, 1].astype(np.float32) + \
                        b * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   4 + x_shift, self.base_shift + y_shift + i, 2].astype(np.float32)

                    images_45d[:, :, i] = \
                        r * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + y_shift + i, self.base_shift + y_shift + i, 0].astype(np.float32) + \
                        g * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + y_shift + i, self.base_shift + y_shift + i, 1].astype(np.float32) + \
                        b * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + y_shift + i, self.base_shift + y_shift + i, 2].astype(np.float32)

                    images_m45d[:, :, i] = \
                        r * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + x_shift + i, self.angular_views[-1] + y_shift - i, 0].astype(np.float32) + \
                        g * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + x_shift + i, self.angular_views[-1] + y_shift - i, 1].astype(np.float32) + \
                        b * images[x_start: x_start + scale * image_size: scale,
                                   y_start: y_start + scale * image_size: scale,
                                   self.base_shift + x_shift + i, self.angular_views[-1] + y_shift - i, 2].astype(np.float32)

                disparity = (1.0 / scale) * label[
                                x_start + scale * crop_half: x_start + scale * crop_half + scale * label_size: scale,
                                y_start + scale * crop_half: y_start + scale * crop_half + scale * label_size: scale]
        return images_90d, images_0d, images_45d, images_m45d, disparity

    @staticmethod
    def data_augmentation_for_train(images_90d: np.ndarray,
                                    images_0d: np.ndarray,
                                    images_45d: np.ndarray,
                                    images_m45d: np.ndarray,
                                    label: np.ndarray,
                                    ):
        """
        :param images_90d:
        :param images_0d:
        :param images_45d:
        :param images_m45d:
        :param label:
        :return:
            images_90d: (H, W, C)
            images_0d: (H, W, C)
            images_45d: (H, W, C)
            images_m45d: (H, W, C)
            disparity: (H, W)
        """
        images_90d /= 255.0
        images_0d /= 255.0
        images_45d /= 255.0
        images_m45d /= 255.0

        gray_rand = 0.4 * np.random.rand() + 0.8
        images_90d = pow(images_90d, gray_rand)
        images_0d = pow(images_0d, gray_rand)
        images_45d = pow(images_45d, gray_rand)
        images_m45d = pow(images_m45d, gray_rand)

        k = np.random.randint(0, 5)
        # rotate 90, 180, 270 and flip along the diagonal
        if k in [1, 2, 3]:
            images_90d_rotate = np.rot90(images_90d, k, (0, 1)).copy()  # 沿 H 维度向 W 维度旋转, 即沿坐标轴逆时针旋转
            images_0d_rotate = np.rot90(images_0d, k, (0, 1)).copy()
            images_45d_rotate = np.rot90(images_45d, k, (0, 1)).copy()
            images_m45d_rotate = np.rot90(images_m45d, k, (0, 1)).copy()

            if k == 1:
                images_90d = images_0d_rotate[:, :, ::-1]  # correct
                images_0d = images_90d_rotate  # correct
                images_45d = images_m45d_rotate  # correct
                images_m45d = images_45d_rotate[:, :, ::-1]  # correct
            elif k == 2:
                images_90d = images_90d_rotate[:, :, ::-1]  # correct
                images_0d = images_0d_rotate[:, :, ::-1]  # correct
                images_45d = images_45d_rotate[:, :, ::-1]  # correct
                images_m45d = images_m45d_rotate[:, :, ::-1]  # correct
            else:
                images_90d = images_0d_rotate  # correct
                images_0d = images_90d_rotate[:, :, ::-1]  # correct
                images_45d = images_m45d_rotate[:, :, ::-1]  # correct
                images_m45d = images_45d_rotate  # correct

            label = np.rot90(label, k, (0, 1)).copy()
        elif k == 4:
            # np.transpose(images_batch_90d, (1, 0, 2)) will convert (H, W, C) to (W, H, C)
            # this operator corresponds to the flip in the paper, which is specifically flipped along the diagonal.
            images_90d_transpose = np.transpose(images_90d, (1, 0, 2)).copy()
            images_0d_transpose = np.transpose(images_0d, (1, 0, 2)).copy()
            images_45d_transpose = np.transpose(images_45d, (1, 0, 2)).copy()
            images_m45d_transpose = np.transpose(images_m45d, (1, 0, 2)).copy()

            images_90d = images_0d_transpose
            images_0d = images_90d_transpose
            images_45d = images_45d_transpose
            images_m45d = images_m45d_transpose[:, :, ::-1].copy()

            label = np.transpose(label, (1, 0)).copy()

        images_90d = (images_90d * 255.0).astype(np.float32)
        images_0d = (images_0d * 255.0).astype(np.float32)
        images_45d = (images_45d * 255.0).astype(np.float32)
        images_m45d = (images_m45d * 255.0).astype(np.float32)
        return images_90d, images_0d, images_45d, images_m45d, label

    def generate_full_size_data(self, images: np.ndarray, label: np.ndarray, margin: int = 11):
        """
        :param images: (512, 512, 9, 9, 3)
        :param label: (512, 512)
        :param margin: int, default 11.
        :return:
            images_90d: (H, W, C)
            images_0d: (H, W, C)
            images_45d: (H, W, C)
            images_m45d: (H, W, C)
            disparity: (H, W)
        """
        num_angular_views = len(self.angular_views)
        images_90d = np.zeros((512, 512, num_angular_views), dtype=np.float32)
        images_0d = np.zeros((512, 512, num_angular_views), dtype=np.float32)
        images_45d = np.zeros((512, 512, num_angular_views), dtype=np.float32)
        images_m45d = np.zeros((512, 512, num_angular_views), dtype=np.float32)

        r = 0.299
        g = 0.587
        b = 0.114
        for i in self.angular_views:
            images_90d[:, :, i] = \
                r * images[0:512, 0:512, self.base_shift + i, 4, 0].astype(np.float32) + \
                g * images[0:512, 0:512, self.base_shift + i, 4, 1].astype(np.float32) + \
                b * images[0:512, 0:512, self.base_shift + i, 4, 2].astype(np.float32)

            images_0d[:, :, i] = \
                r * images[0:512, 0:512, 4, self.base_shift + i, 0].astype(np.float32) + \
                g * images[0:512, 0:512, 4, self.base_shift + i, 1].astype(np.float32) + \
                b * images[0:512, 0:512, 4, self.base_shift + i, 2].astype(np.float32)

            images_45d[:, :, i] = \
                r * images[0:512, 0:512, self.base_shift + i, self.base_shift + i, 0].astype(np.float32) + \
                g * images[0:512, 0:512, self.base_shift + i, self.base_shift + i, 1].astype(np.float32) + \
                b * images[0:512, 0:512, self.base_shift + i, self.base_shift + i, 2].astype(np.float32)

            images_m45d[:, :, i] = \
                r * images[0:512, 0:512, self.base_shift + i, self.angular_views[-1] - i, 0].astype(np.float32) + \
                g * images[0:512, 0:512, self.base_shift + i, self.angular_views[-1] - i, 1].astype(np.float32) + \
                b * images[0:512, 0:512, self.base_shift + i, self.angular_views[-1] - i, 2].astype(np.float32)

        disparity = label[margin:-margin, margin:-margin]
        return images_90d, images_0d, images_45d, images_m45d, disparity
