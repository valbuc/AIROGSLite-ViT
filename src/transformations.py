import numpy as np
from PIL import Image
from torchvision.transforms.functional import equalize, center_crop, get_dimensions
import random
import cv2
import torch

class EqualizeIgnoreBlack(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, threshold, kernel_size):
        self.threshold = threshold
        self.kernel_size = kernel_size

    def __call__(self, PilImage):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        img = np.array(PilImage)

        # get mask
        grey = np.max(img, axis = 2)
        mask = grey > self.threshold
        mask = mask.astype(float)

        # remove noise from mask
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((self.kernel_size, self.kernel_size)))
        mask_inverted = mask_closed == 0

        # make mask 3d
        mask_inverted = mask_inverted.reshape((mask_inverted.shape[0], mask_inverted.shape[1], 1))
        mask_inv_3d = np.concatenate((mask_inverted, mask_inverted, mask_inverted), axis=2)

        # replace black with uniform noise
        count = np.count_nonzero(mask_inv_3d)
        uniform_noise = np.random.randint(0, 255, count)
        img[mask_inv_3d] = uniform_noise

        # equalize
        equal = equalize(Image.fromarray(img))

        # substitute noise
        equal = np.array(equal)
        equal[mask_inv_3d] = 123
        return Image.fromarray(equal)


class Equalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self):
        pass

    def __call__(self, array):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return equalize(array)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        x = (tensor - self.mean[:, None, None]) / self.std[:, None, None]
        return x


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class EqualizeTransform(object):
    def __init__(self, args):
        self.equalize = None
        if args.aug_hist_equalize.lower() == 'ignoreblack':
            self.equalize = EqualizeIgnoreBlack(9, 69)
        elif args.aug_hist_equalize.lower() == 'yes':
            self.equalize = Equalize()

    def __call__(self, sample):
        if self.equalize:
            return self.equalize(sample)
        else:
            return sample


class CenterCrop(object):
    def __init__(self, factor: float = 0.5, jitter: float = 0.0):
        self.factor = factor
        self.jitter = jitter

    def __call__(self, img):
        factor = self.factor
        if self.jitter > 0:
            factor += random.uniform(-self.jitter, self.jitter)
        _, _, side_pxl = get_dimensions(img)
        side_pxl = int(round(side_pxl * factor, 0))
        return center_crop(img, side_pxl)


class Translate(object):
    def __init__(self, prob: float = 0.5, ratio: float = 0.25):
        self.prob = prob
        self.ratio = ratio

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): square tensor image of size (C, H, W) to be translated

        Returns:
            Tensor: translated Tensor image.
        """
        translated = tensor
        rand = random.random()
        if rand < self.prob:
            # vertical translation    
            translate_vertical = random.uniform(-self.ratio, self.ratio)
            if translate_vertical > 0:
                down = int(round(translated.shape[1]*translate_vertical, 0))
                add = torch.zeros(3, down, translated.shape[2])
                translated = translated[:, :-down,:]
                translated = torch.cat((add, translated), dim=1)
            elif translate_vertical < 0:
                up = -int(round(translated.shape[1]*translate_vertical, 0))
                add = torch.zeros(3, up, translated.shape[2])
                translated = translated[:, up:,:]
                translated = torch.cat((translated, add), dim=1)

            # horizontal translation
            translate_horizontal = random.uniform(-self.ratio, self.ratio)
            if translate_horizontal > 0:
                right = int(round(translated.shape[2]*translate_horizontal, 0))
                add = torch.zeros(3, translated.shape[1], right)
                translated = translated[:, :, :-right]
                translated = torch.cat((add, translated), dim=2)
            elif translate_horizontal < 0:
                left = -int(round(translated.shape[2]*translate_horizontal, 0))
                add = torch.zeros(3, translated.shape[1], left)
                translated = translated[:, :, left:]
                translated = torch.cat((translated, add), dim=2)
        return translated