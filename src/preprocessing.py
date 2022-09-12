import cv2
import numpy as np


def get_retinal_image_diameter_as_horizontal_segment(raw_jpg: np.ndarray):
    '''
    returns the bounds in pixel indices of the image 

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset

    Returns
    -------
    top: int
        top bound
    bottom: int
        bottom bound
    left: int
        left bound
    right: int
        right bound
    '''
    hor = np.max(raw_jpg, axis=(0,2))
    horbounds = np.where(hor > 10)[0]

    ver = np.max(raw_jpg, axis=(1,2))
    verbounds = np.where(ver > 10)[0]
    
    return verbounds[0], verbounds[-1], horbounds[0], horbounds[-1]


def get_square_retinal_img(raw_jpg: np.ndarray, top: int, bottom: int, left: int, right: int):
    '''
    returns the square around the retina

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    top: int
        top bound
    bottom: int
        bottom bound
    left: int
        left bound
    right: int
        right bound

    Returns
    -------
    square: numpy.ndarray
    '''
    retina = raw_jpg[top:bottom+1, left:right+1]
    diff = retina.shape[1] - retina.shape[0]

    if diff == 0:
        return retina
    elif diff > 0:
        addtop = diff // 2
        addbottom = diff - (diff // 2)
        square = cv2.copyMakeBorder(
            retina,
            top=addtop,
            bottom=addbottom,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )
        return square
    else:
        diff = - diff
        addleft = diff // 2
        addright = diff - (diff // 2)
        square = cv2.copyMakeBorder(
            retina,
            top=0,
            bottom=0,
            left=addleft,
            right=addright,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )
        return square


def resize_square(img: np.ndarray, side: int):
    '''
    resizes sqaure image to fixed resolution

    Parameters
    ----------
    img: numpy.ndarray
        the original square image
    side: int
        number of pixels per side

    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    '''
    return cv2.resize(img, (side, side), interpolation= cv2.INTER_LINEAR)


def square_resize(raw_jpg: np.ndarray, side: int):
    '''
    cuts out square around retina and resizes image to fox resolution

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    side: int
        number of pixels per side
    
    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    '''
    top, bottom, left, right = get_retinal_image_diameter_as_horizontal_segment(raw_jpg)
    retinal_img_sq = get_square_retinal_img(raw_jpg, top, bottom, left, right)
    return resize_square(retinal_img_sq, side)


def make_square(raw_jpg: np.ndarray):
    '''
    cuts out square around retina and resizes image to fox resolution

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    side: int
        number of pixels per side
    
    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    '''
    top, bottom, left, right = get_retinal_image_diameter_as_horizontal_segment(raw_jpg)
    return get_square_retinal_img(raw_jpg, top, bottom, left, right)