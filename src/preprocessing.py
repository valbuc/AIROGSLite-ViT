import cv2
import numpy as np
import math


def get_retinal_image_diameter_as_horizontal_segment(raw_jpg: np.ndarray, threshold: int):
    """
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
    threshold: int
        threshold value used for the cutoff
    """
    hor = np.max(raw_jpg, axis=(0, 2))
    horbounds = np.where(hor > threshold)[0]

    ver = np.max(raw_jpg, axis=(1, 2))
    verbounds = np.where(ver > threshold)[0]
    
    return verbounds[0], verbounds[-1], horbounds[0], horbounds[-1]


def get_square_retinal_img(raw_jpg: np.ndarray, top: int, bottom: int, left: int, right: int):
    """
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
        the new square image
    cutting: tuple
        how much was cut from the top, bottom, left, right
    padding: tuple
        how much was padded at top, bottom, left, right after the cutting has been performed
    """
    retina = raw_jpg[top:bottom+1, left:right+1]
    remtop = top
    rembottom = raw_jpg.shape[0] - (bottom+1)
    remleft = left
    remright = raw_jpg.shape[1] - (right+1)

    diff = retina.shape[1] - retina.shape[0]
    addtop = addbottom = addleft = addright = 0

    if diff == 0:
        return retina, (remtop, rembottom, remleft, remright), (addtop, addbottom, addleft, addright)
    elif diff > 0:
        addtop += diff // 2
        addbottom += diff - (diff // 2)
    else:
        diff = - diff
        addleft += diff // 2
        addright += diff - (diff // 2)

    square = cv2.copyMakeBorder(
        retina,
        top=addtop,
        bottom=addbottom,
        left=addleft,
        right=addright,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return square, (remtop, rembottom, remleft, remright), (addtop, addbottom, addleft, addright)


def resize_square(img: np.ndarray, side: int):
    """
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
    """
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_LINEAR)


def square_resize(raw_jpg: np.ndarray, side: int, threshold: int):
    """
    cuts out square around retina and resizes image to fixed resolution

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    side: int
        number of pixels per side
    threshold: int
        threshold value used for determining how much can be cut from the image edges

    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    scaling_factor: float
        multiply new image size by this factor to get original resolution
    cutting: numpy.ndarray
        resized square image
    padding: numpy.ndarray
        resized square image
    """
    top, bottom, left, right = get_retinal_image_diameter_as_horizontal_segment(raw_jpg, threshold)
    retinal_img_sq, cutting, padding = get_square_retinal_img(raw_jpg, top, bottom, left, right)
    assert retinal_img_sq.shape[0] == retinal_img_sq.shape[1]
    new_img = resize_square(retinal_img_sq, side)
    assert side == new_img.shape[0]
    assert side == new_img.shape[1]
    scaling_factor = retinal_img_sq.shape[0] / side

    # consistency test / sanity check: checking that the original resolution can be reconstructed
    added_x = -(padding[0] + padding[1]) + (cutting[0] + cutting[1])
    added_y = -(padding[2] + padding[3]) + (cutting[2] + cutting[3])
    rec_res_x, rec_res_y = round((side * scaling_factor) + added_x), round((side * scaling_factor) + added_y)
    assert rec_res_x == raw_jpg.shape[0]
    assert rec_res_y == raw_jpg.shape[1]

    return new_img, scaling_factor, cutting, padding


def make_square(raw_jpg: np.ndarray, threshold: int):
    """
    cuts out square around retina and resizes image to fix resolution

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    side: int
        number of pixels per side
    threshold: int
        threshold value used for determining how much can be cut from the image edges

    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    """
    top, bottom, left, right = get_retinal_image_diameter_as_horizontal_segment(raw_jpg, threshold)
    return get_square_retinal_img(raw_jpg, top, bottom, left, right)


def crop_od(original_img, odc_x, odc_y, sidelength):
    '''
    will in reality return square of size sidelength+1 
    '''
    radius = int(sidelength/2)
    
    vertical_diff = 0
    top = odc_y - radius
    if top < 0:
        vertical_diff = - top
        top = 0
    bottom = odc_y + radius + vertical_diff
    if bottom > original_img.shape[0]:
        top -= bottom - original_img.shape[0]
        bottom = original_img.shape[0] - 1

    horizontal_diff = 0
    left = odc_x - radius
    if left < 0:
        horizontal_diff = - left
        left = 0
    right = odc_x + radius + horizontal_diff
    if right > original_img.shape[1]:
        left -= right - original_img.shape[1]
        right = original_img.shape[1] - 1
    
    return original_img[top:bottom+1, left:right+1]


def crop_od_fill_if_needed(original_img, odc_x, odc_y, sidelength, fill_value=0):
    half_side_1 = int(sidelength / 2)
    half_side_2 = sidelength - half_side_1

    top = odc_y - half_side_1
    clipped_top = max(0, top)
    add_top = -min(top, 0)

    bottom = odc_y + half_side_2
    clipped_bottom = min(original_img.shape[0], bottom)
    add_bottom = -min(0, original_img.shape[0]-bottom)

    left = odc_x - half_side_1
    clipped_left = max(0, left)
    add_left = -min(0, left)

    right = odc_x + half_side_2
    clipped_right = min(original_img.shape[1], right)
    add_right = -min(0, original_img.shape[1]-right)

    #print(add_top, add_bottom, add_left, add_right)
    new_img = np.full(shape=(sidelength, sidelength, original_img.shape[2]), fill_value=fill_value, dtype=original_img.dtype)
    new_img[add_top:sidelength-add_bottom, add_left:sidelength-add_right, :] = \
        original_img[clipped_top:clipped_bottom, clipped_left:clipped_right, :]

    return new_img
