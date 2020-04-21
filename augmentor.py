import random
import math
from math import floor
import numpy as np
import tensorflow as tf


random.seed(1)
np.random.seed(1)


def random_flip_left_right(image):
    """Flip image left or right randomly."""
    image = tf.image.random_flip_left_right(image)
    return image


def random_brightness(image, min_factor, max_factor):
    """Set random brightness."""
    delta = tf.random.uniform([], min_factor, max_factor)
    image = tf.cond(tf.random.uniform([], 0, 1) > 0.5,
                    lambda: tf.image.adjust_brightness(image, delta),
                    lambda: image)
    return image

def random_contrast(image, min_factor=0., max_factor=0.5):
    """Set random contrast."""
    image = tf.image.random_contrast(image, min_factor, max_factor)
    return image


def zoom_random(image, percentage_area=0.95):
    """Zoom image randomly."""
    r_percentage_area = round(random.uniform(percentage_area, 1.), 2)
    h, w = image.get_shape().as_list()[:2]
    w_new = int(floor(w * r_percentage_area))
    h_new = int(floor(h * r_percentage_area))

    random_left_shift = random.randint(0, (w - w_new))
    random_down_shift = random.randint(0, (h - h_new))

    image = tf.cond(tf.random.uniform([], 0, 1) > 0.5,
                    lambda: image,
                    lambda: tf.image.resize(tf.image.crop_to_bounding_box(image,
                                                          random_down_shift,
                                                          random_left_shift,
                                                          h_new,
                                                          w_new),
                                            (h, w))
                   )
    return image


def random_rotate(image, angle=5, radian=True, mode='black'):
    """Rotate image randomly by `rotation` degree."""
    """
    Rotates a 3D tensor (HWD), which represents an image by given radian angle.

    New image has the same size as the input image.

    mode controls what happens to border pixels.
    mode = 'black' results in black bars (value 0 in unknown areas)
    mode = 'white' results in value 255 in unknown areas
    mode = 'ones' results in value 1 in unknown areas
    mode = 'repeat' keeps repeating the closest pixel known
    """
    angle = float(angle)
    if not radian:
        angle = angle * math.pi / 180.

    angle_rotation = random.uniform(-angle, angle)

    s = image.get_shape().as_list()
    assert len(s) == 3, "Input needs to be 3D."
    assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
    image_center = [floor(x/2) for x in s]

    # Coordinates of new image
    coord1 = tf.range(s[0])
    coord2 = tf.range(s[1])

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [s[1]])

    coord2_vec_unordered = tf.tile(coord2, [s[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - image_center[0]
    coord2_vec_centered = coord2_vec - image_center[1]

    coord_new_centered = tf.cast(tf.stack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.dynamic_stitch([[0], [1], [2], [3]],
                                    [[tf.cos(angle_rotation)], [tf.sin(angle_rotation)],
                                     [-tf.sin(angle_rotation)], [tf.cos(angle_rotation)]])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find nearest neighbor in old image
    coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
    coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

    # Clip values to stay inside image coordinates
    if mode == 'repeat':
        coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0]-1)
        coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1]-1)
    else:
        outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0]-1),
                                     tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1]-1),
                                     tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord_old1_clipped = tf.boolean_mask(coord1_old_nn,
                                             tf.logical_not(outside_ind))
        coord_old2_clipped = tf.boolean_mask(coord2_old_nn,
                                             tf.logical_not(outside_ind))

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

    coord_old_clipped = tf.cast(tf.transpose(tf.stack([coord_old1_clipped,
                                                       coord_old2_clipped]),
                                             [1, 0]),
                                tf.int32)

    # Coordinates of the new image
    coord_new = tf.transpose(tf.cast(tf.stack([coord1_vec, coord2_vec]),
                                     tf.int32),
                             [1, 0])

    image_channel_list = tf.split(image, s[2], 2)

    image_rotated_channel_list = list()
    for image_channel in image_channel_list:
        image_chan_new_values = tf.gather_nd(tf.squeeze(image_channel), coord_old_clipped)

        if (mode == 'black') or (mode == 'repeat'):
            background_color = 0
        elif mode == 'ones':
            background_color = 1
        elif mode == 'white':
            background_color = 255

        image_rotated_channel_list.append(
            tf.compat.v1.sparse_to_dense(coord_new,
                                         [s[0], s[1]],
                                         image_chan_new_values,
                                         background_color,
                                         validate_indices=False))

    image_rotated = tf.transpose(
        tf.stack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated


def random_erase(image, rectangle_area):
    """Erase a rectangle area of the image randomly."""
    def _random_erase(image, rectangle_area):
        h, w, c = image.get_shape().as_list()

        w_occlusion1 = int(w * rectangle_area)
        h_occlusion1 = int(h * rectangle_area)

        w_occlusion2 = int(w * 0.1)
        h_occlusion2 = int(h * 0.1)
        if w_occlusion1 > w_occlusion2:
            w_occlusion_min, w_occlusion_max = w_occlusion2, w_occlusion1
        else:
            w_occlusion_min, w_occlusion_max = w_occlusion1, w_occlusion2

        if h_occlusion1 > h_occlusion2:
            h_occlusion_min, h_occlusion_max = h_occlusion2, h_occlusion1
        else:
            h_occlusion_min, h_occlusion_max = h_occlusion1, h_occlusion2

        w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

        if c == 1:
            rectangle = tf.random.uniform((h_occlusion, w_occlusion), 0, 1) * 255
        else:
            rectangle = tf.random.uniform((h_occlusion, w_occlusion, c), 0, 1) * 255

        random_position_x = random.randint(0, w - w_occlusion)
        random_position_y = random.randint(0, h - h_occlusion)

        mask = np.zeros((h, w, 3))
        mask[random_position_y:random_position_y+h_occlusion,
             random_position_x:random_position_x+w_occlusion, :] = 1
        mask_tensor = tf.constant(mask, dtype='float32')
        erasing_image = tf.cast(tf.fill((h, w, 3), 255.), dtype='float32')
        image = mask_tensor * erasing_image + (1 - mask_tensor) * image
        return image

    image = tf.cond(tf.random.uniform([], 0, 1) > 0.5,
                    lambda: image,
                    lambda: _random_erase(image, rectangle_area))
    return image


def random_shift(image, width_shift_range, height_shift_range):
    """Shift image horizontally with the given range."""
    if width_shift_range > 1. or height_shift_range > 1.:
        raise ValueError('Shift range must be less than 1.')

    shape = image.get_shape().as_list()
    height, width = shape[:2]


    wshift_max = int(width * width_shift_range)
    offset_width = 0
    if wshift_max > 0:
        # Calculate number of pixels which must be shifted
        wshift = tf.random.uniform([], 0, wshift_max, dtype=tf.int32)

        # Randomly chose padding position
        wpad_position = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: 0, lambda: -1)
        wseed_values = tf.cond(tf.equal(wpad_position, 0),
                               lambda: image[:, :1, :],
                               lambda: image[:, -1:, :])

        # Create vertical paddings then pad the image
        wpaddings = tf.tile(wseed_values, [1, wshift, 1])
        image = tf.cond(tf.equal(wpad_position, 0),
                        lambda: tf.concat([wpaddings, image], axis=1),
                        lambda: tf.concat([image, wpaddings], axis=1)
                       )
        offset_width = tf.cond(tf.equal(wpad_position, 0),
                               lambda: 0,
                               lambda: wshift)

    hshift_max = int(height * height_shift_range)
    offset_height = 0
    if hshift_max > 0:
        # Calculate number of pixels which must be shifted
        hshift = tf.random.uniform([], 0, hshift_max, dtype=tf.int32)

        # Randomly chose padding position
        hpad_position = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: 0, lambda: -1)

        # Create horizontal paddings then pad the image.
        hseed_values = tf.cond(tf.equal(hpad_position, 0),
                               lambda: image[:1, :, :],
                               lambda: image[-1:, :, :])
        hpaddings = tf.tile(hseed_values, [hshift, 1, 1])
        image = tf.cond(tf.equal(hpad_position, 0),
                        lambda: tf.concat([hpaddings, image], axis=0),
                        lambda: tf.concat([image, hpaddings], axis=0)
                       )
        offset_height = tf.cond(tf.equal(hpad_position, 0),
                                lambda: 0,
                                lambda: hshift)
    cropped_image = tf.image.crop_to_bounding_box(image,
                                                  offset_height=offset_height,
                                                  offset_width=offset_width,
                                                  target_height=height,
                                                  target_width=width)
    return cropped_image


def random_mixup(image, alpha=0.2):
    """
    """
    alpha = tf.cast(alpha, tf.float32)
    beta = tf.distributions.Beta(alpha, alpha)
    lam = beta.sample(tf.shape(image)[0])
    ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)


def random_solarize(image, threshold=128):
    """
    """
    return tf.cond(tf.uniform([], 0, 1) > 0.5,
                   lambda: image,
                   lambda: tf.where(image < threshold, image, 255 - image))


def random_solarize_add(image, addition=0,  threshold=128):
    """
    """
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.cond(tf.uniform([], 0, 1) > 0.5,
                   lambda: image,
                   lambda: tf.where(image < threshold, added_image, image))


def random_posterize(image, bits):
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.cond(tf.uniform([], 0, 1) > 0.5,
                   lambda: image,
                   lambda: tf.bitwise.left_shift(
                       tf.bitwise.right_shift(image, shift), shift))
