"""This module consists of seresnet version 2 implementation."""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling



def batch_norm(inputs, data_format=None):
    """BatchNormalization layer with specific params."""
    return layers.BatchNormalization(
        axis=3 if data_format == 'channels_last' else 1,
        momentum=0.997, epsilon=1e-5)(inputs)


def conv2d_fixed_padding(inputs, filters, kernel_size,
                         strides, data_format=None):
    """Convolution layer with fixed padding."""
    if strides > 1:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = layers.ZeroPadding2D(padding=[pad_beg, pad_end],
                                      data_format=data_format)(inputs)

    return layers.Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides,
                         padding='same' if strides == 1 else 'valid',
                         kernel_initializer='he_normal',
                         use_bias=False, data_format=data_format)(inputs)


def seblock(inputs, channels, ratio=16, data_format=None):
    """Squeeze-and-Excitation block."""
    x = layers.GlobalAveragePooling2D(data_format=data_format)(inputs)
    x = layers.Dense(channels // ratio,
                     activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.Dense(channels, activation='sigmoid')(x)
    x = layers.multiply([inputs, x])
    return x


def block_v2(inputs, filters, kernel_size,
             strides, projection_shortcut,
             data_format=None):
    """A resnet block version 2.

    Args
    :inputs: the input tensor.
    :filters: the number of filters.
    :kernel_size: kernel size of convolution layer.
    :strides: stride of the block.
    :projection_shortcut: the functino use for projection shortcuts.
    :data_format: the input format.

    Returns
        The tensor of the block.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs, filters, kernel_size=3,
                                  strides=strides, data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)
    inputs = conv2d_fixed_padding(inputs, filters, kernel_size=3,
                                  strides=1, data_format=data_format)

    # Apply seblock here
    inputs = seblock(inputs, filters, 16, data_format)
    return inputs + shortcut


def bottleneck_v2(inputs, filters, strides,
                 projection_shortcut, data_format=None):
    """A resnet block version 2 with bottleneck.

    Args
    :inputs: the input tensor.
    :filters: the number of filters.
    :kernel_size: kernel size of convolution layer.
    :strides: stride of the block.
    :projection_shortcut: the functino use for projection shortcuts.
    :data_format: the input format.

    Returns
        The tensor of the block output.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs, filters, kernel_size=1,
                                  strides=1, data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)
    inputs = conv2d_fixed_padding(inputs, filters, kernel_size=3,
                                  strides=strides, data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)
    inputs = conv2d_fixed_padding(inputs, filters * 4, kernel_size=1,
                                  strides=1, data_format=data_format)
    # Apply seblock here
    inputs = seblock(inputs, filters * 4, 16, data_format)
    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                data_format=None, name=None):
    """Build a resnet layer.

    Args
    :inputs: the input tensor.
    :filters: the number of filters.
    :bottleneck: is the layer created bottleneck blocks.
    :block_fn: the function use to build blocks.
    :strides: the strides for the first convolution layer.
    :data_format: the input format.

    Returns
        The tensor of the layer output.
    """
    filters_out = filters  * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs, filters_out,
                                    kernel_size=1, strides=strides,
                                    data_format=data_format)

    inputs = block_fn(inputs, filters, strides, projection_shortcut, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, 1, None, data_format)
    return inputs


def senet_model(input_shape, num_classes, bottleneck, num_filters,
                first_kernel_size, first_conv_strides,
                first_pool_size, first_pool_strides,
                blocks, block_strides, include_top=True,
                data_format=None):
    """Build a keras model with custom params."""
    if bottleneck:
        block_fn = bottleneck_v2
    else:
        block_fn = block_v2

    inputs = layers.Input(input_shape)
    x = conv2d_fixed_padding(inputs=inputs, filters=num_filters,
                             kernel_size=first_kernel_size,
                             strides=first_conv_strides,
                             data_format=data_format)
    if first_pool_size:
        x = layers.MaxPooling2D(pool_size=first_pool_size,
                                strides=first_pool_strides,
                                padding='same',
                                data_format=data_format)(x)

    for i, num_blocks in enumerate(blocks):
        filters = num_filters * (2**i)
        x = block_layer(inputs=x, filters=filters,
                        bottleneck=bottleneck, block_fn=block_fn,
                        blocks=num_blocks, strides=block_strides[i],
                        data_format=data_format)

    x = batch_norm(x, data_format)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D(data_format=data_format)(x)
    if include_top:
        x = layers.Dense(units=num_classes,
                         kernel_initializer='he_normal',
                         activation='softmax')(x)

    model = Model(inputs, x)
    return model


def senet18_v2(input_shape, num_classes=None,
                include_top=True, data_format=None):
    return senet_model(
        input_shape=input_shape,
        num_filters=16,
        bottleneck=True,
        num_classes=num_classes,
        first_kernel_size=7,
        first_conv_strides=2,
        first_pool_size=3,
        first_pool_strides=2,
        blocks=[2, 2, 2, 2],
        block_strides=[1, 2, 2, 2],
        include_top=include_top,
        data_format=None
    )

