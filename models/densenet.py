"""This module consists of densenet implementation."""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


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


def bottleneck_block(inputs, growth_rate, data_format=None):
    """
    """
    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)
    inputs = conv2d_fixed_padding(inputs, filters=growth_rate * 4, kernel_size=1,
                                  strides=1, data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)
    inputs = conv2d_fixed_padding(inputs, filters=growth_rate, kernel_size=3,
                                  strides=1, data_format=data_format)
    return inputs


def dense_block(inputs, num_blocks, growth_rate, data_format=None):
    """
    """
    axis = 1 if data_format == 'channels_first' else 3
    x = inputs
    for _ in range(num_blocks):
        y = bottleneck_block(x, growth_rate, data_format)
        x = layers.concatenate([x, y], axis=axis)
    return x


def transition_layer(inputs, growth_rate, theta, data_format=None):
    """
    """
    inputs = batch_norm(inputs, data_format)
    inputs = layers.Activation('relu')(inputs)
    num_filters = int(growth_rate * theta)
    inputs = conv2d_fixed_padding(inputs, filters=num_filters,
                                  kernel_size=1, strides=1,
                                  data_format=data_format)

    inputs = layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='same',
                                 data_format=data_format)(inputs)
    return inputs


def densenet_model(input_shape, num_classes,
                   first_kernel_size, first_conv_strides,
                   first_pool_size, first_pool_strides,
                   blocks, growth_rate, theta, include_top=True,
                   data_format=None):
    """Build a keras model with custom params."""
    inputs = layers.Input(input_shape)
    x = conv2d_fixed_padding(inputs=inputs, filters=growth_rate * 2,
                             kernel_size=first_kernel_size,
                             strides=first_conv_strides,
                             data_format=data_format)
    if first_pool_size:
        x = layers.MaxPooling2D(pool_size=first_pool_size,
                                strides=first_pool_strides,
                                data_format=data_format)(x)

    for i, num_blocks in enumerate(blocks):
        x = dense_block(x, num_blocks, growth_rate, data_format)
        if i < len(blocks) - 1:
            x = transition_layer(x, growth_rate, theta, data_format)

    x = batch_norm(x, data_format)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D(data_format=data_format)(x)
    if include_top:
        x = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, x)


def densenet_121(input_shape, num_classes=None,
                 include_top=True, data_format=None):
    return densenet_model(
        input_shape=input_shape,
        num_classes=num_classes,
        include_top=include_top,
        first_kernel_size=7,
        first_conv_strides=2,
        first_pool_size=3,
        first_pool_strides=2,
        blocks=[6, 12, 24, 16],
        growth_rate=32,
        theta=0.5
    )


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    num_classes = 100
    model = densenet_121(input_shape, num_classes)
    model.summary()
