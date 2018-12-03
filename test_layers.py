import tensorflow as tf


def all_views_conv_layer(input_layer, layer_name, number_of_filters=32, filter_size=(3, 3), stride=(1, 1),
                         padding='VALID', biases_initializer=tf.zeros_initializer()):
    """Convolutional layers for 2x2 views input 4-DCN"""


    h = tf.contrib.layers.convolution2d(inputs=input_layer, num_outputs=number_of_filters,
                                                 kernel_size=filter_size, stride=stride, padding=padding,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=biases_initializer)

    return h