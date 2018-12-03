import test_layers as layers



def baseline(x,  nodropout_probability=None, gaussian_noise_std=None):


    h = layers.all_views_conv_layer(x, 'conv1', number_of_filters=32, filter_size=[3, 3], stride=[2, 2])

        # second conv sequence
    h = layers.all_views_max_pool(h, stride=[3, 3])

    h = layers.all_views_global_avg_pool(h)
    h = layers.all_views_flattening_layer(h) #flatening and concatenation
    h = layers.fc_layer(h, number_of_units=1024)

    h = layers.dropout_layer(h, nodropout_probability)

    y_prediction_birads = layers.softmax_layer(h, number_of_outputs=3)

    return  y_prediction_birads