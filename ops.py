import tensorflow as tf


def _variable_on_cpu(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.01)):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name : name of the variable
      shape : list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor"""

    with tf.device('/cpu:0'):
        if shape is None:
            var = tf.get_variable(name, initializer=initializer)
        else:
            var = tf.get_variable(name, shape, initializer=initializer)

    return var


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers."""
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      # ( (grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN) )
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared across tower.
        # So .. we will just return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def conv_conv_pool(input_, n_filters, activation, w_bn, is_train, name, pool=True):
    net = input_

    with tf.variable_scope("layer" + name):
        for i, F in enumerate(n_filters):
            net = conv2d(net, F, name="conv" + str(i + 1), activation=activation,
                         w_bn=w_bn, is_train=is_train)

        if not pool:
            return net
        else:
            return net, maxpool(net)


def conv2d(input_, output_channels, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.01, name="conv_layer",
           activation=None, initializer=tf.truncated_normal_initializer,
           w_bias=True, w_bn=False, is_train=True):
    with tf.variable_scope(name):
        shape = [k_h, k_w, input_.get_shape()[-1], output_channels]
        kernel = _variable_on_cpu(name="kernel", shape=shape,
                                  initializer=initializer(stddev=stddev))
        bias = _variable_on_cpu(name="bias", shape=[output_channels],
                                initializer=tf.zeros_initializer)
        if w_bias:
            conv = tf.nn.conv2d(input_, kernel, strides=[
                                1, d_h, d_w, 1], padding="SAME") + bias
        else:
            conv = tf.nn.conv2d(input_, kernel, strides=[
                                1, d_h, d_w, 1], padding="SAME")

        if w_bn:
            conv = tf.layers.batch_normalization(
                conv, training=is_train, name="BN")

    if activation is None:
        return conv
    else:
        return activation(conv)


def linear(input_, output_size, stddev=0.01, name="fc_layer",
           activation=None, initializer=tf.truncated_normal_initializer,
           w_bias=True, w_bn=False, is_train=True):
    with tf.variable_scope(name):
        shape = [input_.get_shape()[-1], output_size]
        matrix = tf.get_variable(name="Matrix", shape=shape,
                                 initializer=initializer(stddev=stddev))
        bias = tf.get_variable(name="bias", shape=[output_size],
                               initializer=tf.zeros_initializer)
        if w_bias:
            fc = tf.matmul(input_, matrix) + bias
        else:
            fc = tf.matmul(input_, matrix)

        if w_bn:
            fc = tf.layers.batch_normalization(
                fc, training=is_train, name="BN")

    if activation is None:
        return fc
    else:
        return activation(fc)


def tanh(x):
    return tf.nn.tanh(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def softmax(x):
    return tf.nn.softmax(x)


def relu(x):
    return tf.nn.relu(x)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def elu(x):
    return tf.nn.elu(x)


def maxpool(x, k_h=2, k_w=2, s_h=2, s_w=2):
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding="SAME")


def avg_unpool(input_):
    outt = tf.concat([input_, input_], 3)
    out = tf.concat([outt, outt], 2)
    sh = input_.get_shape().as_list()
    return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])


def resize_nn(input_, scale=2):
    sh = input_.get_shape().as_list()
    re_sh = [scale * sh[1], scale * sh[2]]
    return tf.image.resize_nearest_neighbor(input_, (re_sh[0], re_sh[1]))
