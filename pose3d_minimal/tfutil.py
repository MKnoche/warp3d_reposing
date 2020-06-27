import tensorflow as tf
import functools


def soft_argmax(inp, axis):
    softmaxed = softmax(inp, axis=axis)
    return tf.stack(decode_heatmap(softmaxed, axis=axis), axis=-1)


def softmax(target, axis=-1, name=None):
    with tf.name_scope(name, 'softmax', values=[target]):
        max_along_axis = tf.reduce_max(target, axis, keepdims=True)
        exponentiated = tf.exp(target - max_along_axis)
        normalizer_denominator = tf.reduce_sum(exponentiated, axis, keepdims=True)
        return exponentiated / normalizer_denominator


def jensen_shannon_loss(logit1, logit2, axis=-1, name=None):
    with tf.name_scope(name, 'jensen_shannon', values=[logit1, logit2]):
        probability1 = softmax(logit1, axis=axis)
        probability2 = softmax(logit2, axis=axis)
        logsumexp1 = tf.reduce_logsumexp(logit1, axis=axis, keepdims=True)
        logsumexp2 = tf.reduce_logsumexp(logit2, axis=axis, keepdims=True)
        logit_diff = logit1 - logit2
        probability_diff = probability1 - probability2
        logsumexp_diff = logsumexp1 - logsumexp2
        return 0.5 * tf.reduce_sum(probability_diff * (logit_diff - logsumexp_diff), axis=axis)


def decode_heatmap(inp, axis=-1):
    shape = inp.get_shape().as_list()
    ndims = inp.get_shape().ndims

    def relative_coords_along_axis(ax):
        grid_shape = [1] * ndims
        grid_shape[ax] = shape[ax]
        grid = tf.reshape(tf.linspace(0.0, 1.0, shape[ax]), grid_shape)
        return tf.cast(grid, inp.dtype)

    # Single axis:
    if not isinstance(axis, (tuple, list)):
        return tf.reduce_sum(relative_coords_along_axis(axis) * inp, axis=axis)

    # Multiple axes.
    # Convert negative axes to the corresponding positive index (e.g. -1 means last axis)
    heatmap_axes = [ax if ax >= 0 else ndims + ax + 1 for ax in axis]
    result = []
    for ax in heatmap_axes:
        other_heatmap_axes = tuple(set(heatmap_axes) - {ax})
        summed_over_other_axes = tf.reduce_sum(inp, axis=other_heatmap_axes, keepdims=True)
        coords = relative_coords_along_axis(ax)
        decoded = tf.reduce_sum(coords * summed_over_other_axes, axis=ax, keepdims=True)
        result.append(tf.squeeze(decoded, heatmap_axes))

    return result


def in_variable_scope(default_name, mixed_precision=True):
    """Puts the decorated function in a TF variable scope with the provided default name.
    The function also gains two extra arguments: "scope" and "reuse" which get passed to
    tf.variable_scope.
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, scope=None, reuse=None, **kwargs):
            with tf.variable_scope(
                    scope, default_name, reuse=reuse,
                    custom_getter=mixed_precision_getter if mixed_precision else None):
                return f(*args, **kwargs)

        return decorated

    return decorator


def mixed_precision_getter(
        getter, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True,
        *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the compute precision."""
    # print(f'mixed prec asked for {dtype} ({name})')
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(
        name, shape, dtype=storage_dtype, initializer=initializer, regularizer=regularizer,
        trainable=trainable, *args, **kwargs)

    if storage_dtype != dtype:
        return tf.cast(variable, dtype)

    return variable
