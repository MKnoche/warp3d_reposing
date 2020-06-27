import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
import time
from pose3d_minimal.main import predict_pose3d

tfgan = tf.contrib.gan

perception_model = None


def init_perception_model():
    global perception_model
    start = time.time()
    with tf.name_scope('Perceptual'):
        vgg = vgg19.VGG19(weights='imagenet', include_top=False)
        perception_model = Model(inputs=vgg.input, outputs=[
            vgg.get_layer('block1_conv2').output,
            vgg.get_layer('block2_conv2').output,
            vgg.get_layer('block3_conv2').output,
            vgg.get_layer('block4_conv2').output,
            vgg.get_layer('block5_conv2').output
        ])
        for layer in perception_model.layers:
            layer.trainable = False

    print('Loaded perception model:', time.time() - start)


def perception_output(x):
    if perception_model is None:
        raise RuntimeError('perception model is not initialized')

    def preprocess_for_vgg(x):
        x = 255 * (x + 1) / 2
        mean = tf.constant([103.939, 116.779, 123.68])
        mean = tf.reshape(mean, (1, 1, 1, 3))
        x = x - mean
        x = x[..., ::-1]
        return x

    x = preprocess_for_vgg(x)
    x = perception_model(x)
    return x


def get_feature_loss(target, generated):
    target = perception_output(target)
    generated = perception_output(generated)
    loss = 0
    for t, g, w in zip(target, generated, [1. / 32, 1. / 16, 1. / 8, 1. / 4, 1.]):
        loss += w * tf.reduce_mean(tf.abs(tf.subtract(t, g)))
    return loss


def init_pose_model(sess, weight_path):
    start = time.time()
    checkpoint_scope = f'MainPart/resnet_v2_50'
    loaded_scope = f'Pose/MainPart/resnet_v2_50'
    do_not_load = ['Adam', 'Momentum']

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=loaded_scope)
    var_dict = {v.op.name[v.op.name.index(checkpoint_scope):]: v for v in var_list}
    var_dict = {k: v for k, v in var_dict.items() if not any(excl in k for excl in do_not_load)}

    saver = tf.train.Saver(var_list=var_dict)
    saver.restore(sess, weight_path)
    print('Loaded pose model:', time.time() - start)


def get_pose_loss(target, generated):
    with tf.variable_scope('Pose', reuse=tf.AUTO_REUSE):
        target = tf.transpose(target, (0, 3, 1, 2)) / 2
        generated = tf.transpose(generated, (0, 3, 1, 2)) / 2
        target, target_logits = predict_pose3d(target)
        generated, generated_logits = predict_pose3d(generated)
        return tf.reduce_mean(tf.abs(target - generated)) * 2.2
