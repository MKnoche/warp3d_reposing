from . import tfutil
import tensorflow as tf
from . import resnet_v2
import numpy as np
import tensorflow.contrib.slim as slim
import imageio

import cv2


def get_iper_intrinsics():
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = 1125 * 257 / 1024
    intrinsics[1, 1] = 1125 * 257 / 1024
    intrinsics[:2, 2] = 257 / 2
    intrinsics = np.expand_dims(intrinsics, 0)
    return intrinsics


def main():
    im = imageio.imread('/globalwork/datasets/iper/images/001_1_1/frame_000000.jpg')
    im = cv2.resize(im, (257, 257))[np.newaxis].astype(np.float32) / 255 * 2 - 1
    im = np.transpose(im, [0, 3, 1, 2])

    # Simpler, unscaled ResNet50 root-relative estimation
    # pose = predict_pose3d(tf.convert_to_tensor(im))
    # path = ('/globalwork/sarandi/trainings/2020-02-01/iper/metric_nocrop_seed1/model.ckpt-160684')
    # with tf.Session() as sess:
    #     load_pretrained(sess, path, 'resnet_v2_50')
    #     print(sess.run(pose))

    # Absolute non-root-relative estimation in millimeters
    pose = predict_pose3d_abs(tf.convert_to_tensor(im), tf.convert_to_tensor(get_iper_intrinsics()))

    # Or by setting intrinsics=None it assumes 60 degree field of view by default:
    # pose = predict_pose3d_abs(tf.convert_to_tensor(im), intrinsics=None)

    model_path = ('/globalwork/sarandi/trainings/2020-02-01/'
                  'merged/with_coco_resnet152_fulldata_upperbodyaug/model.ckpt-515515')
    with tf.Session() as sess:
        load_weights(sess, model_path, 'resnet_v2_152')
        pose_arr = sess.run(pose)
        joint_names = ['neck', 'nose', 'lsho', 'lelb', 'lwri', 'lhip', 'lkne', 'lank', 'rsho',
                       'relb', 'rwri', 'rhip', 'rkne', 'rank', 'leye', 'lear', 'reye', 'rear',
                       'pelv']
        edges = [(1, 0), (0, 18), (0, 2), (2, 3), (3, 4), (0, 8), (8, 9), (9, 10), (18, 5), (5, 6),
                 (6, 7), (18, 11), (11, 12), (12, 13), (15, 14), (14, 1), (17, 16), (16, 1)]
        visualize_pose(image=np.transpose(im, [0, 2, 3, 1])[0], coords=pose_arr[0], edges=edges)


def predict_pose3d_abs(im, intrinsics=None):
    proc_side = 257

    if intrinsics is None:
        # Assume a default of 60 degree FOV
        fov = np.deg2rad(60)
        f = proc_side / (2 * np.tan(fov / 2))
        intrinsics_np = np.array([[f, 0, proc_side / 2], [0, f, proc_side / 2], [0, 0, 1]])
        intrinsics = tf.convert_to_tensor(intrinsics_np, dtype=tf.float32)
        intrinsics = tf.tile(tf.expand_dims(intrinsics, 0), [tf.shape(im)[0], 1, 1])

    mirror_joint_mapping = [
        0, 1, 8, 9, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 16, 17, 14, 15, 18, 19, 20, 22, 21, 24, 23,
        25, 26, 27, 29, 28, 30, 31, 32, 33, 35, 34, 36]
    coords2d, coords3d, weights = image_to_unscaled_coords(im)

    # Horizontal flip augmentation
    coords2d_flip, coords3d_flip, weights_flip = image_to_unscaled_coords(im[:, :, :, ::-1])
    # To get back to the original (non-flipped) coordinate frame, we need to swap
    # left and right joints along the joint ID axis...
    coords2d_flip = tf.gather(coords2d_flip, mirror_joint_mapping, axis=1)
    coords3d_flip = tf.gather(coords3d_flip, mirror_joint_mapping, axis=1)
    # ... and subtract the x coordinates from 1
    coords2d_flip = tf.concat([1 - coords2d_flip[..., :1], coords2d_flip[..., 1:]], axis=-1)
    coords3d_flip = tf.concat([1 - coords3d_flip[..., :1], coords3d_flip[..., 1:]], axis=-1)

    # Then average the flipped and non-flipped results
    weights_flip = tf.gather(weights_flip, mirror_joint_mapping, axis=1)
    coords2d = (coords2d + coords2d_flip) * 0.5
    coords3d = (coords3d + coords3d_flip) * 0.5
    weights = (weights + weights_flip) * 0.5

    # Scale the coordinates to be in pixels (2D) and millimeters (3D)
    box_size = 2200
    coords3d_rel = tf.concat([
        coords3d[:, :, :2] * box_size * (proc_side - 1) / proc_side,
        coords3d[:, :, 2:] * box_size], axis=-1)
    coords2d = coords2d * (proc_side - 1)

    # Normalize the image coordinates to be intrinsics invariant
    inv_intrinsics = tf.linalg.inv(intrinsics)
    coords2d_homog = tf.concat([coords2d, tf.ones_like(coords2d[..., :1])], axis=-1)
    coords2d_normalized = tf.einsum('Bij,BCj->BCi', inv_intrinsics, coords2d_homog)

    # Reconstruct the unknown reference point
    coords_abs_3d_based, ref = reconstruct_ref(coords2d_normalized[:, :, :2], coords3d_rel, weights)

    # Reproject the result into image coordinates
    coords2d_reprojected = coords_abs_3d_based / coords_abs_3d_based[..., 2:]
    coords2d_reproj_pixels = tf.einsum('Bij,BCj->BCi', intrinsics[:, :2], coords2d_reprojected)

    # Check if the reprojected joints are within the image boundary (field-of-view)
    is_predicted_to_be_in_fov = tf.reduce_all(
        tf.logical_and(coords2d_reproj_pixels >= 0, coords2d_reproj_pixels < proc_side),
        axis=-1, keepdims=True)
    is_predicted_to_be_in_fov = tf.tile(is_predicted_to_be_in_fov, [1, 1, 3])

    # Back-project the 2D head's predicted coordinates according to the depths from the
    # 3D head and reconstruction.
    coords_abs_2d_based = (coords2d_normalized *
                           (coords3d_rel[:, :, 2:] + tf.expand_dims(ref[:, 2:], 1)))

    # Prefer the backprojected variant for joints within the image
    coords_abs = tf.where(is_predicted_to_be_in_fov, coords_abs_2d_based, coords_abs_3d_based)

    # Return the first 19 joints, these correspond to the COCO/OpenPose/CMU-Panoptic joints
    return coords_abs[:, :19]


def reconstruct_ref(normalized_2d, coords3d_delta, weights):
    """Reconstructs the reference point location.

    Args:
      normalized_2d: normalized image coordinates of the joints
         (without intrinsics applied), shape [batch_size, n_points, 2]
      coords3d_delta: 3D camera coordinate offsets from the unknown reference
         point which we want to reconstruct, shape [batch_size, n_points, 3]
      weights: how important each joint should be in the weighted linear least squares optimization
         shape [batch_size, n_points]

    Returns:
      The 3D reference point in camera coordinates, shape [batch_size, 3]
    """

    def root_mean_square(x):
        return tf.sqrt(tf.reduce_mean(tf.square(x)))

    n_batch = tf.shape(normalized_2d)[0]
    n_points = normalized_2d.get_shape().as_list()[1]
    reshaped2d = tf.reshape(normalized_2d, [-1, n_points * 2, 1])
    scale2d = root_mean_square(reshaped2d)
    eyes = tf.tile(tf.expand_dims(tf.eye(2, 2), 0), [n_batch, n_points, 1])
    expanded_weights = tf.sqrt(tf.reshape(
        tf.tile(tf.expand_dims(weights, axis=-1), [1, 1, 2]), [-1, n_points * 2, 1]))

    A = tf.concat([eyes, -reshaped2d / scale2d], axis=2)
    rel_backproj = normalized_2d * coords3d_delta[:, :, 2:] - coords3d_delta[:, :, :2]
    b = tf.reshape(rel_backproj, [-1, n_points * 2, 1])
    scale_b = root_mean_square(b)
    b = b / scale_b
    ref = tf.linalg.lstsq(
        A * expanded_weights,
        b * expanded_weights, fast=True)

    ref = tf.concat([ref[:, :2], ref[:, 2:] / scale2d], 1) * scale_b
    ref = tf.squeeze(ref, -1)
    coords_abs = coords3d_delta + tf.reshape(ref, [-1, 1, 3])
    return coords_abs, ref


def image_to_unscaled_coords(im):
    depth = 8
    n_joints = 37
    stride = 32
    n_outs = [n_joints, n_joints + depth * n_joints]
    net_outputs = resnet_v2_spatial(
        im, resnet_v2.resnet_v2_152, n_out=n_outs, stride=stride, split_after_block=3,
        is_training=False, reuse=tf.AUTO_REUSE, scope='MainPart')
    side = net_outputs[0].get_shape().as_list()[2]

    # 3D head
    logits3d = net_outputs[1][:, :-n_joints]
    logits3d = tf.reshape(logits3d, [-1, depth, n_joints, side, side])
    logits3d = tf.transpose(logits3d, [0, 2, 3, 4, 1])
    softmaxed3d = tfutil.softmax(logits3d, axis=[2, 3, 4])
    coords3d = tf.stack(tfutil.decode_heatmap(softmaxed3d, [3, 2, 4]), axis=-1)

    # 2D head
    logits2d = net_outputs[0]
    softmaxed2d = tfutil.softmax(logits2d, axis=[2, 3])
    coords2d = tf.stack(tfutil.decode_heatmap(softmaxed2d, [3, 2]), axis=-1)

    # Weight head
    confidences = net_outputs[1][:, -n_joints:]
    confidences = tf.reduce_sum(confidences * tf.reduce_sum(softmaxed3d, axis=4), axis=[2, 3])
    weights = tfutil.softmax(confidences, axis=1)
    return coords2d, coords3d, weights


def predict_pose3d(im):
    depth = 8
    n_joints = 19
    stride = 32
    net_output = resnet_v2_spatial(
        im, resnet_v2.resnet_v2_50, n_out=[depth * n_joints], stride=stride, is_training=False,
        reuse=tf.AUTO_REUSE, scope='MainPart')[0]
    n, c, h, w = net_output.get_shape().as_list()
    reshaped = tf.reshape(net_output, [-1, depth, n_joints, h, w])
    transposed = tf.transpose(reshaped, [0, 2, 3, 4, 1])
    return root_relative(tfutil.soft_argmax(transposed, [3, 2, 4])), transposed


def root_relative(coords):
    return coords - coords[:, -1:]


def load_weights(sess, weight_path, resnet_name='resnet_v2_50'):
    checkpoint_scope = f'MainPart/{resnet_name}'
    loaded_scope = f'MainPart/{resnet_name}'
    do_not_load = ['Adam', 'Momentum']

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=loaded_scope)
    var_dict = {v.op.name[v.op.name.index(checkpoint_scope):]: v for v in var_list}
    var_dict = {k: v for k, v in var_dict.items() if
                not any(excl in k for excl in do_not_load)}
    saver = tf.train.Saver(var_list=var_dict)
    saver.restore(sess, weight_path)


def resnet_arg_scope(is_training):
    batch_norm_params = dict(
        decay=0.997, epsilon=1e-5, scale=True, is_training=is_training, fused=True,
        data_format='NCHW')

    with slim.arg_scope(
            [slim.conv2d, slim.conv3d],
            weights_regularizer=slim.l2_regularizer(1e-4),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        with slim.arg_scope(
                [slim.conv2d, slim.conv3d, slim.conv3d_transpose, slim.conv2d_transpose,
                 slim.avg_pool2d, slim.separable_conv2d, slim.max_pool2d, slim.batch_norm,
                 slim.spatial_softmax],
                data_format='NCHW'):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc


@tfutil.in_variable_scope('Resnet_spatial', mixed_precision=True)
def resnet_v2_spatial(inp, resnet_fn, n_out, stride, is_training, split_after_block=5):
    with slim.arg_scope(resnet_arg_scope(is_training)):
        x = tf.cast(inp, tf.float16)
        xs, end_points = resnet_fn(
            x, num_classes=n_out, is_training=is_training, global_pool=False,
            output_stride=stride, split_after_block=split_after_block)
        xs = [tf.cast(x, tf.float32) for x in xs]
        return xs


def visualize_pose(image, coords, edges):
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    # Matplotlib interprets the Z axis as vertical, but our pose
    # has Y as the vertical axis.
    # Therefore we do a 90 degree rotation around the horizontal (X) axis
    coords2 = coords.copy()
    coords[:, 1], coords[:, 2] = coords2[:, 2], -coords2[:, 1]

    fig = plt.figure()
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.set_title('Input')
    image_ax.imshow(image)

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.set_title('Prediction')
    range_ = 800
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-range_, range_)

    for i_start, i_end in edges:
        pose_ax.plot(*zip(coords[i_start], coords[i_end]), marker='o', markersize=2)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
