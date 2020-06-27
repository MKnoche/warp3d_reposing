import tensorflow as tf
from parameters import params
import numpy as np
from utils import extend_spatial_sizes, reduce_spatial_sizes
from adapted_resnet_v2 import resnet_v2_block, resnet_v2, resnet_arg_scope, bottleneck_transposed


def group_norm(x):
    if x.shape[-1] >= 32:
        return tf.contrib.layers.group_norm(x)
    else:
        return tf.contrib.layers.layer_norm(x)


def build_coords(shape):
    xx, yy, zz = tf.meshgrid(tf.range(shape[1]), tf.range(shape[0]), tf.range(shape[2]))  # in image notation
    ww = tf.ones(xx.shape)
    coords = tf.concat([tf.expand_dims(tf.cast(a, tf.float32), -1) for a in [xx, yy, zz, ww]], axis=-1)
    return coords


# input in matrix notation
def transform_single(volume, transform, interpolation):
    volume = tf.transpose(volume, [1, 0, 2, 3])  # switch to image notation
    coords = build_coords(volume.shape[:3])
    coords_shape = coords.shape
    coords_reshaped = tf.reshape(coords, [-1, 4])
    pointers_reshaped = tf.linalg.matmul(transform, coords_reshaped, transpose_b=True)
    pointers = tf.reshape(tf.transpose(pointers_reshaped, [1, 0]), coords_shape)  # undo transpose_b
    pointers = pointers[:, :, :, :3]
    if interpolation == 'NEAREST':
        pointers = tf.cast(tf.math.round(pointers), dtype=tf.int32)
        with tf.device('/gpu:0'):
            res = tf.gather_nd(volume, pointers)
    elif interpolation == 'TRILINEAR':
        c3s = {}
        for c in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
            c3s[c] = tf.gather_nd(volume, tf.cast(tf.floor(pointers), dtype=tf.int32) + c)
        d = pointers - tf.floor(pointers)
        c2s = {}
        for c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            c2s[c] = c3s[(0,) + c] * (1 - d[:, :, :, 0:1]) + c3s[(1,) + c] * (d[:, :, :, 0:1])
        c1s = {}
        for c in [(0,), (1,)]:
            c1s[c] = c2s[(0,) + c] * (1 - d[:, :, :, 1:2]) + c2s[(1,) + c] * (d[:, :, :, 1:2])
        res = c1s[(0,)] * (1 - d[:, :, :, 2:3]) + c1s[(1,)] * (d[:, :, :, 2:3])
    else:
        raise ValueError
    return res


def volumetric_transform(volumes, transforms, interpolation='NEAREST'):
    return tf.map_fn(lambda x: transform_single(x[0], x[1], interpolation), (volumes, transforms), dtype=tf.float32,
                     parallel_iterations=128)


def warp_3d(vol_batch, masks_batch, transform_batch, reduce=True):
    n, h, w, d, c = vol_batch.get_shape().as_list()
    with tf.name_scope('warp_3d'):
        net = {}

        part_count = transform_batch.shape[1]

        net['bodypart_masks'] = masks_batch

        init_volume_size = (params['image_size'], params['image_size'], params['image_size'])
        z_scale = (d - 1) / (h - 1)
        v_scale = (h - 1) / init_volume_size[0]
        affine_mul = [[1, 1, 1 / z_scale, v_scale],
                      [1, 1, 1 / z_scale, v_scale],
                      [z_scale, z_scale, 1, v_scale * z_scale],
                      [1, 1, 1 / z_scale, 1]]
        affine_mul = np.array(affine_mul).reshape((1, 1, 4, 4))
        affine_transforms = transform_batch * affine_mul
        affine_transforms = tf.reshape(affine_transforms, (-1, 4, 4))

        expanded_tensor = tf.expand_dims(vol_batch, -1)
        multiples = [1, part_count, 1, 1, 1, 1]
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, (
            n * part_count, h, w, d, c))

        transposed_masks = tf.transpose(masks_batch, [0, 4, 1, 2, 3])
        reshaped_masks = tf.reshape(transposed_masks, [n * part_count, h, w, d])
        repeated_tensor = repeated_tensor * tf.expand_dims(reshaped_masks, axis=-1)

        net['masked_bodyparts'] = repeated_tensor
        warped = volumetric_transform(repeated_tensor, affine_transforms, interpolation='TRILINEAR')
        net['masked_bodyparts_warped'] = warped

        res = tf.reshape(warped, [-1, part_count, h, w, d, c])
        res = tf.transpose(res, [0, 2, 3, 4, 1, 5])
        if reduce:
            res = tf.reduce_max(res, reduction_indices=[-2])
        return res, net


def warp_2d_3d(vol_batch, masks_batch, transform_batch, reduce=True):
    n, h, w, d, c = vol_batch.get_shape().as_list()
    with tf.name_scope('warp_2d_3d'):
        net = {}
        part_count = transform_batch.shape[1]

        # MASKS 3D

        net['bodypart_masks'] = masks_batch

        img_batch = tf.reshape(vol_batch, (n, h, w, d * c))

        init_image_size = (params['image_size'], params['image_size'])
        affine_mul = [1, 1, init_image_size[0] / h,
                      1, 1, init_image_size[1] / w,
                      1, 1]
        affine_mul = np.array(affine_mul).reshape((1, 1, 8))

        expanded_tensor = tf.expand_dims(img_batch, -1)
        multiples = [1, part_count, 1, 1, 1]
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(img_batch) * np.array([part_count, 1, 1, 1]))

        affine_transforms = transform_batch / affine_mul
        affine_transforms = tf.reshape(affine_transforms, (-1, 8))

        transposed_masks = tf.transpose(masks_batch, [0, 3, 1, 2])
        reshaped_masks = tf.reshape(transposed_masks, [n * part_count, h, w])
        repeated_tensor = repeated_tensor * tf.expand_dims(reshaped_masks, axis=-1)
        warped = tf.contrib.image.transform(repeated_tensor, affine_transforms)
        res = tf.reshape(warped, [-1, part_count, h, w, d * c])
        res = tf.transpose(res, [0, 2, 3, 1, 4])
        if reduce:
            res = tf.reduce_max(res, reduction_indices=[-2])

        res = tf.reshape(res, (n, h, w, d, -1))
        return res, net


def residual_unit_3d(x):
    filters = x.shape[-1]
    r = x
    for i in range(2):
        r = group_norm(r)
        r = tf.nn.relu(r)
        r = tf.layers.conv3d(r, filters, kernel_size=3, padding='SAME')
    return x + r


def tf_pose_map_3d(poses, shape):
    y = tf.unstack(poses, axis=1)
    y[0], y[1] = y[1], y[0]
    poses = tf.stack(y, axis=1)
    image_size = tf.constant(params['image_size'], tf.float32)
    shape = tf.constant(shape, tf.float32)
    sigma = tf.constant(6, tf.float32)
    poses = tf.unstack(poses, axis=0)
    pose_mapss = []
    for pose in poses:
        pose = pose / image_size * shape[:, tf.newaxis]
        joints = tf.unstack(pose, axis=-1)
        pose_maps = []
        for joint in joints:
            xx, yy, zz = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), tf.range(shape[2]), indexing='ij')
            mesh = tf.cast(tf.stack([xx, yy, zz]), dtype=tf.float32)
            pose_map = mesh - joint[:, tf.newaxis, tf.newaxis, tf.newaxis]
            pose_map = pose_map / shape[:, tf.newaxis, tf.newaxis, tf.newaxis] * image_size
            pose_map = tf.exp(-tf.reduce_sum(pose_map ** 2, axis=0) / (2 * sigma ** 2))
            pose_maps.append(pose_map)
        pose_map = tf.stack(pose_maps, axis=-1)
        if params['2d_3d_pose']:
            pose_map = tf.reduce_max(pose_map, axis=2, keepdims=True)
            pose_map = tf.tile(pose_map, [1, 1, params['depth'], 1])
        pose_mapss.append(pose_map)
    return tf.stack(pose_mapss, axis=0)


def resnet_encoder(img_batch):
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    with tf.contrib.slim.arg_scope(resnet_arg_scope()):
        x, network = resnet_v2(img_batch, blocks, output_stride=4, scope='resnet_v2_50', spatial_squeeze=False,
                               reuse=tf.AUTO_REUSE, checkpoint_backward_compatibility=True)
        # checkpoint_backward_compatibility must be set to True if the provided generator checkpoints should work
        # if a new model should be trained, it can be set to False

    return network['GAN/Generator/resnet_v2_50/block2']


def pose_encoder(pose_batch, features=32):
    x = tf.layers.conv3d(pose_batch, features, kernel_size=3, padding='SAME')
    x = residual_unit_3d(x)
    x = residual_unit_3d(x)
    x = residual_unit_3d(x)
    return x


def resnet_decoder(x):
    x = bottleneck_transposed(x, depth=256 * 4, depth_bottleneck=256, stride=1)
    x = bottleneck_transposed(x, depth=256 * 4, depth_bottleneck=256, stride=1)
    x = bottleneck_transposed(x, depth=128 * 4, depth_bottleneck=128, stride=2)
    x = bottleneck_transposed(x, depth=128 * 4, depth_bottleneck=128, stride=1)
    x = bottleneck_transposed(x, depth=128 * 4, depth_bottleneck=128, stride=1)
    x = bottleneck_transposed(x, depth=64 * 4, depth_bottleneck=64, stride=2)
    return x


def background_inpainter(img_batch, masks_batch):
    # adapted from https://github.com/MathiasGruber/PConv-Keras
    if params['2d_3d_warp']:
        bg_mask = masks_batch
    else:
        bg_mask = tf.reduce_max(masks_batch, axis=3)
    bg_mask = bg_mask[:, :-1, :-1]
    bg_mask = tf.image.resize_images(bg_mask, (params['image_size'], params['image_size']),
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    bg_mask = tf.reduce_max(bg_mask, axis=3)
    bg_mask = 1 - bg_mask
    bg_mask = tf.pad(bg_mask, [[0, 0]] + [[0, 1]] * (len(bg_mask.shape) - 1))
    img_batch = img_batch * bg_mask[..., tf.newaxis]

    def pconv(imgs, masks, filters, kernel_size, strides=1):
        with tf.variable_scope(None, default_name='pconv'):
            ps = int((kernel_size - 1) / 2)
            imgs = tf.pad(imgs, [[0, 0], [ps, ps], [ps, ps], [0, 0]])
            masks = tf.pad(masks, [[0, 0], [ps, ps], [ps, ps], [0, 0]])

            input_dim = int(imgs.shape[-1])
            kernel = tf.get_variable('kernel', shape=(kernel_size, kernel_size, input_dim, filters),
                                     initializer=tf.initializers.glorot_uniform())
            imgs = tf.nn.conv2d(imgs, kernel, strides=(1, strides, strides, 1), padding='VALID')

            mask_kernel = tf.ones((kernel_size, kernel_size, input_dim, filters), dtype=tf.float32,
                                  name='mask_kernel')
            masks = tf.nn.conv2d(masks, mask_kernel, strides=(1, strides, strides, 1), padding='VALID')

            mask_ratio = kernel_size ** 2 / (masks + 1e-8)

            masks = tf.clip_by_value(masks, 0, 1)

            mask_ratio = mask_ratio * masks

            imgs = imgs * mask_ratio

            bias = tf.get_variable('bias', shape=(filters,), initializer=tf.initializers.zeros())
            imgs = tf.nn.bias_add(imgs, bias)

        return [imgs, masks]

    bg_mask = bg_mask[..., tf.newaxis]
    bg_mask = tf.tile(bg_mask, [1, 1, 1, img_batch.shape[-1]])

    img_batch = img_batch / 2 + .5
    img_batch = img_batch + (1 - bg_mask)

    def encoder_layer(img_in, mask_in, filters, kernel_size, gn=True):
        img_in = mask_in * img_in
        conv, mask = pconv(img_in, mask_in, filters, kernel_size, strides=2)
        if gn:
            conv = group_norm(conv)
        conv = tf.nn.relu(conv)
        return conv, mask

    e_conv1, e_mask1 = encoder_layer(img_batch, bg_mask, 64, 7, gn=False)
    e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
    e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
    e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
    e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
    e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
    e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3)

    def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, gn=True, activation=tf.nn.leaky_relu):
        img_in = img_in * mask_in
        up_img = tf.image.resize_images(img_in, (img_in.shape[1] * 2 - 1, img_in.shape[2] * 2 - 1),
                                        method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        up_mask = tf.image.resize_images(mask_in, (mask_in.shape[1] * 2 - 1, mask_in.shape[2] * 2 - 1),
                                         method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

        concat_img = tf.concat([e_conv, up_img], axis=3)
        concat_mask = tf.concat([e_mask, up_mask], axis=3)
        conv, mask = pconv(concat_img, concat_mask, filters, kernel_size)
        if gn:
            conv = group_norm(conv)
        if activation:
            conv = activation(conv)
        return conv, mask

    d_conv10, d_mask10 = decoder_layer(e_conv7, e_mask7, e_conv6, e_mask6, 512, 3)
    d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3)
    d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3)
    d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
    d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
    d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
    x, _ = decoder_layer(d_conv15, d_mask15, img_batch, bg_mask, 3, 3, gn=False)

    x = tf.nn.tanh(x)
    return x


def warp3d_generator(img_batch, masks_batch, params_batch, pose_batch):
    if params['volume_size'] != 64:
        raise ValueError('parameter volume_size must be 64 instead of', params['volume_size'])
    n, h, w, c = img_batch.shape
    noise = tf.random.normal(img_batch[..., :1].shape)
    img_batch = tf.concat([img_batch, noise], axis=-1)

    net = {}
    x = resnet_encoder(img_batch)
    p = pose_encoder(pose_batch)
    b = background_inpainter(img_batch, masks_batch)

    # convert image stream to 3D
    x = group_norm(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, (params['depth'] + 1) * params['residual_channels'], kernel_size=3, padding='SAME')

    x = tf.reshape(x, (n, x.shape[1], x.shape[2], params['depth'] + 1, params['residual_channels']))

    net['first_3d'] = x

    # network center
    for i in range(params['before_count']):
        x = residual_unit_3d(x)

    net['volume'] = x

    net['masks'] = masks_batch

    if params['2d_3d_warp']:
        x, subnet = warp_2d_3d(x, masks_batch, params_batch)
    else:
        x, subnet = warp_3d(x, masks_batch, params_batch)

    net = {**net, **subnet}

    net['warped'] = x
    x = tf.concat([x, p], axis=-1)

    for i in range(params['after_count']):
        x = residual_unit_3d(x)

    net['last_3d'] = x

    # convert to 2D
    x = tf.reshape(x, (n, x.shape[1], x.shape[2], -1))

    x = resnet_decoder(x)

    x = group_norm(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 4, kernel_size=3, padding='SAME')
    mask = tf.nn.sigmoid(x[..., 3])[..., tf.newaxis]
    x = tf.nn.tanh(x[..., :3])
    net['foreground'] = x
    x = mask * x + (1 - mask) * b
    net['background'] = b
    net['foreground_mask'] = mask
    return x, net


def resnet_discriminator(x):
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    with tf.contrib.slim.arg_scope(resnet_arg_scope()):
        x, network = resnet_v2(x, blocks, output_stride=None, scope='resnet_v2_50')
    x = network['GAN/Discriminator/resnet_v2_50/block3']
    x = group_norm(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 1, kernel_size=3, padding='SAME')
    x = tf.reduce_mean(x)
    return x


def generator(x):
    img_batch, masks_batch, params_batch, input_pose_batch, target_pose_batch = x

    img_batch = extend_spatial_sizes(img_batch)

    masks_batch = extend_spatial_sizes(masks_batch)
    target_pose_batch = tf_pose_map_3d(target_pose_batch,
                                       [params['volume_size'], params['volume_size'], params['depth']])
    target_pose_batch = extend_spatial_sizes(target_pose_batch)

    x, net = warp3d_generator(img_batch, masks_batch, params_batch, target_pose_batch)
    return reduce_spatial_sizes(x), net


def discriminator(generated, x):
    if isinstance(generated, tuple):
        generated = generated[0]
    N, H, W, C = generated.shape
    generated = extend_spatial_sizes(generated)
    img_batch, _, _, _, pose_batch = x
    img_batch = extend_spatial_sizes(img_batch)
    pose_batch = tf_pose_map_3d(pose_batch, [params['image_size'], params['image_size'], params['depth']])
    pose_batch = tf.reshape(pose_batch,
                            (N, params['image_size'], params['image_size'], -1))
    pose_batch = extend_spatial_sizes(pose_batch)
    x = tf.concat([img_batch, generated, pose_batch], axis=-1)

    return reduce_spatial_sizes(resnet_discriminator(x))
