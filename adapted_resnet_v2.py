# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# THIS CODE WAS HEAVILY ADAPTED AND DOES NOT CORRESPOND TO THE ORIGINAL TENSORFLOW IMPLEMENTATION

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

import adapted_resnet_utils

slim = contrib_slim
resnet_arg_scope = adapted_resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.group_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = adapted_resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = adapted_resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def bottleneck_transposed(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    assert stride in (1, 2)

    with tf.variable_scope(scope, 'bottleneck_v2_transposed', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.group_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            if stride > 1:
                raise Exception('We cannot do spatial expansion by subsampling!')
            shortcut = adapted_resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                preact, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None,
                scope='shortcut')
            if stride > 1:
                shortcut = tf.image.resize_images(shortcut,
                                                  (shortcut.shape[1] * stride - 1, shortcut.shape[2] * stride - 1),
                                                  method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')

        if stride > 1:
            residual = tf.image.resize_images(residual,
                                              (residual.shape[1] * stride - 1, residual.shape[2] * stride - 1),
                                              method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        residual = slim.conv2d(residual, depth_bottleneck, 3, stride=1, scope='conv2')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')
        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=False,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None,
              checkpoint_backward_compatibility=False):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             adapted_resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.group_norm]):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        if checkpoint_backward_compatibility:
                            res = 0
                            res += adapted_resnet_utils.conv2d_same(net[..., :3], 64, 7, stride=2, scope='conv1')
                            if 1 * 3 > net.shape[-1]:
                                print(True)
                                exit()
                                res += adapted_resnet_utils.conv2d_same(net[..., 3:], 64, 7, stride=2, scope='conv1p')
                        else:
                            res = adapted_resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                        net = res
                    net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')
                net = adapted_resnet_utils.stack_blocks_dense(net, blocks, output_stride)
                net = slim.group_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v2_block(scope, base_depth, num_units, stride):
    return adapted_resnet_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])
