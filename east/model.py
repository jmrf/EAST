import os
import cv2
import time
import logging
import datetime
import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import east.lanms as lanms
from east.nets import resnet_v1
from east.icdar import restore_rectangle
from east.utils import resize_image, sort_poly


logger = logging.getLogger(__name__)


def unpool(inputs):
    return tf.image.resize_bilinear(
        inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + \
        tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_diceloss', loss)
    return loss


def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classificationloss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)

    # scale classification loss to match the iou loss part
    classificationloss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo,
                                                    num_or_size_splits=5,
                                                    axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo,
                                                              num_or_size_splits=5,
                                                              axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)

    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect

    L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)

    tf.summary.scalar('geometry_AABB',
                      tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta',
                      tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classificationloss


def def_model(images, weight_decay=1e-5, is_training=True, text_scale=512):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(
            images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                logger.debug('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(
                        tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                logger.debug('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape,
                                                                i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(
                g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(
                g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid,
                                     normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry


def _detect(score_map, geo_map, timer, score_map_thresh=0.8,
            box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)

    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4,
                                          geo_map[xy_text[:, 0],
                                                  xy_text[:, 1], :])  # N*4*2

    logger.debug('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start

    # nms part
    start = time.time()
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is
    # different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape(
            (-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


class EASTPredictor():

    def __init__(self):
        # intialize the tensorflow session
        self.graph = tf.Graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # build the model
        with self.graph.as_default():
            self._build_model()

    def _build_model(self):
        # input placeholders and training vars
        self.input_images = tf.placeholder(tf.float32,
                                           shape=[None, None, None, 3],
                                           name='input_images')

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # define the model and get the final ops
        self.f_score, self.f_geometry = def_model(self.input_images,
                                                  is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        self.saver = tf.train.Saver(variable_averages.variables_to_restore())

    def load(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileExistsError("checkpoint path not found")

        # checkpoint restoration
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path,
                                  os.path.basename(ckpt_state.model_checkpoint_path))
        logger.info('Restoring checkpoint from {}'.format(model_path))

        self.saver.restore(self.sess, model_path)

    def predict(self, img):

        # collect some stats of the run
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(im_resized.shape[1],
                                                  im_resized.shape[0])

        # model detection run
        start = time.time()
        score, geometry = self.sess.run([self.f_score, self.f_geometry],
                                        feed_dict={self.input_images: [im_resized[:, :, ::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = _detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, '
                    'restore {:.0f}ms, '
                    'nms {:.0f}ms'.format(timer['net'] * 1000,
                                          timer['restore'] * 1000,
                                          timer['nms'] * 1000))

        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        # Overall run time
        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:

            for box, score in zip(boxes, scores):

                box = sort_poly(box.astype(np.int32))
                if (np.linalg.norm(box[0] - box[1]) < 5 or
                        np.linalg.norm(box[3] - box[0]) < 5):
                    continue

                # detections coordinates as orderdict & scores
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten()))
                )
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        return ret
