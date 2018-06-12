
import tensorflow as tf
from .network import Network
from utils.config import cfg


class model_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='im_info')  # 高, 宽, 放缩比
#        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        # ANCHOR_SCALES = [16]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))

        (self.feed('pool4').conv(1, 1, 256, 1, 1, name='conv1x1_1'))
        (self.feed('conv1x1_1').conv(1, 5, 256, 1, 1, name='incep_1'))

        (self.feed('pool4').conv(1, 1, 256, 1, 1, name='conv1x1_2'))
        (self.feed('conv1x1_2').conv(5, 1, 256, 1, 1, name='incep_2'))
        (self.feed('conv5_3', 'incep_1', 'incep_2').incept(name='inception'))


        #========= RPN ============
        (self.feed('inception').conv(3, 3, 512, 1, 1, name='rpn_conv'))

        (self.feed('rpn_conv').rpn_fc(512, len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        (self.feed('rpn_conv').rpn_fc(512, len(anchor_scales) * 10 * 2, name='rpn_cls_score'))

        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 10 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))
