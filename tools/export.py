from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
sys.path.append(os.getcwd())
from networks.model_test import model_test
from utils.config import cfg, cfg_from_file
from networks.test import test_ctpn
from utils.timer import Timer
from utils.text_connector.detectors import TextDetector
from utils.text_connector.text_connect_cfg import Config as TextLineCfg


if __name__ == '__main__':
    cfg_from_file('tools/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = model_test()
    # load model
    print(('Loading network {:s}... '.format("model_test")), end=' \n')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    print(' done.')

    print('all nodes are:\n')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    node_names = [node.name for node in input_graph_def.node]
    for x in node_names:
        print(x)


    '''
    output_node_names = 'rois/Reshape'
#    output_node_names = 'Reshape_2,rpn_bbox_pred/Reshape_1'
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(','))
    output_graph = 'net.pb'
    with tf.gfile.FastGFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    '''

    sess.close()

