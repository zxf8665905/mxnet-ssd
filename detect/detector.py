from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import cv2
import time
from tools.box import boxes_nms
from tools.data import box_encode, box_decode
from collections import namedtuple
from tools.data import draw_box

nms_iou_threhold = 0.1
nms_topk = 100



def multibox_detection(cls_prob, attrs_preds, anchors):
    # cls_prob.shape [batch_size, cls_num]
    batch_size = cls_prob.shape[0]
    o_cls_id = []
    o_probs = []
    o_boxes = []
    for i in range(batch_size):
        single_batch_o_boxes =[]
        single_batch_cls_prob, single_batch_attrs_preds, single_batch_anchors = \
            cls_prob[i].T, attrs_preds[i].asnumpy(), anchors[0].asnumpy()
        cls_id = mx.nd.argmax(single_batch_cls_prob, axis=1).asnumpy().astype(np.int)
        keep = cls_id > 0
        cls_id = cls_id[keep]
        probs = np.array([probs[cls_id[i]] for i,probs in enumerate(single_batch_cls_prob.asnumpy()[keep])])
        single_batch_anchors = single_batch_anchors[keep]
        single_batch_attrs_preds = single_batch_attrs_preds[keep]

        for j in range(len(single_batch_anchors)):
            w = single_batch_attrs_preds[j,0]
            h = single_batch_attrs_preds[j,1]
            theta = single_batch_attrs_preds[j, 1]
            trans = (single_batch_anchors[j][0:2] + single_batch_anchors[j][2:4])/2
            box = box_encode(trans=trans, w=w, h=h, theta=theta)
            single_batch_o_boxes.append(box)
            # convert to nms_box for nms
            nms_box = np.array([np.min(box[:,0]), np.min(box[:,0]), np.min(box[:,0]), np.min(box[:,0]),probs[j]])
        keep_ids = boxes_nms(np.array(single_batch_o_boxes),probs, nms_iou_threhold)

        o_cls_id.append(cls_id[keep_ids])
        o_probs.append(probs[keep_ids])
        o_boxes.append(np.array(single_batch_o_boxes)[keep_ids])
    return np.array(o_cls_id)-1, np.array(o_probs), np.array(o_boxes)



class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        result = []
        detections = []
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        for pred, _, _ in self.mod.iter_predict(det_iter):
            o_cls_id, o_probs, o_boxes = multibox_detection(pred[1], pred[2], pred[3])
            detections.append([o_cls_id, o_probs, o_boxes])
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        return detections

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        import random

        # plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        o_cls_id, o_probs, o_boxes = dets
        for i in range(o_cls_id.shape[0]):
            cls_id = int(o_cls_id[i])
            score = o_probs[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            box = np.zeros_like(o_boxes[i])
            box = o_boxes[i]*height
            draw_box(img=img, box=box,color=colors[cls_id])
        output_path = './output/{}.png'.format(int(time.time()*1000))
        print('write: {}'.format(output_path))
        cv2.imwrite(output_path, img)

    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        import cv2
        from tools.image_processing import preprocess
        Batch = namedtuple('Batch', ['data'])
        for k, im_path in enumerate(im_list):
            img = preprocess(cv2.imread(im_list[k]))
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            data_batch = Batch(data=[mx.nd.transpose(mx.nd.array([img]), axes=(0, 3, 2, 1))])
            self.mod.forward(data_batch, is_train=False)
            out = self.mod.get_outputs()
            img = (img*255).astype(np.uint8)
            o_cls_id, o_probs, o_boxes = multibox_detection(out[1], out[2], out[3])
            self.visualize_detection(img, [o_cls_id[k], o_probs[k], o_boxes[k]], classes, thresh)



if __name__ == '__main__':
    # test multibox
    import pickle
    out = pickle.load(open('../predict.dump', 'rb'))
    o_cls_id, o_probs, o_boxes =multibox_detection(out[0][1], out[0][2], out[0][3])
    print('o_cls_id:{}'.format(o_cls_id))
    print('o_probs:{}'.format(o_probs))
    print('o_boxes:{}'.format(o_boxes))