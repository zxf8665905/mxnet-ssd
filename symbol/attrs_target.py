"""
attrs_label: [height, width, orientation]
"""
import numpy as np
import mxnet as mx
import pickle
import mxnet as mx
from collections import namedtuple
import numpy as np
import cv2
import utils.bbox as bbox


class AttrsTarget(mx.operator.CustomOp):
    def __init__(self, threshold = 0.5):
        self.threshold = threshold

    def forward(self, is_train, req, in_data, out_data, aux):
        loc_cls_label, attrs_label, anchors = in_data[0].asnumpy(), in_data[1].asnumpy(), in_data[2].asnumpy()
        batch_size = loc_cls_label.shape[0]
        batch_attrs_target = []
        batch_attrs_target_mask = []
        for i in range(batch_size):
            ids = np.where(loc_cls_label[i, :, 5] != -1.)
            loc_cls_label_valid = loc_cls_label[i, ids, :]
            attrs_label_valid = attrs_label[i, ids, :]
            loc_label_valid = loc_cls_label_valid[..., 1:6][0]
            attrs_label_valid = attrs_label_valid[0]
            overlaps = bbox.bbox_overlaps(np.ascontiguousarray(anchors[0], dtype=np.float),
                                          np.ascontiguousarray(loc_label_valid, dtype=np.float))
            obj_ids = overlaps.argmax(axis=1)  # which object is max overlap
            attrs_target = attrs_label_valid[obj_ids]
            batch_attrs_target.append(attrs_target.flatten())
            # mask
            obj_overlaps = overlaps.take(obj_ids)
            obj_overlaps_mask = (obj_overlaps > self.threshold).astype(np.float32)
            attrs_target_mask = np.array([obj_overlaps_mask, obj_overlaps_mask, obj_overlaps_mask]).T.flatten()
            batch_attrs_target_mask.append(attrs_target_mask)
        self.assign(out_data[0], req[0], mx.nd.array(batch_attrs_target))
        self.assign(out_data[1], req[1], mx.nd.array(batch_attrs_target_mask))


@mx.operator.register("attrs_target_op")
class AttrsTargetProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AttrsTargetProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['loc_cls_label', 'attrs_label', 'anchors']

    def list_outputs(self):
        return ['batch_attrs_target', 'batch_attrs_target_mask']

    def infer_shape(self, in_shape):
        output_shape = [[in_shape[0][0], in_shape[2][1] * in_shape[1][2]],
                        [in_shape[0][0], in_shape[2][1] * in_shape[1][2]]]
        return [in_shape[0], in_shape[1], in_shape[2]], output_shape, []

    def infer_type(self, in_type):
        return in_type, [in_type[0], in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return AttrsTarget()


if __name__ == '__main__':

    # test case
    loc_cls_label = mx.sym.Variable(name='loc_cls_label')
    attrs_label = mx.sym.Variable(name='attrs_label')
    anchors = mx.sym.Variable(name='anchors')

    loc_preds, orien_preds, cls_preds, anchor_boxes = pickle.load(open('../multibox_layer_output.dump', 'rb'))
    batch_attrs_target, batch_attrs_target_mask = mx.symbol.Custom(loc_cls_label=loc_cls_label, attrs_label=attrs_label,
                              anchors=anchor_boxes, op_type='attrs_target_op')

    batch_size = 4
    loc_cls_label = np.ones((batch_size, 58, 6), dtype=np.float32) * -1.0
    loc_cls_label[:, 0:2, :] = np.array([[1, 0, 0, 0.25, 0.25, 0], [1, 0.25, 0.25, 0.5, 0.5, 0]])
    loc_cls_label, loc_cls_label.shape

    attrs_label = np.ones((batch_size, 58, 3), dtype=np.float32) * -1.0
    attrs_label[:, 0:2, 0] = np.array([0.1, 0.2])
    attrs_label[:, 0:2, 1] = np.array([0.3, 0.3])
    attrs_label[:, 0:2, 2] = np.array([0.4, 0.42])

    attrs_label, attrs_label.shape

    mod = mx.mod.Module(mx.symbol.Group([batch_attrs_target, batch_attrs_target_mask ]),
                        context=mx.gpu(3), data_names=['data'], label_names=['loc_cls_label', 'attrs_label'])
    mod.bind(data_shapes=[('data', (batch_size, 3, 300, 300))],
             label_shapes = [('loc_cls_label', (batch_size, 58, 6)),
                             ('attrs_label', (batch_size, 58, 3))])
    mod.init_params()
    Batch = namedtuple('Batch', ['data', 'label'])
    mod.forward(Batch(data=[mx.nd.ones((batch_size, 3, 300, 300))],
                      label= [mx.nd.array(loc_cls_label), mx.nd.array(attrs_label)]), is_train=True)
    print(mod.get_outputs())

    # loc_preds, orien_preds, cls_preds, anchor_boxes = pickle.load(open('../multibox_layer_output.dump', 'rb'))
    # mod = mx.mod.Module(anchor_boxes, data_names=['data'], label_names=[])
    # mod.bind(data_shapes=[('data', (4, 3, 300, 300))])
    # mod.init_params()
    # Batch = namedtuple('Batch', ['data'])
    # data = [mx.nd.ones((4, 3, 300, 300))]
    # mod.forward(Batch(data))
