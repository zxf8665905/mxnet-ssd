import numpy as np
from shapely.geometry import Polygon


def box2d_area(box):
    return Polygon(zip(*box.T)).area


def box2d_iou(box_a, box_b):
    # oriented XY iou
    xy_poly_a = Polygon(zip(*box_a.T))
    xy_poly_b = Polygon(zip(*box_b.T))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area

    return xy_intersection / (xy_poly_a.area + xy_poly_b.area - xy_intersection)


def boxes_nms(boxes, scores, thresh):
    """Python NMS include box orientation"""

    areas = np.array([box2d_area(box) for box in boxes])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter = np.array([box2d_iou(boxes[i], boxes[j]) for j in order[1:]])
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == '__main__':
    import time

    box_a = np.array([[0, 10],
                      [0, 10],
                      [20, 10],
                      [20, 0]])
    box_b = np.array([[10, 10],
                      [10, 10],
                      [30, 10],
                      [30, 0]])
    t0 = time.time()
    iou = box2d_iou(box_a, box_b)
    t1 = time.time()
    print('iou: {} time: {}'.format(iou, t1 - t0))

    """
    test py_cpu_nms
    """
    import time

    dets = np.array([
        [[0.86886532, 0.70776505],
         [0.73530097, 0.85193947],
         [0.13113468, 0.29223495],
         [0.26469903, 0.14806053]],

        [[0.44045055, 0.68556669],
         [0.39230181, 0.72659735],
         [-0.06545055, 0.18943331],
         [-0.01730181, 0.14840265]],

        [[0.40937683, 0.51612011],
         [0.03756054, 0.61896242],
         [-0.03437683, 0.35887989],
         [0.33743946, 0.25603758]],

        [[0.30611549, 0.59892498],
         [0.23087375, 0.60549543],
         [0.22329631, 0.51872208],
         [0.29853804, 0.51215163]],

        [[0.34588767, 0.52961504],
         [0.22582201, 0.55475421],
         [0.18352412, 0.35273791],
         [0.30358979, 0.32759873]],

        [[0.40830199, 0.3840931],
         [0.03849015, 0.31384985],
         [0.00346272, 0.49825985],
         [0.37327456, 0.5685031]],

        [[0.91190979, 0.32549156],
         [0.78446095, 0.29038777],
         [0.71309021, 0.54950844],
         [0.84053905, 0.58461223]]])
    scores = np.array([ 1., 0.99989438 , 0.99837935, 0.99059266, 0.98407125, 0.89159685, 0.61995888])
    t0 = time.time()
    keep = boxes_nms(dets, scores, 0.1)
    t1 = time.time()
    print('time:{} keep: {}'.format(t1 - t0,keep))
