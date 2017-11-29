import cv2
import json
import glob
from image_processing import preprocess
import numpy as np
from collections import namedtuple
import math

def draw_box(img, box,color=(255,0,255), thickness=2):
    box = box.astype(np.int32)
    assert box.shape == (4,2)
    for i in range(3):
        cv2.line(img, tuple(box[i]), tuple(box[i+1]), color, thickness)
    cv2.line(img, tuple(box[3]), tuple(box[0]), color, thickness)


def load_label(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def box_decode(box):
    """
    attr: [height, width, orientation]
    """
    box = box.copy()/300.
    bbox = np.array([np.min(box, axis=0), np.max(box, axis=0)])
    height = (np.sum((box[0]-box[3])**2)**0.5 + np.sum((box[1]-box[2])**2)**0.5)/2
    width = (np.sum((box[0] - box[1]) ** 2) ** 0.5 + np.sum((box[3] - box[2]) ** 2) ** 0.5) / 2
    mid0, mid1 = (box[0]+box[1])/2, (box[3]+box[2])/2
    rotation = (math.atan2((mid0-mid1)[1], (mid0-mid1)[0]) + math.pi/2)/ (math.pi*2)

    label = namedtuple('label',['bbox','attr'])
    label.bbox = bbox
    label.attr = np.array([height, width, rotation])
    return label


def rotate_box(box, theta):
    # print('func: box {} theta {}'.format(box,theta))
    box = box.copy()
    rotMat = np.array([ \
        [np.cos(theta), -np.sin(theta)], \
        [np.sin(theta), np.cos(theta)]])
    return np.matmul(rotMat, box.T).T


def box_encode(trans, w, h, theta):
    box = np.array([[-w / 2, -h / 2],
                    [w / 2, -h / 2],
                    [w / 2, h / 2],
                    [-w / 2, h / 2]])
    box_rotated = rotate_box(box, theta)
    box_rotated = box_rotated + trans
    return box_rotated

if __name__ == '__main__':
    data_paths = glob.glob('../data/demo/*.png')
    datas = [cv2.imread(path) for path in data_paths]
    labels = load_label('../data/demo/2_frame_label.json')
    label_color={}
    label_color['parking_space'] = (25,125,23)
    label_color['road'] = (13,13,80)

    for n_frame,data in enumerate(datas):
        data_pre = preprocess(data)
        vis_img = (data_pre.copy() * 255).astype(np.uint8)
        for n_obj,obj in enumerate(labels['frames'][n_frame]['objects']):
            box = np.array(obj['point']).reshape((4,2))
            vis_img = draw_box(vis_img, box, color=label_color[obj['type']])
            label = box_decode(box)
        o_path = '../data/demo/output/{}.png'.format(n_frame)
        cv2.imwrite(o_path, vis_img)
        print('writed:{}'.format(o_path))
