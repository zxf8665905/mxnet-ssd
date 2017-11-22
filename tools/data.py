import cv2
import json
import glob
from image_processing import preprocess

def draw_box(img, box,color=(255,0,255), thickness=2):
    assert box.shape == (4,2)
    for i in range(4):
        cv2.line(img, box[i], box[i+1], color, thickness, cv2.LINE_AA)

def load_label(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    data_paths = glob.glob('../data/demo/*.png')
    datas = [cv2.imread(path) for path in data_paths]
    labels = load_label('../data/demo/2_frame_label.json')

    for i,data in enumerate(datas):
        data_pre = preprocess(data)
        n_objs = len(labels['frames'][i])
