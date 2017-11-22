import numpy as np
import cv2

def remove_car(img):
    img[300:486, 300:398] = 0
    return img


def resize(img, size=(300,300)):
    img = img[0:696,:]
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    return img

def norm(img):
    img = img / 6.
    return img

def preprocess(img):
    img = img.copy()
    img = remove_car(img)
    img = resize(img)
    img = norm(img)
    return img


