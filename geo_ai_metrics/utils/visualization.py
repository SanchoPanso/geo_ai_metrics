import cv2
import numpy as np


def draw_segments(img, segmentation, thickness, color):
    mask = np.zeros(img.shape[:2], dtype='uint8')
    
    for segment in segmentation:
        segment = np.array(segment)
        segment = segment.reshape(-1, 1, 2).astype('int32')
        cv2.polylines(img, [segment], True, color, thickness)
        cv2.fillPoly(mask, [segment], 1)
    
    color_layer = np.ones_like(img)
    for i in range(3):
        color_layer[..., i] *= color[i]
    
    mask = 0.5 * mask[..., np.newaxis].astype('float32')
    color_layer = color_layer.astype('float32')
    img = mask * color_layer + (1 - mask) * img
    img = img.astype('uint8')
    
    # cv2.imshow('img', img)
    # cv2.waitKey()
    
    return img
    
