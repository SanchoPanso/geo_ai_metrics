import sys
import pytest
import numpy as np
from geo_ai_metrics.utils.utils import box_iou


def test_box_iou_empty():
    
    iou = box_iou(np.zeros((0, 4)), np.zeros((0, 4)))
    assert iou.shape == (0, 0)
    
    iou = box_iou(np.zeros((1, 4)), np.zeros((0, 4)))
    assert iou.shape == (1, 0)
    
    iou = box_iou(np.zeros((0, 4)), np.zeros((1, 4)))
    assert iou.shape == (0, 1)
    
    

def test_box_iou_1x1():
    bbox1 = np.array([[0, 0, 1, 1]])
    bbox2 = np.array([[0, 0, 1, 1]])
    iou = box_iou(bbox1, bbox2)

    assert iou.shape == (1, 1)
    assert np.isclose(iou, np.array([[1.0]])).min()


def test_box_iou_2x3():
    bbox1 = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    bbox2 = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [0, 0, 2, 2]])
    iou = box_iou(bbox1, bbox2)

    assert iou.shape == (2, 3)
    assert np.isclose(iou, np.array([[1.0, 0.0, 0.25], [0.0, 1.0, 0.25]])).min()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))


