import sys
import pytest
import numpy as np
from geo_ai_metrics.utils.utils import mask_iou


def test_mask_iou_empty():
    
    iou = mask_iou(np.zeros((0, 4)), np.zeros((0, 4)))
    assert iou.shape == (0, 0)
    
    iou = mask_iou(np.zeros((1, 4)), np.zeros((0, 4)))
    assert iou.shape == (1, 0)
    
    iou = mask_iou(np.zeros((0, 4)), np.zeros((1, 4)))
    assert iou.shape == (0, 1)

def test_mask_iou_1x1():
    mask1 = np.array([[1, 1, 0, 0]])
    mask2 = np.array([[1, 1, 1, 1]])
    iou = mask_iou(mask1, mask2)

    assert iou.shape == (1, 1)
    assert np.isclose(iou, np.array([[0.5]])).min()


def test_mask_iou_2x3():
    mask1 = np.array([[1, 0], [0, 1]])
    mask2 = np.array([[1, 0], [0, 1], [0, 0]])
    iou = mask_iou(mask1, mask2)

    assert iou.shape == (2, 3)
    assert np.isclose(iou, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])).min()
    

if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))


