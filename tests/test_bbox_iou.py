import sys
import pytest
import torch
from geo_ai_metrics.utils.utils import bbox_iou


def test_bbox_iou():
    bbox1 = torch.tensor([0, 0, 1, 1])
    bbox2 = torch.tensor([0, 0, 1, 1])
    iou = bbox_iou(bbox1, bbox2)

    assert iou.shape == (1,)
    assert torch.isclose(iou, torch.tensor([1.0]))


def test_bbox_iou_xyxy():
    bbox_xyxy1 = torch.tensor([10, 10, 20, 20])
    bbox_xyxy2 = torch.tensor([15, 15, 20, 20])
    iou = bbox_iou(bbox_xyxy1, bbox_xyxy2, xywh=False)

    assert iou.shape == (1,)
    assert torch.isclose(iou, torch.tensor([0.25]))



if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))


