import os
import sys
import copy
import numpy as np
import random
from pathlib import Path
from geo_ai_metrics.annotation import AnnotatedImage, AnnotatedObject, Annotation
from geo_ai_metrics.annotation import read_yolo, write_coco


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    labels_dir = str(data_dir / 'aerial' / 'labels')
    
    gt_annot = read_yolo(labels_dir, (2133, 2133), None)
    print(gt_annot)
    write_coco(gt_annot, str(data_dir / 'aerial' / 'annotations' / 'gt.json'), '.png')

    pred_annot = create_noisy_annot(gt_annot)
    print(pred_annot)
    write_coco(pred_annot, str(data_dir / 'aerial' / 'annotations' / 'pred.json'), '.png')


def create_noisy_annot(annot: Annotation) -> Annotation:
    for name in annot.images:
        image = annot.images[name]
        for an in image.annotations:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            
            bbox = an.bbox
            bbox[0], bbox[2] = int(bbox[0] + dx), int(bbox[2] + dx)
            bbox[1], bbox[3] = int(bbox[1] + dy), int(bbox[3] + dy)
            
            s = an.segmentation
            if len(s) == 0:
                continue
            
            s = np.array(s).reshape(-1, 1, 2)
            s[..., 0] += dx
            s[..., 1] += dy
            an.segmentation = s.reshape(1, -1).astype('int32').tolist()
    
    return annot


def change_annotation(annot: Annotation, new_classes: list):
    classes = annot.categories
    
    conformity = {}
    for i in range(len(classes)):
        if classes[i] in new_classes:
            conformity[i] = new_classes.index(classes[i])
    
    new_annot = Annotation(categories=new_classes, images={})
    images = annot.images
    
    for name in images:
        new_bboxes = []
        
        for bbox in images[name].annotations:
            bbox = copy.copy(bbox)
            if int(bbox.category_id) not in conformity:
                continue
            
            bbox.category_id = conformity[bbox.category_id]
            new_bboxes.append(bbox)
            
        new_image = AnnotatedImage(width=images[name].width, height=images[name].height, annotations=new_bboxes)
        new_annot.images[name] = new_image
        
    return new_annot


if __name__ == '__main__':
    main()

