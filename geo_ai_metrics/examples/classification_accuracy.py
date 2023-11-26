import os
import argparse
from pathlib import Path
from geo_ai_metrics.annotation import Annotation, read_coco
from geo_ai_metrics.metrics import ClassificationAccuracy


def main():
    
    args = parse_args()

    gt_annot_path = args.gt_annot_path
    pred_annot_path = args.pred_annot_path
    images_dir = args.images_dir
    save_dir = args.save_dir
    
    gt_annot = read_coco(gt_annot_path)
    pred_annot = read_coco(pred_annot_path)
    
    metric = ClassificationAccuracy(mode='box')    
    acc = metric(gt_annot, pred_annot, save=True, save_dir=save_dir, images_dir=images_dir)
    
    print("Accuracy:", acc)


def parse_args():
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / 'geo_ai_metrics' / 'data'
    
    gt_annot_path = str(data_dir / 'aerial' / 'annotations' / 'gt.json')
    pred_annot_path = str(data_dir / 'aerial' / 'annotations' / 'pred.json')
    images_dir = str(data_dir / 'aerial' / 'images')
    save_dir = str(project_dir / 'outs' / 'classification')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_annot_path', type=str, default=gt_annot_path)
    parser.add_argument('--pred_annot_path', type=str, default=pred_annot_path)
    parser.add_argument('--images_dir', type=str, default=images_dir)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
