import os
import argparse
from pathlib import Path
from geo_ai_metrics.annotation import Annotation, read_coco_geoai_360
from geo_ai_metrics.metrics import Localization360Accuracy


def main():
    
    args = parse_args()

    gt_annot_path = args.gt_annot_path
    pred_annot_path = args.pred_annot_path
    images_dir = args.images_dir
    save_dir = args.save_dir
    
    gt_annot = read_coco_geoai_360(gt_annot_path)
    pred_annot = read_coco_geoai_360(pred_annot_path)
    
    metric = Localization360Accuracy(mode='box')
    mean, std = metric(gt_annot, pred_annot, save=True, save_dir=save_dir, images_dir=images_dir)
    
    print("Mean:", mean)
    print("Std:", std)
    

def parse_args():
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / 'geo_ai_metrics' / 'data'
    
    gt_annot_path = str(data_dir / '360' / 'annotations' / 'gt.json')
    pred_annot_path = str(data_dir / '360' / 'annotations' / 'pred.json')
    images_dir = str(data_dir / '360' / 'images')
    save_dir = str(project_dir / 'outs' / 'localization360')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_annot_path', type=str, default=gt_annot_path)
    parser.add_argument('--pred_annot_path', type=str, default=pred_annot_path)
    parser.add_argument('--images_dir', type=str, default=images_dir)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
