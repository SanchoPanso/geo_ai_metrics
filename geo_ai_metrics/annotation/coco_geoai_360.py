import json
import os
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage, AnnotatedObject
from geo_ai_metrics.annotation.annotation import rle2polygons


def read_coco_geoai_360(path: str) -> Annotation:
    """ TODO  
    """
    
    with open(path) as f:
        coco_dict = json.load(f)
    
    # Read list of coco categories which is list dicts with info
    # Then create conformity between ctg id and ctg number with ctg name   
    coco_categories = coco_dict['categories']
    coco_categories_conformity = {}
    for i, ctg in enumerate(coco_categories):
        coco_categories_conformity[ctg['id']] = {'num': i, 'name': ctg['name']}
    
    # Sort category ids
    ctg_ids = list(coco_categories_conformity.keys())
    ctg_ids.sort()  # CHECK
    
    # Create list of category names with order defined by sorted list of ids
    categories = []
    for ctg_id in ctg_ids:
        categories.append(coco_categories_conformity[ctg_id]['name'])
    
    # Create conformity between image id and image name (without ext)
    coco_images = coco_dict['images']
    coco_images_conformity = {img['id']: os.path.splitext(img['file_name'])[0] for img in coco_images}
    
    # Get labeled images
    labeled_images = {}
    for coco_image in coco_images:
        image_name = os.path.splitext(coco_image['file_name'])[0]
        labeled_images[image_name] = AnnotatedImage(
            height=coco_image['height'],
            width=coco_image['width'],
            annotations=[],
        )
    
    
    coco_annotations = coco_dict['annotations']
    for coco_bbox in coco_annotations:
        image_id = coco_bbox['image_id']
        ctg_id = coco_bbox['category_id']
        bbox_coords = coco_bbox['bbox']
        points = coco_bbox['points']
        
        if type(coco_bbox['segmentation']) == list:
            segmentation = coco_bbox['segmentation']
        else:
            segmentation = rle2polygons(coco_bbox['segmentation'])
            
        file_name = coco_images_conformity[image_id]
        bbox = AnnotatedObject(
            category_id=coco_categories_conformity[ctg_id]['num'],
            bbox=bbox_coords,
            segmentation=segmentation,
            points=points,
        )
        labeled_images[file_name].annotations.append(bbox)
    
    annotation = Annotation(categories=categories, images=labeled_images)
    return annotation

