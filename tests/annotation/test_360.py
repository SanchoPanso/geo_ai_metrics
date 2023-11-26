import sys
import os

from geo_ai_metrics.annotation.coco_geoai_360 import Annotation, AnnotatedImage, AnnotatedObject, read_coco_geoai_360


if __name__ == '__main__':
    path = r"D:\CodeProjects\PythonProjects\geo_ai_360\be_test_structure.lnk\1\.cache\Unnamed Run  1_Camera 4 360_0_0.json"
    annot = read_coco_geoai_360(path)
    print(annot)

