import os
import sys
import copy
import numpy as np
import random
from pathlib import Path
import json


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    src_annot_path = str(data_dir / '360' / 'annotations' / 'Unnamed Run  1_Camera 4 360_0_2.json')
    dst_annot_path = str(data_dir / '360' / 'annotations' / 'gt.json')
    
    with open(src_annot_path) as f:
        data = json.load(f)
    
    data['images'][0]['id'] = 1
    for an in data['annotations']:
        an['image_id'] = 1

    with open(dst_annot_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()

