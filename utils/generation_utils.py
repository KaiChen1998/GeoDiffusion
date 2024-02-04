import os
import json
import random
import numpy as np
import bbox_visualizer as bbv
from itertools import cycle

import torch
import diffusers

#####################
# Constants
#####################
dataset2classes = dict(
    nuimages=dict(
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]
    ),
    coco_stuff=dict(
        classes=[
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 
            'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 
            'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 
            'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 
            'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 
            'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 
            'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 
            'tent', 'textile-other', 'towel', 'tree', 'vegetable', 
            'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 
            'waterdrops', 'window-blind', 'window-other', 'wood', 'other'
        ]
    )
)
dataset2classes['nuimages']['class2text'] = {each: each for each in dataset2classes['nuimages']['classes']}
dataset2classes['nuimages']['class2text'].update(dict(construction_vehicle='construction', traffic_cone='cone'))
dataset2classes['coco_stuff']['class2text'] = {each: each.replace('-', ' ') for each in dataset2classes['coco_stuff']['classes']}

COLOR_PALETTE = [(30, 118, 179), (255, 126, 13), (43, 159, 43), (213, 38, 39), (147, 102, 188),
                 (139, 85, 74), (226, 118, 193), (126, 126, 126), (187, 188, 33), (22, 189, 206)]

#####################
# Loading
#####################
def load_checkpoint(ckpt_path, pipeline=diffusers.StableDiffusionPipeline):
    pipe = pipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
    generation_config = json.load(open(os.path.join(ckpt_path, "generation_config.json")))
    generation_config['dataset2classes'] = dataset2classes[generation_config['dataset']]

    if generation_config['dataset'] == 'coco_stuff':
        assert '0.16.0' in diffusers.__version__, "Be default, we adopt diffusers==0.16.0 to adopt DPMSolver++ for inference on COCO-Stuff."
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe, generation_config

#####################
# Bbox encoding
#####################
def bbox_encode(bboxs, configs):
    random.shuffle(bboxs)
    assert len(bboxs) <= configs["max_num_bbox"], 'number of bboxes should not exceed {}, but got {}'.format(configs["max_num_bbox"], len(bboxs))
    
    objs = []
    for bbox in bboxs:
        # bbox sanity check
        label, bbox = bbox[0], bbox[1:]
        assert label in configs["dataset2classes"]["classes"], "bbox class should be selected from the {} dataset, but got {}".format(configs['dataset'], label)
        assert bbox[2] > bbox[0] and bbox[3] > bbox[1] and bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= 1 and bbox[3] <= 1, "bbox coord should be in the (x1, x2, y1, y2) format, but got {}".format(bbox)

        # bbox encode
        bbox = tokenize_coordinates(bbox[0], bbox[1], configs["num_bucket_per_side"]) + ' ' + tokenize_coordinates(bbox[2], bbox[3], configs["num_bucket_per_side"])
        objs.append(' '.join([configs["dataset2classes"]["class2text"][label], bbox]))
    return ' '.join(objs)

# code borrowed from https://github.com/CompVis/taming-transformers
def tokenize_coordinates(x, y, num_bucket_per_side):
    """
    Express 2d coordinates with one number.
    Example: assume num_bucket_per_side = [4, 4]:
    0  0  0  0
    0  0  #  0
    0  0  0  0
    0  0  0  x
    Then the # position corresponds to token 6, the x position to token 15.
    @param x: float in [0, 1]
    @param y: float in [0, 1]
    @return: discrete tokenized coordinate
    """
    x_discrete = int(round(x * (num_bucket_per_side[0] - 1)))
    y_discrete = int(round(y * (num_bucket_per_side[1] - 1)))
    return "<l{}>".format(y_discrete * num_bucket_per_side[0] + x_discrete)

#####################
# Visualization
#####################
def draw_layout(bboxes):
    canvas = np.ones((384, 384, 3)) * 255
    canvas = canvas.astype(np.uint8)
    canvas = bbv.draw_rectangle(canvas, [0, 0, 383, 383], bbox_color=(0, 0, 0), thickness=3)    
    for bbox, color in zip(bboxes, cycle(COLOR_PALETTE)):
        label, x1, y1, x2, y2 = bbox
        bbox = [int(x1 * 384), int(y1 * 384), int(x2 * 384), int(y2 * 384)]
        canvas = bbv.draw_rectangle(canvas, bbox, bbox_color=color)
        canvas = bbv.add_label(canvas, label, bbox, top=False, text_bg_color=color, text_color=(255, 255, 255))
    return canvas