import random

import torch

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class NuImageDataset(CocoDataset):
    CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]

    def __init__(self, prompt_version='v1', num_bucket_per_side=None, 
                 foreground_loss_mode=None, foreground_loss_weight=1.0, foreground_loss_norm=False, feat_size=64,
                 uncond_prob=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.prompt_version = prompt_version
        self.no_sections = num_bucket_per_side
        print('Using prompt version: {}, num_bucket_per_side: {}'.format(prompt_version, num_bucket_per_side))
        
        self.FEAT_SIZE = [each // 8 for each in kwargs['pipeline'][2].img_scale][::-1]
        print('Using feature size: {}'.format(self.FEAT_SIZE))
        
        self.foreground_loss_mode = foreground_loss_mode
        self.foreground_loss_weight = foreground_loss_weight
        self.foreground_loss_norm = foreground_loss_norm
        print('Using foreground_loss_mode: {}, foreground_loss_weight: {}, foreground_loss_norm: {}'.format(foreground_loss_mode, foreground_loss_weight, foreground_loss_norm))
        
        self.uncond_prob = uncond_prob
        print('Using unconditional generation probability: {}'.format(uncond_prob))
        
        self.class2text = {
            'car': 'car', 
            'truck': 'truck', 
            'trailer': 'trailer', 
            'bus': 'bus',
            'construction_vehicle': 'construction',
            'bicycle': 'bicycle', 
            'motorcycle': 'motorcycle', 
            'pedestrian': 'pedestrian',
            'traffic_cone': 'cone',
            'barrier': 'barrier', 
        }
        
    def __getitem__(self, idx):
        ##################################
        # Data item: {pixel_values: tensor of (3, H, W),  text: string}
        ##################################
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            bboxes = data['gt_bboxes'].data
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            labels = [self.CLASSES[each].split('.')[-1] for each in data['gt_labels'].data]
            camera = ' '.join(data['img_metas'].data['ori_filename'].split('/')[-2].split('_')[1:])
            
            if self.prompt_version == 'v1':
                pad_shape = data['img_metas'].data['pad_shape']
                img_shape = torch.tensor([pad_shape[1], pad_shape[0], pad_shape[1], pad_shape[0]])
                bboxes /= img_shape
                
                # random shuffle bbox annotations
                index = list(range(len(labels)))
                random.shuffle(index)
                index = index[:22] # 9+3*22+2=77
                
                # generate bbox mask and text prompt
                # constant: background -> 0, foreground -> self.foreground_loss_weight
                # area:     background -> 0, foreground -> 1 / area ^ self.foreground_loss_weight (for area, smaller weight, larger variance with respect to areas)
                objs = []
                bbox_mask = torch.zeros(self.FEAT_SIZE).float() # [H, W]
                for each in index:
                    label = labels[each]
                    bbox = bboxes[each]
                    
                    # generate bbox mask
                    FEAT_SIZE = torch.tensor([self.FEAT_SIZE[1], self.FEAT_SIZE[0], self.FEAT_SIZE[1], self.FEAT_SIZE[0]])
                    coord = torch.round(bbox * FEAT_SIZE).int().tolist()
                    # bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight if self.foreground_loss_mode == 'constant' else \
                    #                                                     1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), self.foreground_loss_weight)
                    if label in ['truck', 'bicycle', 'motorcycle']:
                        bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 2 * 1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), 0.2) if self.foreground_loss_mode == 'constant' else \
                                                                        1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), self.foreground_loss_weight)
                    else:
                        bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), 0.2) if self.foreground_loss_mode == 'constant' else \
                                                                        1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), self.foreground_loss_weight)
                    
                    # generate text prompt
                    bbox = self.token_pair_from_bbox(bbox.tolist())
                    objs.append(' '.join([self.class2text[label], bbox]))
                
                # bbox_mask[bbox_mask == 0] = 1 if self.foreground_loss_mode == 'constant' else 1 / torch.pow(torch.sum(bbox_mask == 0), self.foreground_loss_weight)
                bbox_mask[bbox_mask == 0] = 1 * 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), 0.2) if self.foreground_loss_mode == 'constant' else 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), self.foreground_loss_weight)
                bbox_mask = bbox_mask / torch.sum(bbox_mask) * self.FEAT_SIZE[0] * self.FEAT_SIZE[1] if self.foreground_loss_norm else bbox_mask
                
                if self.uncond_prob > 0:
                    text = 'A driving scene image of ' + camera.lower() + ' camera with ' + ' '.join(objs) if random.random() > self.uncond_prob else ""
                else:
                    text = 'A driving scene image of ' + camera.lower() + ' camera with ' + ' '.join(objs)
                
            else:
                raise NotImplementedError("Prompt version {} is not supported!".format(self.prompt_version))
            
            example = {}
            example["pixel_values"] = data['img'].data
            example["text"] = text
            if self.foreground_loss_mode is not None:
                example["bbox_mask"] = bbox_mask

            return example
    
    # code borrowed from https://github.com/CompVis/taming-transformers
    def tokenize_coordinates(self, x: float, y: float) -> int:
        """
        Express 2d coordinates with one number.
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        x_discrete = int(round(x * (self.no_sections[0] - 1)))
        y_discrete = int(round(y * (self.no_sections[1] - 1)))
        return "<l{}>".format(y_discrete * self.no_sections[0] + x_discrete)

    def token_pair_from_bbox(self, bbox):
        return self.tokenize_coordinates(bbox[0], bbox[1]) + ' ' + self.tokenize_coordinates(bbox[2], bbox[3])