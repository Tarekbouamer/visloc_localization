from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from features.models.transforms import tfn_grayscale

from features.models.model_factory import create_model, load_pretrained
from features.models.model_register import register_model, get_pretrained_cfg


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (
        keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


    
 

class SuperPoint(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()

        # cfg
        self.cfg = cfg
        
        # model
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)


        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.cfg['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)
        
    def transform_inputs(self, data):
        """ transform model inputs   """        
        data["image"] = tfn_grayscale(data["image"])
        
        return data
        
    def extract_features(self, data):
                
        # transform
        data = self.transform_inputs(data)
        
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        return x
        
                
    def detect(self, data, features=None):
        """ Compute keypoints, scores, descriptors for image """

        # extract
        if features is None:
            features = self.extract_features(data)
            
        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(features))
        scores = self.convPb(cPa)
        
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        
        scores = simple_nms(scores, self.cfg['nms_radius'])

        # extract keypoints
        keypoints = [torch.nonzero(s > self.cfg['keypoint_threshold']) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.cfg['remove_borders'], h*8, w*8) for k, s in zip(keypoints, scores)
            ]))

        # keep the k keypoints with highest score
        if self.cfg['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.cfg['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        #
        out = {
            'keypoints': keypoints,
            'scores': scores}
        
        return out
        
    def compute(self, data, features):
        #
        keypoints = data["keypoints"]
        scores = data["scores"]
        
        cDa = self.relu(self.convDa(features))
        descriptors = self.convDb(cDa)
        
        #
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]
        
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors
        }
    

    def forward(self, data):
        
        #
        features = self.extract_features(data)

        # detect
        preds = self.detect(data, features)
        
        # compute
        preds = self.compute({**data, **preds}, features)
        
        return preds


def _cfg(url='', drive='', descriptor_dim=128, **kwargs):
    return {
        'url': url,
        'drive': drive,
        'descriptor_dim': descriptor_dim,
        **kwargs}


default_cfgs = {
    'superpoint':
        _cfg(
            drive='https://drive.google.com/uc?id=1JjRJ5RLa3yx4VOSZ17mryJoXiWGbeCa1',
            descriptor_dim=256, nms_radius=4, keypoint_threshold=0.005, max_keypoints=-1, remove_borders=4)
}


def _make_model(name, cfg=None, pretrained=True,**kwargs):
    
    #
    default_cfg = get_pretrained_cfg(name)
    
    #
    model = SuperPoint(cfg=default_cfg)
    
    if pretrained:
        load_pretrained(model, name, default_cfg) 
        
    return model
    

@register_model
def superpoint(cfg=None, **kwargs):
    return _make_model(name="superpoint", cfg=cfg)


if __name__ == '__main__':
    from features.utils.io import read_image, show_cv_image, \
        cv_to_tensor, show_cv_image_keypoints
    
    img_path = "features/graffiti.png"

    image, image_size = read_image(img_path)

    image = cv_to_tensor(image)
    
    detector = create_model("superpoint")
    with torch.no_grad():
        preds = detector.detect({'image': image})
        preds = detector({'image': image})

    kpts = preds['keypoints'][0]
    show_cv_image_keypoints(image, kpts)
