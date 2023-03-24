
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfn
from PIL import Image
from features.models.transforms import tfn_image_net

import torch.nn.functional as functional

from features.models.model_factory import create_model, load_pretrained
from features.models.model_register import register_model, get_pretrained_cfg


class NonMaxSuppression (nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        # reliability, repeatability = reliability[0], repeatability[0]
        
        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


class BaseNet (nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = functional.softplus(ux)
            # for sure in [0,1], much less plateaus than softmax
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return functional.softmax(ux, dim=1)[:, 1:2]

    def normalize(self, x, ureliability, urepeatability):
        
        return dict(descriptors=functional.normalize(x, p=2, dim=1),
                    repeatability=self.softmax(urepeatability),
                    reliability=self.softmax(ureliability))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k: [r[k] for r in res if k in r] 
               for k in {k for r in res for k in r}}
        
        return dict(res, imgs=imgs, **kw)

    def extract_features(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        return x

class PatchNet (BaseNet):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """

    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, k_pool=1, pool_type='max'):
        # as in the original implementation, dilation is applied at the end of layer, so it will have impact only from next layer
        d = self.dilation * dilation
        if self.dilated:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append(nn.Conv2d(self.curchan, outd,
                        kernel_size=k, **conv_params))
        if bn and self.bn:
            self.ops.append(self._make_bn(outd))
        if relu:
            self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

        if k_pool > 1:
            if pool_type == 'avg':
                self.ops.append(nn.AvgPool2d(kernel_size=k_pool))
            elif pool_type == 'max':
                self.ops.append(nn.MaxPool2d(kernel_size=k_pool))
            else:
                print(f"Error, unknown pooling type {pool_type}...")

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n, op in enumerate(self.ops):
            x = op(x)
        return self.normalize(x)

class Quad_L2Net (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """

    def __init__(self, dim=128, mchan=4, relu22=False, **kw):
        PatchNet.__init__(self, **kw)
        self._add_conv(8*mchan)
        self._add_conv(8*mchan)
        self._add_conv(16*mchan, stride=2)
        self._add_conv(16*mchan)
        self._add_conv(32*mchan, stride=2)
        self._add_conv(32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv(32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim

class Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """

    def __init__(self, **kw):
        Quad_L2Net.__init__(self, **kw)
        
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        
        return self.normalize(x, ureliability, urepeatability)


class Fast_Quad_L2Net (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, k_pool = downsample_factor) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        
        # Go back to initial image resolution with upsampling
        self.ops.append(nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
        
        self.out_dim = dim

class Fast_Quad_L2Net_ConfCFS (Fast_Quad_L2Net):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Fast_Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)
        

class R2d2Net(nn.Module):
    def __init__(self, net, cfg):
        super().__init__()

        # 
        self.cfg = cfg

        # 
        self.net = net
        self.detector = NonMaxSuppression(
                                    rel_thr=cfg["reliability_threshold"],
                                    rep_thr=cfg["repetability_threshold"])

    def transform_inputs(self, data):
        """ transform model inputs   """   
        data["image"]  = tfn_image_net(data["image"] )
        return data
    
    def extract_features(self, data):
        #
        data = self.transform_inputs(data)
        
        return self.net.extract_features(data["image"])
    
    def detect(self, data, features=None):
 
        # extract features
        if features is None:
            features = self.extract_features(data)
        
        # ureliability &&  urepeatability
        ureliability = self.net.clf(features**2)
        urepeatability = self.net.sal(features**2)
        
        # reliability &&  repeatability
        repeatability = self.net.softmax(urepeatability)
        reliability   = self.net.softmax(ureliability)
        
        # nms 
        y, x = self.detector(reliability, repeatability) 
        
        # scores
        C = reliability[0, 0, y, x]
        Q = repeatability[0, 0, y, x]
        scores = C * Q

        # keypoints
        X = x.float()
        Y = y.float() 
        keypoints = torch.stack([X, Y], dim=-1)
        
        if self.cfg['max_keypoints'] > 0:
            idxs = (-scores).argsort()[:self.cfg['max_keypoints']]
            keypoints = keypoints[idxs]
            scores = scores[idxs]
        
        return {
            'keypoints': keypoints,
            'scores': scores
            }

    def compute(self, data, features):
        #
        keypoints = data["keypoints"]
        scores = data["scores"]
        
        x, y = keypoints[:, 0], keypoints[:, 1]
        x = x.long()
        y = y.long()

        descriptors = functional.normalize(features, p=2, dim=1)        
        descriptors = descriptors[0, : , y, x].t()
        
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
    'r2d2_WASF_N16':
        _cfg(
            drive='https://drive.google.com/uc?id=1yHiLse1yopT7Ylsx6iVZ3M-_WRaKssp9',
            max_keypoints=5000, reliability_threshold=0.7, repetability_threshold=0.7),

    'r2d2_WASF_N8_big':
        _cfg(drive='https://drive.google.com/uc?id=1qUtQMZPU8x4Kv0jwbm22bEK6tNvO7qPi',
            max_keypoints=5000, reliability_threshold=0.7, repetability_threshold=0.7),

    'r2d2_WAF_N16':
        _cfg(drive='https://drive.google.com/uc?id=1SPPnagMOXv0aFEBUAhlY42WFZ2C6ArFg',
            max_keypoints=5000, reliability_threshold=0.7, repetability_threshold=0.7),
        
    'faster2d2_WASF_N16':
        _cfg(drive='https://drive.google.com/uc?id=1glXoORF9-7N6zR4-fFengt_J1lMyQaZV',
            max_keypoints=5000, reliability_threshold=0.7, repetability_threshold=0.7),

    'faster2d2_WASF_N8_big':
        _cfg(drive='https://drive.google.com/uc?id=1gvRap5g0ORnk9s4YCR7md-qs2JMeMGqn',
            max_keypoints=5000, reliability_threshold=0.7, repetability_threshold=0.7),
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):

    #
    default_cfg = get_pretrained_cfg(name)
    cfg = {**default_cfg, **cfg}

    #
    if name == "r2d2_WASF_N16":
        net = Fast_Quad_L2Net_ConfCFS()

    if name == "r2d2_WASF_N8_big":
        net = Quad_L2Net_ConfCFS(mchan=6)

    if name == "r2d2_WAF_N16":
        net = Quad_L2Net_ConfCFS()
        
    if name == "faster2d2_WASF_N16":
        net = Fast_Quad_L2Net_ConfCFS()    

    if name == "faster2d2_WASF_N8_big":
        net = Fast_Quad_L2Net_ConfCFS(mchan=6)    
        
    if pretrained:
        load_pretrained(net, name, cfg, state_key="state_dict", 
                        replace=('module.',''))
    
    return R2d2Net(net, cfg)


@register_model
def r2d2_WASF_N16(cfg=None, **kwargs):
    return _make_model(name="r2d2_WASF_N16", cfg=cfg)

@register_model
def r2d2_WASF_N8_big(cfg=None, **kwargs):
    return _make_model(name="r2d2_WASF_N8_big", cfg=cfg)

@register_model
def r2d2_WAF_N16(cfg=None, **kwargs):
    return _make_model(name="r2d2_WAF_N16", cfg=cfg)

@register_model
def faster2d2_WASF_N16(cfg=None, **kwargs):
    return _make_model(name="faster2d2_WASF_N16", cfg=cfg)

@register_model
def faster2d2_WASF_N8_big(cfg=None, **kwargs):
    return _make_model(name="faster2d2_WASF_N8_big", cfg=cfg)



if __name__ == '__main__':
    from features.utils.io import read_image, show_cv_image, \
        cv_to_tensor, show_cv_image_keypoints

    img_path = "features/graffiti.png"

    image, image_size = read_image(img_path)

    image = cv_to_tensor(image)

    detector = create_model("r2d2_WASF_N16", cfg={"max_keypoints": 1024})
    
    print(detector)

    with torch.no_grad():
        preds = detector({'image': image})
    
    for k,v in preds.items():
        print(k, "    ",v.shape)

    show_cv_image_keypoints(image, preds['keypoints'])
