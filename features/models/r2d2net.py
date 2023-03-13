
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfn
from PIL import Image

import torch.nn.functional as functional

from features.models.model_factory import create_model, load_pretrained
from features.models.model_register import register_model, get_pretrained_cfg


RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tfn.Compose(
    [tfn.ToTensor(), tfn.Normalize(mean=RGB_mean, std=RGB_std)])


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        print('Launching on CPU')

    return


def model_size(model):
    ''' Computes the number of parameters of the model 
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print(checkpoint.keys())

    print(type(checkpoint['net']))

    print("\n>> Creating net = " + checkpoint['net'])

    net = eval(checkpoint['net'])
    nb_of_weights = model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict(
        {k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


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
                self.ops.append(torch.nn.AvgPool2d(kernel_size=k_pool))
            elif pool_type == 'max':
                self.ops.append(torch.nn.MaxPool2d(kernel_size=k_pool))
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


class R2d2Net:
    def __init__(self,
                 num_features=2000,
                 scale_f=2**0.25,
                 min_size=256,
                 max_size=1300,  # 1024,
                 min_scale=0,
                 max_scale=1,
                 reliability_thr=0.7,
                 repeatability_thr=0.7,
                 do_cuda=True):
        print('Using R2d2Feature2D')
        self.model_base_path = 'features'
        self.model_weights_path = self.model_base_path + '/r2d2_WASF_N16.pt'
        # print('model_weights_path:',self.model_weights_path)

        self.pts = []
        self.kps = []
        self.des = []
        self.frame = None

        self.num_features = num_features
        self.scale_f = scale_f
        self.min_size = min_size
        self.max_size = max_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.reliability_thr = reliability_thr
        self.repeatability_thr = repeatability_thr
        self.do_cuda = do_cuda
        if do_cuda:
            gpus = [0]
        else:
            gpus = -1
        self.gpus = gpus
        self.do_cuda = torch_set_gpu(gpus)

        print('==> Loading pre-trained network.')

        self.net = load_network(self.model_weights_path)

        if self.do_cuda:
            self.net = self.net.cuda()

        # create the non-maxima detector
        self.detector = NonMaxSuppression(
            rel_thr=reliability_thr, rep_thr=repeatability_thr)

        print('==> Successfully loaded pre-trained network.')


def _make_model(name, cfg=None, pretrained=True, **kwargs):

    #
    default_cfg = get_pretrained_cfg(name)

    #
    if name == "r2d2_WASF_N16":
        model = Quad_L2Net_ConfCFS()

    if pretrained:
        load_pretrained(model, name, default_cfg)

    return model


@register_model
def r2d2_WASF_N16(cfg=None, **kwargs):
    return _make_model(name="r2d2_WASF_N16", cfg=cfg)


if __name__ == '__main__':
    from features.utils.io import read_image, show_cv_image, \
        cv_to_tensor, show_cv_image_keypoints

    img_path = "features/graffiti.png"

    image, image_size = read_image(img_path)

    image = cv_to_tensor(image)

    detector = create_model("r2d2_WASF_N16")
    with torch.no_grad():
        preds = detector({'image': image})
        preds = detector({'image': image})

    kpts = preds['keypoints'][0]
    show_cv_image_keypoints(image, kpts)
