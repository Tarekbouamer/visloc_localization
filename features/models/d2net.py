import torch
import torch.nn as nn
import torch.nn.functional as F

from features.models.model_factory import create_model, load_pretrained
from features.models.model_register import get_pretrained_cfg, register_model


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512

        self.use_relu = use_relu

        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)


class EmptyTensorError(Exception):
    pass


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        
        pos = (pos - 0.5) / 2
    return pos


def interpolate_dense_features(pos, dense_features, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    _, h, w = dense_features.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
        w_top_left * dense_features[:, i_top_left, j_top_left] +
        w_top_right * dense_features[:, i_top_right, j_top_right] +
        w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left] +
        w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack([
            torch.stack([i_top_left, j_top_left], dim=0),
            torch.stack([i_top_right, j_top_right], dim=0),
            torch.stack([i_bottom_left, j_bottom_left], dim=0),
            torch.stack([i_bottom_right, j_bottom_right], dim=0)
        ], dim=0)
        return [descriptors, pos, ids, corners]


class D2Net(nn.Module):
    def __init__(self, cfg={}):
        super(D2Net, self).__init__()

        # cfg
        self.cfg = cfg

        #
        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_relu=True, use_cuda=False
        )

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()
    
    
    def transform_inputs(self, data):
        """ transform model inputs   """ 
        image = data["image"]       
        # caffe normalization
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = (image * 255 - norm.view(1, 3, 1, 1))  
        
        data["image"] = image
        return data
    
    
    def extract_features(self, data):
        
        data = self.transform_inputs(data)
        
        return self.dense_feature_extraction(data["image"])
    
    
    def detect(self, data, features=None):
        
        if features is None:
            features = self.extract_features(data)
        
        # detections 
        detections = self.detection(features)[0]        #list [0]
        fmap_pos = torch.nonzero(detections).t()        #     cpu()
        
        # displacements
        displacements = self.localization(features)[0]

        displacements_i = displacements[0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]]
        displacements_j = displacements[1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]]
        
        # msk
        mask = torch.min(torch.abs(displacements_i) < 0.5,
                         torch.abs(displacements_j) < 0.5
                         )
        
        # valid map and displacement 
        fmap_pos = fmap_pos[:, mask]
        valid_displacements = torch.stack([displacements_i[mask],
                                           displacements_j[mask]
                                           ], dim=0)  
        
        # 
        fmap_keypoints = fmap_pos[1:, :].float() + valid_displacements

        
        #
        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        scores = features[0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]]

        
        keypoints = keypoints.transpose(1,0)  # N, 2
        keypoints = keypoints[:, [1, 0]]    # x, y

        out = {
            'keypoints': keypoints,
            'scores': scores}

        return out


    def compute(self, data, features):

        #
        keypoints = data["keypoints"]
        scores = data["scores"]
        
        # kpts (fmap)
        keypoints_t = keypoints[:, [1, 0]]          # swap x y
        keypoints_t = keypoints_t.transpose(1,0)    # N, 2

        # downscale 
        fmap_keypoints = downscale_positions(keypoints_t, scaling_steps=2)

        # descriptors
        raw_descriptors, _, ids = interpolate_dense_features(fmap_keypoints, features[0])
        descriptors = F.normalize(raw_descriptors, dim=0)
        
        #
        out =  {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors
        }
        
        return out
          
            
    def forward(self, data):
        
        # features
        features = self.extract_features(data)

        # detect
        preds = self.detect(data, features)
        
        # compute
        preds = self.compute({**preds, **data}, features)
        
        return preds



def _cfg(url='', drive='', descriptor_dim=128, **kwargs):
    return {
        'url': url,
        'drive': drive,
        'descriptor_dim': descriptor_dim,
        **kwargs}


default_cfgs = {
    'd2net_ots': _cfg(url='https://dsmn.ml/files/d2-net/d2_ots.pth',
                      multiscale=False),
    'd2net_tf': _cfg(url='https://dsmn.ml/files/d2-net/d2_tf.pth',
                     multiscale=False),
    'd2_tf_no_phototourism': _cfg(url='https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth',
                                  multiscale=False)
}


def _make_model(name, cfg=None, pretrained=True, **kwargs):

    #
    default_cfg = get_pretrained_cfg(name)

    #
    model = D2Net(cfg=default_cfg)

    if pretrained:
        load_pretrained(model, name, default_cfg, state_key="model")

    return model


@register_model
def d2net_ots(cfg=None, **kwargs):
    return _make_model(name="d2net_ots", cfg=cfg)


@register_model
def d2net_tf(cfg=None, **kwargs):
    return _make_model(name="d2net_tf", cfg=cfg)


@register_model
def d2_tf_no_phototourism(cfg=None, **kwargs):
    return _make_model(name="d2_tf_no_phototourism", cfg=cfg)


if __name__ == '__main__':
    from features.utils.io import read_image, cv_to_tensor, show_cv_image_keypoints
    
    img_path = "features/graffiti.png"

    image, image_size = read_image(img_path)
    image = cv_to_tensor(image)
    detector = create_model("d2_tf_no_phototourism")
    
    with torch.no_grad():
        preds = detector.detect({'image': image})
        preds = detector({'image': image})
    
    print(preds["keypoints"].shape)
    
    show_cv_image_keypoints(image, preds['keypoints'])
