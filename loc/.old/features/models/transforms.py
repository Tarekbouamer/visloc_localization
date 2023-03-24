from torchvision import transforms as tfn

from loc.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


tfn_image_net = tfn.Compose([
    tfn.Normalize(
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD),
]
)

tfn_grayscale = tfn.Compose([
    tfn.Grayscale(),
]
)