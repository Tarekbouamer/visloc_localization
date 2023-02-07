
from torchvision import transforms

from loc.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


normalize_img_net = transforms.Compose([
    transforms.Normalize(
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD),
]
)

to_gray = transforms.Compose([
    transforms.Grayscale(),
]
)
