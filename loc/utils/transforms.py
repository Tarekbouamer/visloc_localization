
from torchvision import transforms

from loc.utils.constants import *


normalize_img_net = transforms.Compose([
                        transforms.Normalize(
                          mean=IMAGENET_DEFAULT_MEAN, 
                          std=IMAGENET_DEFAULT_STD),
                        ]
                                      )