import warnings
from typing import Dict, List, Optional, Tuple

import torch

from kornia.color import rgb_to_grayscale
from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK_LAF


from kornia.feature.hardnet import HardNet

from kornia.feature.laf import extract_patches_from_pyramid, scale_laf

def get_laf_descriptors(
    img: Tensor, lafs: Tensor, patch_descriptor: Module, patch_size: int = 32, grayscale_descriptor: bool = True
) -> Tensor:
    r"""Function to get local descriptors, corresponding to LAFs (keypoints).
    Args:
        img: image features with shape :math:`(B,C,H,W)`.
        lafs: local affine frames :math:`(B,N,2,3)`.
        patch_descriptor: patch descriptor module, e.g. :class:`~kornia.feature.SIFTDescriptor`
            or :class:`~kornia.feature.HardNet`.
        patch_size: patch size in pixels, which descriptor expects.
        grayscale_descriptor: True if ``patch_descriptor`` expects single-channel image.
    Returns:
        Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
    """
    KORNIA_CHECK_LAF(lafs)
    patch_descriptor = patch_descriptor.to(img)
    patch_descriptor.eval()

    timg: Tensor = img
    if lafs.shape[1] == 0:
        warnings.warn(f"LAF contains no keypoints {lafs.shape}, returning empty tensor")
        return torch.empty(lafs.shape[0], lafs.shape[1], 128)
    if grayscale_descriptor and img.size(1) == 3:
        timg = rgb_to_grayscale(img)

    patches: Tensor = extract_patches_from_pyramid(timg, lafs, patch_size)
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :)
    B, N, CH, H, W = patches.size()
    return patch_descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
  
  
class LAFDescriptor(Module):
    r"""Module to get local descriptors, corresponding to LAFs (keypoints).
    Internally uses :func:`~kornia.feature.get_laf_descriptors`.
    Args:
        patch_descriptor_module: patch descriptor module, e.g. :class:`~kornia.feature.SIFTDescriptor`
            or :class:`~kornia.feature.HardNet`. Default: :class:`~kornia.feature.HardNet`.
        patch_size: patch size in pixels, which descriptor expects.
        grayscale_descriptor: ``True`` if patch_descriptor expects single-channel image.
    """

    def __init__(
        self, patch_descriptor_module: Optional[Module] = None, patch_size: int = 32, grayscale_descriptor: bool = True
    ) -> None:
        super().__init__()
        if patch_descriptor_module is None:
            patch_descriptor_module = HardNet(True)
        self.descriptor = patch_descriptor_module
        self.patch_size = patch_size
        self.grayscale_descriptor = grayscale_descriptor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(descriptor={self.descriptor.__repr__()}, "
            f"patch_size={self.patch_size}, "
            f"grayscale_descriptor='{self.grayscale_descriptor})"
        )

    def forward(self, img: Tensor, lafs: Tensor) -> Tensor:
        r"""Three stage local feature detection.
        First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.
        Args:
            img: image features with shape :math:`(B,C,H,W)`.
            lafs: local affine frames :math:`(B,N,2,3)`.
        Returns:
            Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
        """
        return get_laf_descriptors(img, lafs, self.descriptor, self.patch_size, self.grayscale_descriptor)
      
      
class LocalFeature(Module):
    """Module, which combines local feature detector and descriptor.
    Args:
        detector: the detection module.
        descriptor: the descriptor module.
        scaling_coef: multiplier for change default detector scale (e.g. it is too small for KeyNet by default)
    """

    def __init__(self, detector: Module, descriptor: LAFDescriptor, scaling_coef: float = 1.0) -> None:
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor
        if scaling_coef <= 0:
            raise ValueError(f"Scaling coef should be >= 0, got {scaling_coef}")
        self.scaling_coef = scaling_coef

    def forward(self, img: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            img: image to extract features with shape :math:`(B,C,H,W)`.
            mask: a mask with weights where to apply the response function.
                The shape must be the same as the input image.
        Returns:
            - Detected local affine frames with shape :math:`(B,N,2,3)`.
            - Response function values for corresponding lafs with shape :math:`(B,N,1)`.
            - Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
        """
        lafs, responses = self.detector(img, mask)
        lafs = scale_laf(lafs, self.scaling_coef)
        descs = self.descriptor(img, lafs)
        return (lafs, responses, descs)


# detector (img)
# descriptor (img, kpts)
# nms maybe