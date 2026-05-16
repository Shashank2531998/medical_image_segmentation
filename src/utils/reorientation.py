"""
Reorientation utilities for VoxTell predictions.

This module provides functions to reorient segmentation masks from the 
RAS (processing) space back to the original image orientation.
"""

from typing import Dict

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from src.utils.logging import get_logger


logger = get_logger(__name__)


def reorient_seg_from_props(seg: np.ndarray, properties: dict):
    # revert transpose
    seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)

    seg_nib = nib.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
    img_ornt = io_orientation(properties['nibabel_stuff']['original_affine'])
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    seg_nib_reoriented = seg_nib.as_reoriented(from_canonical)
    if not np.allclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine):
        logger.warning("Restored affine does not match original affine.")
        logger.warning("Original affine\n%s", properties['nibabel_stuff']['original_affine'])
        logger.warning("Restored affine\n%s", seg_nib_reoriented.affine)
    return seg_nib_reoriented
