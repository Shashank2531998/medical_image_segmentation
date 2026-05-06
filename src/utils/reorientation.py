"""
Reorientation utilities for VoxTell predictions.

This module provides functions to reorient segmentation masks from the 
RAS (processing) space back to the original image orientation.
"""

from typing import Dict

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform


def reorient_seg_to_original(seg_data: np.ndarray, properties: dict) -> nib.Nifti1Image:
    """
    Reorient segmentation from RAS (processing) space back to original image orientation.
    
    Uses the same approach as nnUNetv2's NibabelIOWithReorient.write_seg()
    
    Args:
        seg_data: Segmentation array (single volume, not batched)
        properties: Metadata from NibabelIOWithReorient.read_images()
        
    Returns:
        Reoriented NIfTI image matching original image orientation
    """
    nibabel_stuff = properties['nibabel_stuff']
    original_affine = nibabel_stuff['original_affine']
    reoriented_affine = nibabel_stuff['reoriented_affine']
    
    # CRITICAL: Reverse the transpose that was applied during read_images()
    # NibabelIOWithReorient transposes as (2, 1, 0) for SimpleITK compatibility
    seg_data = seg_data.transpose((2, 1, 0))
    
    # Create temporary NIfTI with reoriented affine
    seg_nib = nib.Nifti1Image(seg_data, affine=reoriented_affine)
    
    # Compute reorientation transform back to original orientation
    img_ornt = io_orientation(original_affine)
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    
    # Apply inverse reorientation
    seg_nib_reoriented = seg_nib.as_reoriented(from_canonical)
    
    # Verify affine matches
    if not np.allclose(original_affine, seg_nib_reoriented.affine):
        print(f'WARNING: Restored affine does not match original affine')
        print(f'Original affine:\n{original_affine}')
        print(f'Restored affine:\n{seg_nib_reoriented.affine}')
    
    return seg_nib_reoriented


def reorient_seg_from_props(seg: np.ndarray, properties: dict):
    # revert transpose
    seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)

    seg_nib = nib.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
    img_ornt = io_orientation(properties['nibabel_stuff']['original_affine'])
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    seg_nib_reoriented = seg_nib.as_reoriented(from_canonical)
    if not np.allclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine):
        print(f'WARNING: Restored affine does not match original affine.')
        print(f'Original affine\n', properties['nibabel_stuff']['original_affine'])
        print(f'Restored affine\n', seg_nib_reoriented.affine)
    return seg_nib_reoriented
