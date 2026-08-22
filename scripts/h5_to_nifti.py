import os
import h5py
import numpy as np
import nibabel as nib
from glob import glob


def convert_single(h5_path, img_dir, mask_dir, normalize=False):
    base_name = os.path.splitext(os.path.basename(h5_path))[0]

    img_out = os.path.join(img_dir, f"{base_name}.nii.gz")
    mask_out = os.path.join(mask_dir, f"{base_name}.nii.gz")

    # -------------------------
    # Skip if already exists
    # -------------------------
    if os.path.exists(img_out) and os.path.exists(mask_out):
        print(f"[SKIP] {base_name} already exists")
        return

    with h5py.File(h5_path, "r") as f:
        image = f["echo1"][()]
        seg = f["seg"][()]

    # -------------------------
    # IMAGE
    # -------------------------
    image = image.astype(np.float32)

    if normalize:
        image = (image - image.mean()) / (image.std() + 1e-8)

    img_nifti = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(img_nifti, img_out)

    # -------------------------
    # SEGMENTATION (6-class)
    # -------------------------
    seg_labels = np.argmax(seg, axis=-1).astype(np.uint8)

    # 1. Identify which pixels are entirely background (sum of channels == 0)
    is_background = (np.sum(seg, axis=-1) == 0)
    
    # 2. Shift all anatomical labels up by 1 (so Patellar Cartilage = 1)
    seg_labels = np.argmax(seg, axis=-1) + 1
    
    # 3. Override the background pixels to be exactly 0
    seg_labels[is_background] = 0
    seg_labels = seg_labels.astype(np.uint8)

    mask_nifti = nib.Nifti1Image(seg_labels, affine=np.eye(4))
    
    nib.save(mask_nifti, mask_out)

    print(f"[DONE] {base_name}")


def convert_folder(h5_folder, img_dir, mask_dir, normalize=False):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    h5_files = sorted(glob(os.path.join(h5_folder, "*.h5")))

    print(f"Found {len(h5_files)} .h5 files")

    for h5_path in h5_files:
        convert_single(h5_path, img_dir, mask_dir, normalize)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":

    h5_path = "/home/woody/iwi5/iwi5326h/projects/VoxTell/data/skm_tea/h5_img_files"
    img_dir = "/home/vault/iwi5/iwi5326h/projects/VoxTell/data/skm_tea/images"
    mask_dir = "/home/vault/iwi5/iwi5326h/projects/VoxTell/data/skm_tea/annotations"

    convert_folder(
        h5_path,
        img_dir,
        mask_dir,
        normalize=False
    )