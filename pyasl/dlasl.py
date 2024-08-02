import os
import numpy as np
import glob
import nibabel as nib
from scipy.ndimage import affine_transform
from utils.models import dilated_net_wide
from utils.utils import read_data_description, load_img


def dlasl_resample(v, data):
    original_shape = data.shape
    original_affine = v.affine

    new_shape = (64, 64, 24)
    scale_factors = [o / n for o, n in zip(original_shape, new_shape)]
    new_affine = np.copy(original_affine)
    np.fill_diagonal(
        new_affine[:3, :3], np.diag(original_affine[:3, :3] * scale_factors)
    )

    resampled_data = affine_transform(
        data,
        matrix=np.linalg.inv(new_affine) @ original_affine,
        output_shape=new_shape,
        order=3,
    )

    resampled_img = nib.Nifti1Image(
        resampled_data,
        new_affine,
    )
    return resampled_img, resampled_data


def get_subj_c123(dir: str):
    found_files = glob.glob(os.path.join(dir, ".*c1.*\.(nii|nii\.gz)$"))
    if not found_files:
        raise FileNotFoundError("No c1 segmentation nii/nii.gz files!")
    v, c1 = load_img(found_files[0])

    found_files = glob.glob(os.path.join(dir, ".*c2.*\.(nii|nii\.gz)$"))
    if not found_files:
        raise FileNotFoundError("No c2 segmentation nii/nii.gz files!")
    _, c2 = load_img(found_files[0])

    found_files = glob.glob(os.path.join(dir, ".*c3.*\.(nii|nii\.gz)$"))
    if not found_files:
        raise FileNotFoundError("No c3 segmentation nii/nii.gz files!")
    _, c3 = load_img(found_files[0])

    c123 = c1.get_fdata() + c2.get_fdata() + c3.get_fdata()
    return v, c123


def dlasl_pipeline(root: str, model_selection=1, pattern=".*_CBF\.(nii|nii\.gz)$"):
    data_descrip = read_data_description(root)
    model = dilated_net_wide(3)
    if model_selection == 0:
        model_name = "model_068.hdf5"  # trained model on healthy subjects
    else:
        model_name = "model_099.hdf5"  # fine-tuned model on the ADNI dataset
    model.load_weights(os.path.join("models", model_name))
    for key, value in data_descrip["Images"].items():
        key = key.replace("rawdata", "derivatives")
        vmask, mask = get_subj_c123(os.path.join(key, "anat"))
        vmask, mask = dlasl_resample(vmask, mask)
        mask = (mask > 0.5).astype(np.int)
        vmask.header.set_data_dtype(np.int)
        found_cbf_files = glob.glob(os.path.join(key, "perf", pattern))
        if not found_cbf_files:
            raise FileNotFoundError(
                f"No CBF files found matching the pattern: {pattern}"
            )
        for cbf_file in found_cbf_files:
            item_obj, item = load_img(cbf_file)
            if item.ndim == 4:
                item = np.mean(item, axis=3)
            vitem, item = dlasl_resample(item_obj, item)
            item[mask[:, :, :] < 0.1] = 2
            y = np.clip(item, 0, 150) / 255.0
            y_ = np.transpose(y, (2, 0, 1))
            y_ = y.reshape((y.shape[0], y.shape[1], y.shape[2], 1))
            x_ = model.predict(y_)
            x_ = np.clip(np.squeeze(x_ * 255.0), 0, 150)
            mask[mask > 0] = 1
            x_ = x_ * mask
            x_nii = nib.Nifti1Image(x_, vitem.affine, vitem.header)
            path, filename = os.path.split(cbf_file)
            x_nii.to_filename(os.path.join(path, f"denoised_{filename}"))
