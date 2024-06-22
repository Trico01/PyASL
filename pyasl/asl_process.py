import os
import numpy as np
import nibabel as nib
import json
from nipype.interfaces import spm
from nipype.interfaces.matlab import MatlabCommand


def read_data_description(root):
    description_file = os.path.join(root, "data_description.json")

    try:
        with open(description_file, "r") as f:
            data_descrip = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"{description_file} not found. Please load data first.")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from {description_file}.")

    return data_descrip


def create_derivatives_folders(data_descrip):
    for key, value in data_descrip["Images"].items():
        der_path = key.replace("rawdata", "derivatives")
        try:
            os.makedirs(der_path, exist_ok=True)
        except OSError:
            raise OSError(f"Could not create directories: {der_path}.")

        der_perf_path = os.path.join(der_path, "perf")
        try:
            os.mkdir(der_perf_path)
        except OSError:
            raise OSError(f"Could not create directory: {der_perf_path}.")

        if value["anat"]:
            der_anat_path = os.path.join(der_path, "anat")
            try:
                os.mkdir(der_anat_path)
            except OSError:
                raise OSError(f"Could not create directory: {der_anat_path}.")


def img_rescale(source_path, target_path):
    img = nib.load(source_path)
    header = img.header
    data = img.get_fdata()

    slope, inter = header.get_slope_inter()
    ss = slope if slope is not None else 1.0
    data = np.nan_to_num(data)
    data = data / ss / ss

    rescaled_img = nib.Nifti1Image(data, img.affine, img.header)
    rescaled_img.header["desrip"] = "4D rescaled images"
    rescaled_img.set_data_dtype(np.float32)
    rescaled_img.header["scl_slope"] = 1
    rescaled_img.header["scl_inter"] = 0

    nib.save(rescaled_img, target_path)


def asl_rescale(data_descrip):
    for key, value in data_descrip["Images"].items():
        for asl_file in value["asl"]:
            asl_path = os.path.join(key, asl_file)
            asl_der_path = asl_path.replace("rawdata", "derivatives")
            img_rescale(asl_path, asl_der_path)

        if value["M0"]:
            m0_path = os.path.join(key, value["M0"])
            m0_der_path = m0_path.replace("rawdata", "derivatives")
            img_rescale(m0_path, m0_der_path)


def asl_realign(data_descrip):
    print("pyasl: Realign ASL data...")

    for key, value in data_descrip["Images"].items():
        key = key.replace("rawdata", "derivatives")
        for asl_file in value["asl"]:
            P = os.path.join(key, asl_file)

            realign = spm.Realign()
            realign.inputs.in_files = P
            realign.inputs.quality = 0.9  # SPM default
            realign.inputs.fwhm = 5
            realign.inputs.register_to_mean = False  # realign to the first timepoint
            realign_result = realign.run()  # realign_result not necessary
            print("Realign generated files:", realign_result.outputs)

            reslice = spm.Reslice()
            reslice.inputs.in_files = P
            reslice.inputs.interp = 4  # SPM default
            reslice.inputs.wrap = [0, 0, 0]  # SPM default
            reslice.inputs.mask = True
            reslice.inputs.write_which = [
                2,
                1,
            ]  # which_writerealign = 2, mean_writerealign = 1
            reslice_result = reslice.run()  # reslice_result not necessary
            print("Reslice generated files:", reslice_result.outputs)


def asl_calculate_diffmap(data_descrip):
    print("pyasl: Calculate difference volume...")

    for key, value in data_descrip["Images"].items():
        key = key.replace("rawdata", "derivatives")
        for asl_file in value["asl"]:
            file_name = os.path.basename(asl_file)
            file_name, _ = os.path.splitext(file_name)
            if data_descrip["SingleDelay"]:
                P = os.path.join(key, "perf", f"r{file_name}.nii")  # check
            else:
                P = os.path.join(key, asl_file)
            img = nib.load(P)
            data = img.get_fdata()
            data = np.nan_to_num(data)
            num_pairs = 0
            ctrl = np.zeros(data.shape[:3])
            labl = np.zeros(data.shape[:3])
            for i, volume_type in enumerate(data_descrip["ASLContext"]):
                if volume_type == "label":
                    labl += data[:, :, :, i]
                    num_pairs += 1
                elif volume_type == "control":
                    ctrl += data[:, :, :, i]
            ctrl /= num_pairs
            labl /= num_pairs
            diff = ctrl - labl

            affine = img.affine
            header = img.header
            header.set_data_dtype(np.float32)
            ctrl_img = nib.Nifti1Image(ctrl, affine, header)
            labl_img = nib.Nifti1Image(labl, affine, header)
            diff_img = nib.Nifti1Image(diff, affine, header)

            ctrl_img.to_filename(os.path.join(key, "perf", f"r{file_name}_ctrl.nii"))
            labl_img.to_filename(os.path.join(key, "perf", f"r{file_name}_labl.nii"))
            diff_img.to_filename(os.path.join(key, "perf", f"r{file_name}_diff.nii"))


def asl_pipeline(root):
    data_descrip = read_data_description(root)
    create_derivatives_folders(data_descrip)
    asl_rescale(data_descrip)

    # check important
    MatlabCommand.set_default_matlab_cmd("matlab")
    MatlabCommand.set_default_paths("E:/toolbox/spm12/spm12")

    if data_descrip["SingleDelay"]:
        asl_realign(data_descrip)
        asl_calculate_diffmap(data_descrip)
    else:
        None
