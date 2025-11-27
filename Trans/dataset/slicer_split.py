import numpy as np
import torch
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import os
import cv2
import math
import argparse
import time


def extract_slice(img, size, spacing, origin, direct, method, default_value=-1000):
    """
    根据给定参数创建切片. 其中method控制插值方式(NN: 最近邻, Linear: 线性插值).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetSize((size[0], size[1], 1))
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direct.reshape(-1))
    resampler.SetOutputSpacing((spacing[0], spacing[1], 1))
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    if (method.lower() == "linear"):
        resampler.SetInterpolator(sitk.sitkLinear)
    if (method.lower() == "nn"):
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    output_image = resampler.Execute(img)
    return sitk.Extract(output_image, (size[0], size[1], 0), (0, 0, 0))

def slice_dir(args, output_dir):
    args.time = time.localtime()

    # --------------- 初始化保存路径 --------------- #
    

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "log.txt"), "w", encoding="UTF-8") as fp:
        print(args, file=fp)
    # --------------- 初始化保存路径 --------------- #

    # --------------- 读取影像数据 --------------- #
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(args.ct)
    reader.SetFileNames(dcm_series)
    ct = reader.Execute()

    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(args.mri)
    reader.SetFileNames(dcm_series)
    mri = reader.Execute()

    size = mri.GetSize()

    if size != (512, 512, 9):
        # raise ValueError(f"Unexpected MRI size: {size}")
        print(args.mri)
        return 0
    # else:
    #     return 0

    spacing = mri.GetSpacing()
    origin = mri.GetOrigin()
    direct = np.array(mri.GetDirection()).reshape(3, 3)
    # --------------- 读取影像数据 --------------- #

    # --------------- 计算切片范围和间距 --------------- #
    # 如果 不给定步长 或 给定步长为负数 或 大于影像原有间隔, 则将其设置为影像原有间隔.
    # 步长默认为正数, 且应当恰好等分影像.
    input_step = args.step
    if (args.step <= 0.0):
        args.step = spacing[2]
    args.step = min(spacing[2], args.step)
    num = round(spacing[2] * (size[2] - 1) / args.step)
    args.step = spacing[2] * (size[2] - 1) / num
    print(f"Step: {input_step}->{args.step}.")

    # 根据切片算法, 影像两端spacing[2] / 2.0的范围内也可以获得切片, 因此可以加宽范围.
    args.begin = -spacing[2] * (size[2] - 1) - args.step * (math.ceil(spacing[2] / (2.0 * args.step)) - 1)
    args.end = args.step * (math.ceil(spacing[2] / (2.0 * args.step)) - 1)

    print(f"Range: [{args.begin}, {args.end}]; Step: {input_step}->{args.step}.")
    print(f"Size: [{size[0] * spacing[0]}, {size[1] * spacing[1]}].")
    # --------------- 计算切片范围和间距 --------------- #

    img_dir = os.path.join(output_dir, "img")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)


    # --------------- 生成切片 --------------- #
    number = 0
    for offset in tqdm(np.linspace(args.begin, args.end, num=(1 + round((args.end - args.begin) / args.step)))):
        slice_origin = origin - direct @ np.array([0.0, 0.0, offset])

        ct_img = extract_slice(ct, size, spacing, slice_origin, direct, args.method, default_value=-1000)
        mri_img = extract_slice(mri, size, spacing, slice_origin, direct, args.method, default_value=0)
        ct_pix_raw = sitk.GetArrayViewFromImage(ct_img)
        mri_pix_raw = sitk.GetArrayViewFromImage(mri_img)
        ct_pix = ct_pix_raw.astype(np.float32)
        mri_pix = mri_pix_raw.astype(np.float32)
        ct_pix /= 1024
        mri_pix /= 2047
        ct_pix = np.clip(ct_pix * 255, 0, 255).astype(np.uint8)
        mri_pix = np.clip(mri_pix * 255, 0, 255).astype(np.uint8)

        merged = np.hstack((ct_pix, mri_pix))
        merged_rgb = cv2.cvtColor(merged[..., None], cv2.COLOR_GRAY2RGB)
        RGB_blend = np.zeros((ct_pix.shape[0], ct_pix.shape[1], 3), dtype=np.uint8)
        RGB_blend[:, :, 0] = ct_pix
        RGB_blend[:, :, 2] = mri_pix
        total = np.hstack((merged_rgb, RGB_blend))
        cv2.imwrite(os.path.join(img_dir, f"merged_{number:03d}.png"), total)
        np.save(os.path.join(data_dir, f"ct_{number:03d}.npy"), ct_pix_raw)
        np.save(os.path.join(data_dir, f"mri_{number:03d}.npy"), mri_pix_raw)

        number += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--step", type=float, default=1.0)  # 切片的间距, 当小于等于0或大于影像间距时将会设置为影像间距.

    parser.add_argument("--ct", type=str, default="data/配对后3D数据/lxl/CT")  # CT数据文件夹路径.
    parser.add_argument("--mri", type=str, default="data/配对后3D数据/lxl/MRI-L")  # MRI数据文件夹路径.

    parser.add_argument("--method", type=str, default="Linear")  # 切片时的插值方式(NN: 最近邻, Linear: 线性插值(默认)).

    args = parser.parse_args()

    seq_dir = Path("data510_raw")
    out_dir = Path("data510/all")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # all_seqs = sorted([x for x in seq_dir.iterdir() if x.is_dir()])
    all_seqs = []
    for seq1 in sorted(seq_dir.glob("*")):
        if seq1.is_dir():
            for seq2 in seq1.glob("*"):
                if seq2.is_dir():
                    for seq3 in seq2.glob("*"):
                        if seq3.is_dir():
                            all_seqs.append(seq3)
                        

    seq = ""
    cnt = 0
    for seq in tqdm(all_seqs, desc="Processing sequences", postfix=str(seq)):
        ct_dir = seq / "CT"
        Lmri_dir = seq / "MRI-L"
        Rmri_dir = seq / "MRI-R"

        ct_dir = str(ct_dir)
        Lmri_dir = str(Lmri_dir)
        Rmri_dir = str(Rmri_dir)
        args.ct = ct_dir
        args.mri = Lmri_dir
        output_dir = out_dir / (seq.parent.parent.name+"_"+seq.parent.name+"_"+seq.name + "_L")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        slice_dir(args, output_dir)
        args.ct = ct_dir
        args.mri = Rmri_dir
        output_dir = out_dir / (seq.parent.parent.name+"_"+seq.parent.name+"_"+seq.name + "_R")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        slice_dir(args, output_dir)
        print(f"Finished {seq.name}!")
    