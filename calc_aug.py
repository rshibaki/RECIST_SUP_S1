from functools import partial
import multiprocessing
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import calc_dice, check_intersect, crop_input, extract_bb, merge_bbox


def fast_dice(gt_msk, pred_msk, gt_crop_zyxzyx):
    pred_crop_zyxzyx = extract_bb(pred_msk, to_open=True)
    if check_intersect(pred_crop_zyxzyx, gt_crop_zyxzyx):
        crop_zyxzyx = merge_bbox(pred_crop_zyxzyx, gt_crop_zyxzyx)
        _gt = crop_input(gt_msk, crop_zyxzyx)
        _pred = crop_input(pred_msk, crop_zyxzyx)
        dice = calc_dice(_gt, _pred)
    else:
        dice = 0.0
    return dice


def dice_aug_pred(no, folder="pred_npz"):
    data = np.load(f"{folder}/{no}.npz")
    gt_msk = data["pred"].astype(np.bool_)
    if gt_msk.sum() == 0:
        # 正解マスクが空
        return no, None

    gt_crop_zyxzyx = extract_bb(gt_msk, to_open=True)
    pred_msk = data["pred"].astype(np.bool_)
    if pred_msk.sum() == 0:
        # 予測マスクが空
        return no, None
    base_dice = fast_dice(gt_msk, pred_msk, gt_crop_zyxzyx)

    out_dict = defaultdict(dict)
    for (min_val, max_val), suffix in zip(
        [[-30, 30], [-45, 45], [-30, 30]],
        ["scaleaug", "rotaug", "shiftaug"],
    ):
        prev_score = None
        for aug in np.arange(min_val, max_val + 1, 1):
            if aug == 0:
                dice = base_dice
            else:
                npz_path = Path(f"{folder}/{no}.{suffix}.{aug}.npz")

                if not npz_path.exists():
                    # すでに計算済みのスコアがある場合はそちらを採用
                    dice = prev_score
                else:
                    pred_msk = np.load(npz_path)["pred"].astype(np.bool_)
                    if pred_msk.sum() == 0:
                        dice = 0.0
                    else:
                        dice = fast_dice(gt_msk, pred_msk, gt_crop_zyxzyx)
                        prev_score = dice
            out_dict[suffix][aug] = dice
    return no, out_dict


def dice_aug_gt(no, folder="pred_npz"):
    data = np.load(f"{folder}/{no}.npz")
    gt_msk = data["gt"].astype(np.bool_)
    if gt_msk.sum() == 0:
        # 正解マスクが空
        return no, None

    gt_crop_zyxzyx = extract_bb(gt_msk, to_open=True)
    pred_msk = data["gt"].astype(np.bool_)
    if pred_msk.sum() == 0:
        # 予測マスクが空
        return no, None
    base_dice = fast_dice(gt_msk, pred_msk, gt_crop_zyxzyx)

    out_dict = defaultdict(dict)
    for (min_val, max_val), suffix in zip(
        [[-30, 30], [-45, 45], [-30, 30]],
        ["scaleaug", "rotaug", "shiftaug"],
    ):
        prev_score = None
        for aug in np.arange(min_val, max_val + 1, 1):
            if aug == 0:
                dice = base_dice
            else:
                npz_path = Path(f"{folder}/{no}.{suffix}.{aug}.npz")

                if not npz_path.exists():
                    # すでに計算済みのスコアがある場合はそちらを採用
                    dice = prev_score
                else:
                    pred_msk = np.load(npz_path)["gt"].astype(np.bool_)
                    if pred_msk.sum() == 0:
                        dice = 0.0
                    else:
                        dice = fast_dice(gt_msk, pred_msk, gt_crop_zyxzyx)
                        prev_score = dice
            out_dict[suffix][aug] = dice
    return no, out_dict


def to_dict(d: defaultdict):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = to_dict(v)
    return dict(d)


def main():
    folder = "data/pred_aug_npz"
    save_path_pred = Path("output/pickle/aug_results_pred_250707.pickle")
    save_path_gt = Path("output/pickle/aug_results_gt_250707.pickle")
    df_list = pd.read_excel("data/excel/data_result_250422.xlsx", sheet_name="Sheet1")
    filtered_df_list = df_list[~df_list["shibaki_comments"].astype(str).str.contains("delete", case=False, na=False)]
    filtered_df_list = filtered_df_list["No"].tolist()
    
    ########## pred ###########
    if save_path_pred.exists():
        print(f"{save_path_pred} already exists.")
    else:
        no_list = filtered_df_list
    #    no_list = list(range(1, 31))
        num_workers = 8

        out_dict = dict()
        aug_func = partial(dice_aug_pred, folder=folder)
        
        with multiprocessing.Pool(num_workers) as pool:
            for no, _dict in tqdm(
                pool.imap_unordered(aug_func, no_list), total=len(no_list)
            ):
                if _dict is None:
                    print(f"empty gt or pred: {no}")
                    continue
                out_dict[no] = _dict

        dice_dict = to_dict(out_dict)
        with open(save_path_pred, "wb") as f:
            pickle.dump(dice_dict, f)
        
    
        ########## gt ##########
    if save_path_gt.exists():
        print(f"{save_path_gt} already exists.")
    else:
        no_list = filtered_df_list
    #    no_list = list(range(1, 31))
        num_workers = 8

        out_dict = dict()
        aug_func = partial(dice_aug_gt, folder=folder)
        
        with multiprocessing.Pool(num_workers) as pool:
            for no, _dict in tqdm(
                pool.imap_unordered(aug_func, no_list), total=len(no_list)
            ):
                if _dict is None:
                    print(f"empty gt or pred: {no}")
                    continue
                out_dict[no] = _dict

        dice_dict = to_dict(out_dict)
        with open(save_path_gt, "wb") as f:
            pickle.dump(dice_dict, f)


if __name__ == "__main__":
    main()
#    plot_example()





