import os
import numpy as np
import pandas as pd
import monai
import torch
from glob import glob
from natsort import natsorted
# from monai.metrics import compute_hausdorff_distance
from utils.calculate import calc_dice, compute_hausdorff_distance
from utils import one_hot
from tqdm import tqdm
import time


from argparse import ArgumentParser


ex_excel_path = "data/excel/data_result_250422.xlsx"


def get_args():
    parser = ArgumentParser()
    parser.add_argument('save_name')
    parser.add_argument('-ext', default='.xlsx')
    parser.add_argument(
        '-save_dir', 
        default='output/excel')
    parser.add_argument(
        '-npz_dir', 
        default='data/pred_npz')
    
    args = parser.parse_args()
    return args 

def main(args):
    print(os.getcwd())
    save_name = args.save_name
    ext = args.ext
    save_dir = args.save_dir
    npz_dir = args.npz_dir
    
    npz_paths = natsorted(
        glob(os.path.join(npz_dir, '*.npz'))
        )
    
    metrics = {}
    error_cases = []
    
    for path in tqdm(npz_paths):
        case_num = path.split('/')[-1].split('.')[0]
        
        try:
            data = np.load(path)
            pred = data["pred"]
            gt = data["gt"]
            spacing = data["spacing_zyx"].tolist()
            
            # dice計算
            dsc = calc_dice(gt, pred)
            
            
            pred_t = torch.tensor(pred.astype(int)).unsqueeze(0)
            gt_t = torch.tensor(gt.astype(int)).unsqueeze(0)
            
            pred_oh = one_hot(pred_t.unsqueeze(0), num_classes=2, dim=1)
            gt_oh = one_hot(gt_t.unsqueeze(0), num_classes=2, dim=1)
            
            hd = compute_hausdorff_distance(pred_oh, gt_oh, spacing=spacing, mode='max').item()
            hd95 = compute_hausdorff_distance(
                pred_oh, gt_oh, 
                spacing=spacing, 
                mode='percentile', percentile=95.).item()
            
            md = compute_hausdorff_distance(
                pred_oh, gt_oh,
                spacing=spacing,
                mode='mean'
            ).item()
            
            metrics[case_num] = {
                'DSC': float(dsc), 
                'HD': float(hd), 
                'HD95': float(hd95), 
                'MeanHD': float(md),
                }
        
        except Exception as e:
            print(f"Error occurred at case: {case_num}")
            print(f"Exception message: {e}")
            error_cases.append(case_num)
            continue  # 次のファイルに進む
        
    # ② metricsをDataFrameに変換
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.index.name = 'No'
    df_metrics.reset_index(inplace=True)  # No列を作る

    # ③ 既存Excel読み込み
    df_existing = pd.read_excel(ex_excel_path)
    
    df_existing['No'] = df_existing['No'].astype(str)

    # ④ マージ（left join）
    df_merged = pd.merge(df_existing, df_metrics, how='left', on='No')

    # ⑤ 保存
    file_name = save_name + ext
    save_path = os.path.join(save_dir, file_name)
    df_merged.to_excel(save_path, index=False)

if __name__ == '__main__':
    main(get_args())