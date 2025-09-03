import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

save_figure_path = "output/figure"
dpi = 300

def aug_viewer(type, target, file_path):
    
    df = pd.read_pickle(file_path)  # dict型として読み込まれます

    # 2. scaleaugデータを集める
    angles = []
    dice_scores = []

    for entry in df.values():
        scaleaug_data = entry.get(f'{type}', {})
        for angle, score in scaleaug_data.items():
            angles.append(angle)
            dice_scores.append(score)

    # 3. DataFrameに変換
    data = pd.DataFrame({
        'angle': angles,
        'dice': dice_scores
    })

    # # 4. バイオリンプロット
    # plt.figure(figsize=(12, 6))
    # sns.violinplot(x='angle', y='dice', data=data, density_norm='width', inner='quartile')
    # plt.xlabel("RC Scale", fontsize=12, fontname="Arial")
    # plt.xticks(fontsize=10, fontname="Arial")
    # plt.ylabel("Dice score", fontsize=12, fontname="Arial")
    # plt.yticks(fontsize=10, fontname="Arial")
    # plt.title(f"Violin Plot of {type}", fontsize=16, fontname="Arial", fontweight="bold")
    # plt.xticks(rotation=90)
    # plt.grid(True)
    # plt.tight_layout()

    # plt.savefig(save_figure_path+f"/{target}_{type}_violinplot.png", dpi=dpi) 
    # plt.close()

    # 5. 折れ線グラフ（中央値と四分位数）
    summary = data.groupby('angle')['dice'].quantile([0.25, 0.5, 0.75]).unstack()

    plt.figure(figsize=(12, 6))
    plt.plot(summary.index, summary[0.5], label='Median', marker='o')
    plt.plot(summary.index, summary[0.25], label='Q1', linestyle='--')
    plt.plot(summary.index, summary[0.75], label='Q3', linestyle='--')
    plt.xlabel("RC Scale", fontsize=12, fontname="Arial")
    plt.xticks(fontsize=10, fontname="Arial")
    plt.ylabel("Dice score", fontsize=12, fontname="Arial")
    plt.ylim(0.0, 1.0)
    plt.yticks(fontsize=10, fontname="Arial")
    plt.title(f"Line Graph of {type}", fontsize=16, fontname="Arial", fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # << 追加 >> 
    plt.ticklabel_format(style='plain', axis='y')
    
    plt.savefig(save_figure_path+f"/{target}_{type}_linegraph.png", dpi=dpi) 
    plt.close()

aug_viewer("scaleaug", "pred", "output/pickle/aug_results_pred_250707.pickle")
aug_viewer("rotaug", "pred", "output/pickle/aug_results_pred_250707.pickle")
aug_viewer("shiftaug", "pred", "output/pickle/aug_results_pred_250707.pickle")

aug_viewer("scaleaug", "gt", "output/pickle/aug_results_gt_250707.pickle")
aug_viewer("rotaug", "gt", "output/pickle/aug_results_gt_250707.pickle")
aug_viewer("shiftaug", "gt", "output/pickle/aug_results_gt_250707.pickle")