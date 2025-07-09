import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def aug_viewer(type, file_path):
    
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

    # 4. バイオリンプロット
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='angle', y='dice', data=data, scale='width', inner='quartile')
    plt.xlabel("RC Scale")
    plt.ylabel("Dice score")
    plt.title(f"Violin Plot of {type}")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. 折れ線グラフ（中央値と四分位数）
    summary = data.groupby('angle')['dice'].quantile([0.25, 0.5, 0.75]).unstack()

    plt.figure(figsize=(12, 6))
    plt.plot(summary.index, summary[0.5], label='Median', marker='o')
    plt.plot(summary.index, summary[0.25], label='Q1', linestyle='--')
    plt.plot(summary.index, summary[0.75], label='Q3', linestyle='--')
    plt.xlabel("RC Scale")
    plt.ylabel("Dice score")
    plt.title(f"Line Graph of {type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(len(data))

