from scipy.stats import spearmanr, pearsonr

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('save_name')
    args = parser.parse_args()
    return args 

args = get_args()

input_path = f"output/excel/{args.save_name}.xlsx"
save_table_path = "output/esmo/table"
save_figure_path = "output/esmo/figure"

dpi = 300
bins_ds = np.linspace(0.0, 1.0, 21) # 0.0-1.0の間を20分割

#データフレームにエクセルを変換
xls = pd.ExcelFile(input_path)
df = pd.read_excel(xls, sheet_name="Sheet1")

# ① 読み込み後 immediately 型変換
filtered_df = df[~df['shibaki_comments'].str.contains('delete', na=False)].copy()

# ② dice列だけ型を強制変換
filtered_df["dice"] = pd.to_numeric(filtered_df["dice"], errors="coerce")


# 必要な列だけ
df_box = filtered_df[["shibaki_cancer_type_label", "shibaki_organ_label", "dice", "MeanHD"]].dropna()




# medianを計算
organ_median = df_box.groupby("shibaki_organ_label")["dice"].median()

# Others を除外してソート（中央値が高い順）
organ_sorted = organ_median.drop(labels=["Others"], errors="ignore").sort_values(ascending=False).index.tolist()

# 最後に Others を追加
if "Others" in organ_median.index:
    organ_sorted.append("Others")

# これを unique_organ_sorted として使う
unique_organ_sorted = organ_sorted

# カテゴリ型に再設定
df_box["shibaki_organ_label"] = pd.Categorical(
    df_box["shibaki_organ_label"],
    categories=unique_organ_sorted,
    ordered=True
)


# 描画
fig, ax1 = plt.subplots(figsize=(15, 20))  # 横方向を広く

# DICE boxplot（X軸にスコア）
positions = range(len(unique_organ_sorted))
dice_data = [df_box[df_box["shibaki_organ_label"] == organ]["dice"].dropna() for organ in unique_organ_sorted]
bpl = ax1.boxplot(
    dice_data,
    positions=positions,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="#99C7C3"),
    showfliers=False,
    vert=False   # ← 横向きにする
)

ax1.set_xlabel("DICE Score", fontsize=20)
ax1.set_xlim(0, 1)
ax1.tick_params(axis='both', labelsize=20)

# Y軸をOrganラベルに
ax1.set_yticks(positions)
ax1.set_yticklabels(unique_organ_sorted, fontsize=25)

# 🔹 Diceスコアの補助線（0.2刻みなど）
for x in np.arange(0.2, 1.0, 0.2):
    ax1.axvline(x, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

# 🔹 Organごとの区切り線（最後は不要）
for p in positions[:-1]:
    ax1.axhline(p + 0.5, color="lightgray", linestyle="--", linewidth=0.7, alpha=0.7)

ax1.invert_yaxis()

plt.tight_layout()
plt.savefig(save_figure_path + "/boxplot_DICE_by_organ_horizontal.png", dpi=300)
plt.close()






cancer_type_median = df_box.groupby("shibaki_cancer_type_label")["dice"].median()
cancer_type_sorted = cancer_type_median.drop(labels=["Others"], errors="ignore").sort_values(ascending=False).index.tolist()

if "Others" in cancer_type_median.index:
    cancer_type_sorted.append("Others")

unique_cancer_type_sorted = cancer_type_sorted

df_box["shibaki_cancer_type_label"] = pd.Categorical(
    df_box["shibaki_cancer_type_label"],
    categories=unique_cancer_type_sorted,
    ordered=True
)



fig, ax1 = plt.subplots(figsize=(15, 20))

positions = range(len(unique_cancer_type_sorted))
dice_data = [df_box[df_box["shibaki_cancer_type_label"] == cancer_type]["dice"].dropna() for cancer_type in unique_cancer_type_sorted]

bpl = ax1.boxplot(
    dice_data,
    positions=positions,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="#99C7C3"),
    showfliers=False,
    vert=False
)

# X軸 = Dice
ax1.set_xlabel("DICE Score", fontsize=20)
ax1.set_xlim(0, 1)
ax1.tick_params(axis='both', labelsize=20)

# Y軸 = Cancer type
ax1.set_yticks(positions)
ax1.set_yticklabels(unique_cancer_type_sorted, fontsize=25)

# 🔹 Diceスコアの補助線
for x in np.arange(0.2, 1.0, 0.2):
    ax1.axvline(x, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

# 🔹 Cancer typeごとの区切り線
for p in positions[:-1]:
    ax1.axhline(p + 0.5, color="lightgray", linestyle="--", linewidth=0.7, alpha=0.5)

ax1.invert_yaxis()

plt.tight_layout()
plt.savefig(save_figure_path + "/boxplot_DICE_by_cancer_type_horizontal.png", dpi=300)
plt.close()


# 以降は positions = range(len(unique_cancer_type_sorted)) を使って描画


# # 描画
# fig, ax1 = plt.subplots(figsize=(30, 10))

# # DICE boxplot（左Y軸）
# positions = range(len(unique_organ_sorted))
# dice_data = [df_box[df_box["shibaki_organ_label"] == organ]["dice"].dropna() for organ in unique_organ_sorted]
# bpl = ax1.boxplot(
#     dice_data,
#     positions=positions, 
#     widths=0.5, 
#     patch_artist=True,
#     boxprops=dict(facecolor="#99C7C3"), 
#     showfliers=False,
#     # whiskerprops=dict(linestyle="none"),
#     # capprops=dict(linestyle='none'),
# )
# ax1.set_ylabel("DICE Score", fontsize=20)
# ax1.set_ylim(0, 1)
# ax1.tick_params(axis='both', labelsize=20) 

# # X軸
# ax1.set_xticks(positions)
# ax1.set_xticklabels(unique_organ_sorted, rotation=-60, fontsize=25)

# # cancer type ごとに区切り線を追加（最後のカテゴリの右端は不要）
# for p in positions[:-1]:
#     ax1.axvline(p+0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

# # # タイトル
# # plt.title("Box plot of DICE and Mean HD by Organ", fontsize=20, fontweight="bold")

# # # 凡例
# # dice_patch = mpatches.Patch(color="#99C7C3", label='DICE')
# # mhd_patch = mpatches.Patch(color="#F4DF6E", label='Mean HD')
# # plt.legend(handles=[dice_patch, mhd_patch], loc="upper right")

# plt.tight_layout()
# plt.savefig(save_figure_path + "/boxplot_DICE_MHD_by_organ.png", dpi=300)
# plt.close()








# # cancer type を多い順に並べる
# cancer_type_counts = df_box["shibaki_cancer_type_label"].value_counts()
# unique_cancer_type_sorted = cancer_type_counts.index.tolist()

# # cancer type を並び順でカテゴリ型にする
# df_box["shibaki_cancer_type_label"] = pd.Categorical(df_box["shibaki_cancer_type_label"], categories=unique_cancer_type_sorted, ordered=True)

# # 描画
# fig, ax1 = plt.subplots(figsize=(30, 10))

# # DICE boxplot（左Y軸）
# positions = range(len(unique_cancer_type_sorted))
# dice_data = [df_box[df_box["shibaki_cancer_type_label"] == cancer_type]["dice"].dropna() for cancer_type in unique_cancer_type_sorted]
# bpl = ax1.boxplot(
#     dice_data,
#     positions=positions,
#     widths=0.35,
#     patch_artist=True,
#     boxprops=dict(facecolor="#99C7C3"),
#     showfliers=False,
#     # whiskerprops=dict(linestyle="none"),
#     # capprops=dict(linestyle='none'),
# )
# ax1.set_ylabel("DICE Score", fontsize=20)
# ax1.set_ylim(0, 1)
# ax1.tick_params(axis='both', labelsize=20) 

# # X軸
# ax1.set_xticks([p for p in positions])
# ax1.set_xticklabels(unique_cancer_type_sorted, rotation=-60, fontsize=25)

# # # タイトル
# # plt.title("Box plot of DICE and Mean HD by Cancer type", fontsize=20, fontweight="bold")

# # cancer type ごとに区切り線を追加（最後のカテゴリの右端は不要）
# for p in positions[:-1]:
#     ax1.axvline(p + 0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)


# # # 凡例
# # import matplotlib.patches as mpatches
# # dice_patch = mpatches.Patch(color="#99C7C3", label='DICE')
# # mhd_patch = mpatches.Patch(color="#F4DF6E", label='Mean HD')
# # plt.legend(handles=[dice_patch, mhd_patch], loc="upper right")

# plt.tight_layout()
# plt.savefig(save_figure_path + "/boxplot_DICE_MHD_by_cancer_type.png", dpi=300)
# plt.close()
