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
save_table_path = "output/table"
save_figure_path = "output/figure"

dpi = 300
bins_ds = np.linspace(0.0, 1.0, 21) # 0.0-1.0の間を20分割

#データフレームにエクセルを変換
xls = pd.ExcelFile(input_path)
df = pd.read_excel(xls, sheet_name="Sheet1")

# ① 読み込み後 immediately 型変換
filtered_df = df[~df['shibaki_comments'].str.contains('delete', na=False)].copy()

# ② dice列だけ型を強制変換
filtered_df["dice"] = pd.to_numeric(filtered_df["dice"], errors="coerce")

# Dice_allヒストグラムを作成と保存
plt.figure(figsize=(8, 5))  # これはインチサイズ
plt.hist(filtered_df["dice"].dropna(), bins=bins_ds, density=False, edgecolor="black", color="#99C7C3")
plt.title(f"DICE score of all lesions (N = {filtered_df.shape[0]})", fontsize=16, fontname="Arial", fontweight="bold")
plt.xlabel("DICE Score", fontsize=12, fontname="Arial")
plt.xlim(0.0, 1.0)
plt.xticks(fontsize=10, fontname="Arial")
plt.ylabel("Frequency", fontsize=12, fontname="Arial")
plt.yticks(fontsize=10, fontname="Arial")
plt.grid(True, axis="y")

plt.savefig(save_figure_path+"/DS_all.png", dpi=dpi) 
plt.close()

#diceの記述統計
dice_stats = filtered_df["dice"].dropna().describe()

dice_summary_data = {
    "Organ": ["All Lesions"],
    "Count": [filtered_df["dice"].dropna().shape[0]],
    "Mean": [dice_stats["mean"]],
    "Median": [filtered_df["dice"].dropna().median()],
    "Q1": [dice_stats["25%"]],
    "Q3": [dice_stats["75%"]],
}
dice_summary_df = pd.DataFrame(dice_summary_data)


########## By organのFigure ##########
# ラベルごとの件数順で並べ替え
organ_counts = filtered_df["shibaki_organ_label"].value_counts()
unique_organ_sorted = organ_counts.index.tolist()

fig, axs = plt.subplots(3, 5, figsize=(35, 15))
axs = axs.flatten()

for i, label in enumerate(unique_organ_sorted):
    subset = filtered_df[filtered_df["shibaki_organ_label"] == label]
    count = len(subset)
    axs[i].hist(subset["dice"].dropna(), bins=bins_ds, edgecolor='black', color="#99C7C3")
    axs[i].set_title(f"DICE Score: {label} (N = {count})", fontname="Arial", fontweight="bold")
    axs[i].set_xlabel("DICE Score", fontname="Arial")
    axs[i].set_ylabel("Frequency", fontname="Arial")
    axs[i].set_xlim(0.0, 1.0)
    axs[i].grid(True, axis="y")

for j in range(len(unique_organ_sorted), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig(save_figure_path+"/DS_organ", dpi=dpi)
plt.close()

# Organ別にまとめた記述統計テーブルを作成
organ_summary = []

for label in unique_organ_sorted:
    subset = filtered_df[filtered_df["shibaki_organ_label"] == label]["dice"].dropna()
    organ_summary.append({
        "Organ": label,
        "Count": subset.shape[0],
        "Mean": subset.mean(),
        "Median": subset.median(),
        "Q1": subset.quantile(0.25),
        "Q3": subset.quantile(0.75),
    })
    
# DataFrame化
organ_summary_df = pd.DataFrame(organ_summary)

# 空の1行
empty_row = pd.DataFrame([[""] * dice_summary_df.shape[1]], columns=dice_summary_df.columns)


########### By_cancer_typesのFigure ##########
# ラベルごとの件数順で並べ替え
filtered_df_type = filtered_df[filtered_df["shibaki_cancer_type_label"] != "OTH"]
type_counts = filtered_df_type["shibaki_cancer_type_label"].value_counts()
unique_type_sorted = type_counts.index.tolist()

fig, axs = plt.subplots(4, 6, figsize=(32, 15))
axs = axs.flatten()

for i, label in enumerate(unique_type_sorted):
    subset = filtered_df[filtered_df["shibaki_cancer_type_label"] == label]
    count = len(subset)
    axs[i].hist(subset["dice"].dropna(), bins=bins_ds, edgecolor='black', color="#99C7C3")
    axs[i].set_title(f"DICE Score: {label} (N = {count})", fontname="Arial", fontweight="bold")
    axs[i].set_xlabel("DICE Score", fontname="Arial")
    axs[i].set_ylabel("Frequency", fontname="Arial")
    axs[i].set_xlim(0.0, 1.0)
    axs[i].grid(True, axis="y")
    
plt.tight_layout(pad=0.5)

plt.savefig(save_figure_path+"/DS_cancer_types", dpi=dpi) 

# cancer type別にまとめた記述統計テーブルを作成
type_summary = []

for label in unique_type_sorted:
    subset = filtered_df[filtered_df["shibaki_cancer_type_label"] == label]["dice"].dropna()
    type_summary.append({
        "Organ": label,
        "Count": subset.shape[0],
        "Mean": subset.mean(),
        "Median": subset.median(),
        "Q1": subset.quantile(0.25),
        "Q3": subset.quantile(0.75),
    })

# DataFrame化
type_summary_df = pd.DataFrame(type_summary)

# 結合
merged_df = pd.concat([dice_summary_df, empty_row, organ_summary_df, empty_row, type_summary_df], ignore_index=True)
# 数値列をまとめて四捨五入
for col in ["Mean", "Median", "Q1", "Q3"]:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").round(2)

# 保存
merged_df.to_csv(save_table_path + "/DS_summary.csv", index=False, encoding="utf-8-sig")


# # By sizeのFigure
# # ヒストグラム用の距離グループを定義
# size_bins = [0, 5, 10, 20, float('inf')]
# labels = ['<5mm', '5-10mm', '10-20mm', '20mm+']
# filtered_df['Distance Group'] = pd.cut(filtered_df["Distance (mm)"], bins=size_bins, labels=labels, right=False)

# # 各グループの件数をカウント
# size_counts = filtered_df['Distance Group'].value_counts().reindex(labels)
# unique_size_sorted = size_counts.index.tolist()

# fig, axs = plt.subplots(2,2, figsize=(18, 10))
# axs = axs.flatten()

# for i, label in enumerate(unique_size_sorted):
#     subset = filtered_df[filtered_df["Distance Group"] == label]
#     count = len(subset)
#     axs[i].hist(subset["dice"].dropna(), bins=np.linspace(0,1.0,21), edgecolor='black', color="#99C7C3")
#     axs[i].set_title(f"DICE Score Histogram: {label} (n={count})")
#     axs[i].set_xlabel("DICE Score")
#     axs[i].set_ylabel("Frequency")
#     axs[i].set_xlim(0, 1.0)
#     axs[i].grid(True)

# plt.savefig(save_figure_path+"/DS_size", dpi=dpi)  # ここが保存！

# # size別にまとめた記述統計テーブルを作成
# size_summary = []

# for label in unique_size_sorted:
#     subset = filtered_df[filtered_df["Distance Group"] == label]["dice"].dropna()
#     size_summary.append({
#         "Organ": label,
#         "Count": subset.shape[0],
#         "Mean": subset.mean(),
#         "Median": subset.median(),
#         "Q1": subset.quantile(0.25),
#         "Q3": subset.quantile(0.75),
#     })

# # DataFrame化
# size_summary_df = pd.DataFrame(size_summary)

# # 保存
# size_summary_df.to_csv(save_table_path + "/DS_size.csv", index=False, encoding="utf-8-sig")


########## MHD ##########
# データのコピーを作成し、5以上を5に丸める
mean_hd = filtered_df["MeanHD"].dropna().copy()
mean_hd_capped = mean_hd.clip(upper=5).dropna()  # 5以上の値を5にする
bins_mhd = np.concatenate([np.linspace(0, 5, 11), [5.5]])#ヒストグラムのビン（例：0～5を10分割し、最後のビンが5以上をカバー）

# MHDヒストグラムを作成と保存
plt.figure(figsize=(8, 5))
plt.hist(mean_hd_capped, bins=bins_mhd, density=False, edgecolor="black", color="#F4DF6E")
plt.title(f"Mean Hausdorff Distance of all lesions (N = {filtered_df.shape[0]})", fontsize=16, fontname="Arial", fontweight="bold")
plt.xlabel("Mean Hausdorff Distance", fontsize=12, fontname="Arial")
plt.ylabel("Frequency", fontsize=12, fontname="Arial")
plt.xticks(fontsize=10, fontname="Arial")
plt.yticks(fontsize=10, fontname="Arial")
plt.grid(True, axis="y")

# X軸目盛をシンプルに
tick_positions = [0, 1, 2, 3, 4, 5, 5.5]
tick_labels = ["0", "1.0", "2.0", "3.0", "4.0", "5.0", "≥5.0"]
plt.xticks(tick_positions, tick_labels, rotation=0)

plt.savefig(save_figure_path+"/MHD_all.png", dpi=dpi)
plt.close()

# MHD全体のまとめた記述統計テーブルを作成
MHD_summary = filtered_df["MeanHD"].dropna()

summary = {
    "Organ": ["All Lesions"],
    "Count": MHD_summary.shape[0],
    "Mean": MHD_summary.mean(),
    "Median": MHD_summary.median(),
    "Min": MHD_summary.quantile(0.0),
    "Q1": MHD_summary.quantile(0.25),
    "Q3": MHD_summary.quantile(0.75),
    "Max": MHD_summary.quantile(1.00),
    }

# DataFrame化
MHD_summary_df = pd.DataFrame(summary)


########## By organのFigure ##########
# ラベルごとの件数順で並べ替え
fig, axs = plt.subplots(3, 5, figsize=(35, 15))
axs = axs.flatten()

for i, label in enumerate(unique_organ_sorted):
    subset = filtered_df[filtered_df["shibaki_organ_label"] == label]
    count = len(subset)
    axs[i].hist(subset["MeanHD"].dropna(), bins=bins_mhd, edgecolor='black', color="#F4DF6E")
    axs[i].set_title(f"Mean Hausdorff Distance: {label} (N = {count})", fontname="Arial", fontweight="bold")
    axs[i].set_xlabel("Mean Hausdorff Distance", fontname="Arial")
    axs[i].set_ylabel("Frequency", fontname="Arial")
    axs[i].grid(True, axis="y")
    axs[i].set_xticks(tick_positions, tick_labels, rotation=0)

for j in range(len(unique_organ_sorted), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig(save_figure_path+"/MHD_organ", dpi=dpi)
plt.close()

# Organ別にまとめた記述統計テーブルを作成
organ_summary = []

for label in unique_organ_sorted:
    subset = filtered_df[filtered_df["shibaki_organ_label"] == label]["MeanHD"].dropna()
    organ_summary.append({
        "Organ": label,
        "Count": subset.shape[0],
        "Mean": subset.mean(),
        "Median": subset.median(),
        "Min": subset.quantile(0.0),
        "Q1": subset.quantile(0.25),
        "Q3": subset.quantile(0.75),
        "Max": subset.quantile(0.100)
    })

# DataFrame化
organ_summary_df = pd.DataFrame(organ_summary)


########### By_cancer_typesのFigure ##########
# ラベルごとの件数順で並べ替え
fig, axs = plt.subplots(4, 6, figsize=(32, 15))
axs = axs.flatten()

for i, label in enumerate(unique_type_sorted):
    subset = filtered_df[filtered_df["shibaki_cancer_type_label"] == label]
    count = len(subset)
    axs[i].hist(subset["MeanHD"].dropna(), bins=bins_mhd, edgecolor='black', color="#F4DF6E")
    axs[i].set_title(f"Mean Hausdorff Distance: {label} (N = {count})", fontname="Arial", fontweight="bold")
    axs[i].set_xlabel("Mean Hausdorff Distance", fontname="Arial")
    axs[i].set_ylabel("Frequency", fontname="Arial")
    axs[i].grid(True, axis="y")
    axs[i].set_xticks(tick_positions, tick_labels, rotation=0)

plt.tight_layout(pad=0.5)

plt.savefig(save_figure_path+"/MHD_cancer_types", dpi=dpi) 

# cancer type別にまとめた記述統計テーブルを作成
type_summary = []

for label in unique_type_sorted:
    subset = filtered_df[filtered_df["shibaki_cancer_type_label"] == label]["MeanHD"].dropna()
    type_summary.append({
        "Organ": label,
        "Count": subset.shape[0],
        "Mean": subset.mean(),
        "Median": subset.median(),
        "Min": subset.quantile(0.0),
        "Q1": subset.quantile(0.25),
        "Q3": subset.quantile(0.75),
        "Max": subset.quantile(0.100),
    })

# DataFrame化
type_summary_df = pd.DataFrame(type_summary)

# 結合
merged_MHD_df = pd.concat([MHD_summary_df, empty_row, organ_summary_df, empty_row, type_summary_df], ignore_index=True)
# 数値列をまとめて四捨五入
for col in ["Mean", "Median", "Min", "Q1", "Q3", "Max"]:
    merged_MHD_df[col] = pd.to_numeric(merged_MHD_df[col], errors="coerce").round(2)

# 保存
merged_MHD_df.to_csv(save_table_path + "/MHD_summary.csv", index=False, encoding="utf-8-sig")


# # By cancer typesのFigure
# # ラベルごとの件数順で並べ替え
# fig, axs = plt.subplots(3, 4, figsize=(32, 15))
# axs = axs.flatten()

# for i, label in enumerate(unique_type_sorted):
#     subset = filtered_df[filtered_df["shibaki_cancer_type_label"] == label]
#     count = len(subset)
#     axs[i].hist(subset["MeanHD"].dropna(), bins=bins, edgecolor='black', color="#F4DF6E")
#     axs[i].set_title(f"DICE Score Histogram: {label} (n={count})")
#     axs[i].set_xlabel("DICE Score")
#     axs[i].set_ylabel("Frequency")
#     axs[i].set_xlim(0, 5.0)
#     axs[i].grid(True)
#     # x軸ラベルの調整（最後のビンを「5+」と表示するには手動で調整する必要あり）
#     tick_labels = [f"{round(bins[i],1)}–{round(bins[i+1],1)}" for i in range(len(bins)-2)]
#     tick_labels.append("5+")
#     plt.xticks(np.linspace(0.25, 5.25, len(tick_labels)), tick_labels, rotation=45)

# plt.savefig(save_figure_path+"/MHD_cancer_types", dpi=dpi)  # ここが保存！

# # cancer type別にまとめた記述統計テーブルを作成
# type_summary = []

# for label in unique_type_sorted:
#     subset = filtered_df[filtered_df["shibaki_cancer_type_label"] == label]["MeanHD"].dropna()
#     type_summary.append({
#         "Organ": label,
#         "Count": subset.shape[0],
#         "Mean": subset.mean(),
#         "Median": subset.median(),
#         "Q1": subset.quantile(0.25),
#         "Q3": subset.quantile(0.75),
#     })

# # DataFrame化
# type_summary_df = pd.DataFrame(type_summary)

# # 保存
# type_summary_df.to_csv(save_table_path + "/MHD_cancer_type.csv", index=False, encoding="utf-8-sig")
# size_counts = filtered_df['Distance Group'].value_counts().reindex(labels)
# unique_size_sorted = size_counts.index.tolist()

# fig, axs = plt.subplots(2,2, figsize=(18, 10))
# axs = axs.flatten()

# for i, label in enumerate(unique_size_sorted):
#     subset = filtered_df[filtered_df["Distance Group"] == label]
#     count = len(subset)
#     axs[i].hist(subset["MeanHD"].dropna(), bins=bins, edgecolor='black', color="#F4DF6E")
#     axs[i].set_title(f"DICE Score Histogram: {label} (n={count})")
#     axs[i].set_xlabel("DICE Score")
#     axs[i].set_ylabel("Frequency")
#     axs[i].set_xlim(0, 5.0)
#     axs[i].grid(True)
#     # x軸ラベルの調整（最後のビンを「5+」と表示するには手動で調整する必要あり）
#     tick_labels = [f"{round(bins[i],1)}–{round(bins[i+1],1)}" for i in range(len(bins)-2)]
#     tick_labels.append("5+")
#     plt.xticks(np.linspace(0.25, 5.25, len(tick_labels)), tick_labels, rotation=45)

# plt.savefig(save_figure_path+"/MHD_size", dpi=dpi)  # ここが保存！

# # size別にまとめた記述統計テーブルを作成
# size_summary = []

# for label in unique_size_sorted:
#     subset = filtered_df[filtered_df["Distance Group"] == label]["MeanHD"].dropna()
#     size_summary.append({
#         "Organ": label,
#         "Count": subset.shape[0],
#         "Mean": subset.mean(),
#         "Median": subset.median(),
#         "Q1": subset.quantile(0.25),
#         "Q3": subset.quantile(0.75),
#     })

# # DataFrame化
# size_summary_df = pd.DataFrame(size_summary)

# # 保存
# size_summary_df.to_csv(save_table_path + "/MHD_size.csv", index=False, encoding="utf-8-sig")


# 必要な列だけ
df_box = filtered_df[["shibaki_cancer_type_label", "shibaki_organ_label", "dice", "MeanHD"]].dropna()

# Organ を多い順に並べる
organ_counts = df_box["shibaki_organ_label"].value_counts()
unique_organ_sorted = organ_counts.index.tolist()

# Organ を並び順でカテゴリ型にする
df_box["shibaki_organ_label"] = pd.Categorical(df_box["shibaki_organ_label"], categories=unique_organ_sorted, ordered=True)

# 描画
fig, ax1 = plt.subplots(figsize=(30, 10))

# DICE boxplot（左Y軸）
positions = range(len(unique_organ_sorted))
dice_data = [df_box[df_box["shibaki_organ_label"] == organ]["dice"].dropna() for organ in unique_organ_sorted]
bpl = ax1.boxplot(
    dice_data,
    positions=positions, 
    widths=0.35, 
    patch_artist=True,
    boxprops=dict(facecolor="#99C7C3"), 
    showfliers=False,
    whiskerprops=dict(linestyle="none"),
    capprops=dict(linestyle='none'),
)
ax1.set_ylabel("DICE Score", fontsize=20)
ax1.set_ylim(0, 1)
ax1.tick_params(axis='both', labelsize=20) 

# MHD boxplot（右Y軸）
ax2 = ax1.twinx()
mhd_data = [df_box[df_box["shibaki_organ_label"] == organ]["MeanHD"].dropna() for organ in unique_organ_sorted]
bpr = ax2.boxplot(
    mhd_data,
    positions=[p + 0.4 for p in positions],
    widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor="#F4DF6E"),
    showfliers=False,
    whiskerprops=dict(linestyle="none"),
    capprops=dict(linestyle='none'),
)
ax2.set_ylabel("Mean Hausdorff Distance", fontsize=20)
ax2.set_ylim(5, 0)  # 逆軸
ax2.tick_params(axis='y', labelsize=20)

# X軸
ax1.set_xticks([p + 0.2 for p in positions])
ax1.set_xticklabels(unique_organ_sorted, rotation=-60, fontsize=25)

# タイトル
plt.title("Box plot of DICE and Mean HD by Organ", fontsize=20, fontweight="bold")

# 凡例

dice_patch = mpatches.Patch(color="#99C7C3", label='DICE')
mhd_patch = mpatches.Patch(color="#F4DF6E", label='Mean HD')
plt.legend(handles=[dice_patch, mhd_patch], loc="upper right")

plt.tight_layout()
plt.savefig(save_figure_path + "/boxplot_DICE_MHD_by_organ.png", dpi=300)
plt.close()








# cancer type を多い順に並べる
cancer_type_counts = df_box["shibaki_cancer_type_label"].value_counts()
unique_cancer_type_sorted = cancer_type_counts.index.tolist()

# cancer type を並び順でカテゴリ型にする
df_box["shibaki_cancer_type_label"] = pd.Categorical(df_box["shibaki_cancer_type_label"], categories=unique_cancer_type_sorted, ordered=True)

# 描画
fig, ax1 = plt.subplots(figsize=(30, 10))

# DICE boxplot（左Y軸）
positions = range(len(unique_cancer_type_sorted))
dice_data = [df_box[df_box["shibaki_cancer_type_label"] == cancer_type]["dice"].dropna() for cancer_type in unique_cancer_type_sorted]
bpl = ax1.boxplot(
    dice_data,
    positions=positions,
    widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor="#99C7C3"),
    showfliers=False,
    whiskerprops=dict(linestyle="none"),
    capprops=dict(linestyle='none'),
)
ax1.set_ylabel("DICE Score", fontsize=20)
ax1.set_ylim(0, 1)
ax1.tick_params(axis='both', labelsize=20) 

# MHD boxplot（右Y軸）
ax2 = ax1.twinx()
mhd_data = [df_box[df_box["shibaki_cancer_type_label"] == cancer_type]["MeanHD"].dropna() for cancer_type in unique_cancer_type_sorted]
bpr = ax2.boxplot(
    mhd_data,
    positions=[p + 0.4 for p in positions],
    widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor="#F4DF6E"),
    showfliers=False,
    whiskerprops=dict(linestyle="none"),
    capprops=dict(linestyle='none'),
)
ax2.set_ylabel("Mean Hausdorff Distance", fontsize=20)
ax2.set_ylim(5, 0)  # 逆軸
ax2.tick_params(axis='y', labelsize=20)

# X軸
ax1.set_xticks([p + 0.2 for p in positions])
ax1.set_xticklabels(unique_cancer_type_sorted, rotation=-60, fontsize=25)

# タイトル
plt.title("Box plot of DICE and Mean HD by Cancer type", fontsize=20, fontweight="bold")

# 凡例
import matplotlib.patches as mpatches
dice_patch = mpatches.Patch(color="#99C7C3", label='DICE')
mhd_patch = mpatches.Patch(color="#F4DF6E", label='Mean HD')
plt.legend(handles=[dice_patch, mhd_patch], loc="upper right")

plt.tight_layout()
plt.savefig(save_figure_path + "/boxplot_DICE_MHD_by_cancer_type.png", dpi=300)
plt.close()
