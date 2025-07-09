import pandas as pd
import numpy as np

# パス
input_csv_path = "data/excel/patients_data.csv"
input_xlsx_path = "data/excel/data_result_250422.xlsx"
output_csv_path = "output/excel/patients_chara.csv"
output_chara_patient_path = "output/table/chara_patients.xlsx"
output_chara_lesion_path = "output/table/chara_lesions.xlsx"

df_csv = pd.read_csv(input_csv_path)
df_csv = df_csv[~((df_csv["kobayashi"] == 0) | (df_csv["kobayashi"].isna()) | (df_csv["kobayashi"] == ""))].copy()
df_excel = pd.read_excel(input_xlsx_path)
df_excel = df_excel[~((df_excel["shibaki_comments"] == "delete"))].copy()
df_excel_filtered = df_excel[["project_id", "No", "shibaki_organ_label", "shibaki_cancer_type", "shibaki_cancer_type_label"]].copy()

# 両方を文字列型に揃える
df_csv["project_id"] = df_csv["project_id"].astype(str)
df_excel_filtered["project_id"] = df_excel_filtered["project_id"].astype(str)

# "Patient ID"で結合 (inner join: 両方に存在するIDのみ)
df_merged = pd.merge(df_csv, df_excel_filtered, on="project_id", how="inner")

# project_id単位で1行だけ残す
df_merged_unique = df_merged.drop_duplicates(subset="project_id").copy()

df_merged_unique['ano_date'] = pd.to_datetime(
    df_merged_unique['ano_date'].astype(str).str.strip(),
    errors='coerce'
)

df_merged_unique['birth_date'] = pd.to_datetime(
    df_merged_unique['birth_date'].astype(str).str.strip(),
    errors='coerce'
)

# 日付型に変換
df_merged_unique.loc[:, 'ano_date']= pd.to_datetime(df_merged_unique['ano_date'], format='%Y/%m/%d', errors='coerce')
df_merged_unique.loc[:, 'birth_date'] = pd.to_datetime(df_merged_unique['birth_date'], format='%Y/%m/%d', errors='coerce')

# 年齢を計算
df_merged_unique.loc[:, 'age'] = (df_merged_unique['ano_date'] - df_merged_unique['birth_date']).dt.days // 365

# CSVとして保存
df_merged_unique.to_csv(output_csv_path, index=False)


########## Patients character tableを作成 ##########
df_calc = pd.read_csv(output_csv_path)
df_calc = df_calc.rename(columns={
    'sex(female:0, male:1)': 'sex',
    'shibaki_cancer_type_y': 'cancer_type_chara'
})

# Ageの人中央値と範囲
age_median = df_calc["age"].median()
age_min = df_calc["age"].min()
age_max = df_calc["age"].max()

# ecog_psの0と1以上の人数と割合
ecog_ps_count = df_calc['ecog_ps'].apply(lambda x: '0' if x == 0 else '1+').value_counts().reset_index()
ecog_ps_count.columns = ['ecog_ps', 'count']
ecog_ps_count['percentage'] = ecog_ps_count['count'] / ecog_ps_count['count'].sum() * 100

# sexの人数と割合
sex_count = df_calc['sex'].map({0: 'female', 1: 'male'}).value_counts().reset_index()
sex_count.columns = ['sex', 'count']
sex_count['percentage'] = sex_count['count'] / sex_count['count'].sum() * 100

# Stageの人数と割合
stage_count = df_calc['Stage'].value_counts().reset_index()
stage_count.columns = ['Stage', 'count']
stage_count['percentage'] = stage_count['count'] / stage_count['count'].sum() * 100

# treatment_lineの中央値と範囲
treatment_line_median = df_calc['treatment_line'].median()
treatment_line_min = df_calc['treatment_line'].min()
treatment_line_max = df_calc['treatment_line'].max()

# cancer_type_labelの人数と割合
cancer_type_count = df_calc['cancer_type_chara'].value_counts().reset_index()
cancer_type_count.columns = ['cancer_type_chara', 'count']
cancer_type_count['percentage'] = cancer_type_count['count'] / cancer_type_count['count'].sum() * 100

# データフレームにtype列を追加
ecog_ps_count['type'] = 'ecog_ps'
sex_count['type'] = 'sex'
stage_count['type'] = 'Stage'
cancer_type_count['type'] = 'cancer_type'

# カラムを揃える
ecog_ps_count = ecog_ps_count.rename(columns={'ecog_ps': 'category'})[['type', 'category', 'count', 'percentage']]
sex_count = sex_count.rename(columns={'sex': 'category'})[['type', 'category', 'count', 'percentage']]
stage_count = stage_count.rename(columns={'Stage': 'category'})[['type', 'category', 'count', 'percentage']]
cancer_type_count = cancer_type_count.rename(columns={'cancer_type_chara': 'category'})[['type', 'category', 'count', 'percentage']]

# 他のテーブルにもmedian, min, max列を追加
for df_ in [ecog_ps_count, sex_count, stage_count, cancer_type_count]:
    df_['median'] = np.nan
    df_['min'] = np.nan
    df_['max'] = np.nan

# treatment_lineをデータフレーム化
age_stats = pd.DataFrame({
    'type': ['age'],
    'category': [pd.NA],
    'count': [np.nan],
    'percentage': [np.nan],
    'median': [age_median],
    'min': [age_min],
    'max': [age_max]
})

# treatment_lineをデータフレーム化
treatment_line_stats = pd.DataFrame({
    'type': ['treatment_line'],
    'category': [pd.NA],
    'count': [np.nan],
    'percentage': [np.nan],
    'median': [treatment_line_median],
    'min': [treatment_line_min],
    'max': [treatment_line_max]
})

# カラム順を揃える
treatment_line_stats = treatment_line_stats[['type', 'category', 'count', 'percentage', 'median', 'min', 'max']]

# すべてを結合
combined_df = pd.concat(
    [age_stats, ecog_ps_count, sex_count, stage_count, cancer_type_count, treatment_line_stats],
    ignore_index=True
)

# Excelに出力
combined_df.to_excel(output_chara_patient_path, index=False)


########## lesions chara作成 ############
df_lesion = df_excel.copy()
df_lesion = df_lesion.rename(columns={
    'shibaki_organ_label': 'organ',
    'shibaki_cancer_type': 'cancer_type',
    "Distance (mm)": "tumor_size"
})

# 全体を記載
all_count = len(df_lesion)

# all lesions 行を作る
all_row = pd.DataFrame({
    'category': ['all'],
    'count': [all_count],
    'percentage': [100.0]
})

# sizeの数と割合
size_bins = df_lesion['tumor_size'].apply(
    lambda x: '<5 mm' if x < 5 else ("5-10 mm" if x < 10 else ">=10 mm")
    )
size_count = size_bins.value_counts().reset_index()
size_count.columns = ['tumor_size', 'count']
size_count['percentage'] = size_count['count'] / size_count['count'].sum() * 100

# organの数と割合
organ_count = df_lesion['organ'].value_counts().reset_index()
organ_count.columns = ['organ', 'count']
organ_count['percentage'] = organ_count['count'] / organ_count['count'].sum() * 100

# cancer typeの数と割合
cancer_type_count_2 = df_lesion['cancer_type'].value_counts().reset_index()
cancer_type_count_2.columns = ['cancer_type', 'count']
cancer_type_count_2['percentage'] = cancer_type_count_2['count'] / cancer_type_count_2['count'].sum() * 100

# データフレームにtype列を追加
all_row["type"] = "all_lesion"
size_count['type'] = 'tumor_size'
organ_count['type'] = 'organ'
stage_count['type'] = 'cancer_type'
cancer_type_count_2['type'] = 'cancer_type'

# カラムを揃える
all_row = all_row.rename(columns={'all_lesion': 'category'})[['type', 'category', 'count', 'percentage']]
size_count = size_count.rename(columns={'tumor_size': 'category'})[['type', 'category', 'count', 'percentage']]
organ_count = organ_count.rename(columns={'organ': 'category'})[['type', 'category', 'count', 'percentage']]
cancer_type_count_2 = cancer_type_count_2.rename(columns={'cancer_type': 'category'})[['type', 'category', 'count', 'percentage']]

# カラム順を揃える
treatment_line_stats = treatment_line_stats[['type', 'category', 'count', 'percentage']]

# すべてを結合
combined_df = pd.concat(
    [all_row, size_count, organ_count, cancer_type_count_2],
    ignore_index=True
)

# Excelに出力
combined_df.to_excel(output_chara_lesion_path, index=False)
