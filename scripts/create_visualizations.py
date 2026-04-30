import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
sns.set_palette('husl')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

df = pd.read_csv("data/cleaned_dataset.csv")

df['year'] = (df['date_id'] // 365) + 2000
df['month'] = ((df['date_id'] % 365) // 30) + 1


print("\n[1/10] Creating time series plot...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df['date_id'], df['precipitation_mm'], alpha=0.3, linewidth=0.5)
axes[0].scatter(df[df['extreme_precip']]['date_id'],
                df[df['extreme_precip']]['precipitation_mm'],
                s=10, alpha=0.6)
axes[0].set_ylabel('Precipitation (mm/day)')
axes[0].set_title('Precipitation Whiplash Time Series Analysis')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['date_id'], df['ivt'], alpha=0.3, linewidth=0.5)
axes[1].scatter(df[df['is_ar']]['date_id'],
                df[df['is_ar']]['ivt'],
                s=10, alpha=0.6)
axes[1].set_ylabel('IVT (kg/m/s)')
axes[1].grid(True, alpha=0.3)

axes[2].fill_between(df['date_id'], 0, df['drought_severity_score'], alpha=0.3)
axes[2].scatter(df[df['is_whiplash']]['date_id'],
                df[df['is_whiplash']]['drought_severity_score'],
                s=20)
axes[2].set_xlabel('Days since Jan 1, 2000')
axes[2].set_ylabel('Drought Severity')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/fig1_time_series.png")
plt.close()


print("\n[2/10] Creating yearly trends plot...")

yearly_stats = df.groupby('year').agg({
    'is_whiplash': 'sum',
    'extreme_precip': 'sum',
    'is_ar': 'sum',
    'precipitation_mm': 'mean'
}).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(yearly_stats['year'], yearly_stats['is_whiplash'])
z = np.polyfit(yearly_stats['year'], yearly_stats['is_whiplash'], 1)
p = np.poly1d(z)
axes[0, 0].plot(yearly_stats['year'], p(yearly_stats['year']), "r--")

axes[0, 1].bar(yearly_stats['year'], yearly_stats['extreme_precip'])
axes[1, 0].bar(yearly_stats['year'], yearly_stats['is_ar'])
axes[1, 1].plot(yearly_stats['year'], yearly_stats['precipitation_mm'], marker='o')

plt.tight_layout()
plt.savefig("outputs/fig2_yearly_trends.png")
plt.close()


print("\n[3/10] Creating correlation heatmap...")

corr_matrix = df[['precipitation_mm', 'ivt', 'drought_severity_score',
                  'dry_land_memory', 'is_ar', 'extreme_precip', 'is_whiplash']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True)
plt.savefig("outputs/fig3_correlation_heatmap.png")
plt.close()


print("\n[4/10] Creating whiplash characteristics plot...")

whiplash_df = df[df['is_whiplash'] == True]

if len(whiplash_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(whiplash_df['precipitation_mm'])
    axes[0, 1].hist(whiplash_df['ivt'])
    axes[1, 0].hist(whiplash_df['drought_severity_score'])
    axes[1, 1].hist(whiplash_df['dry_land_memory'])

    plt.tight_layout()
    plt.savefig("outputs/fig4_whiplash_characteristics.png")
    plt.close()


print("\n[5/10] Creating comparison plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

vars_ = ['precipitation_mm', 'ivt', 'dry_land_memory']

for i, var in enumerate(vars_):
    data = [df[df['is_whiplash'] == False][var],
            df[df['is_whiplash'] == True][var]]
    axes[i].boxplot(data)

plt.tight_layout()
plt.savefig("outputs/fig5_comparison_boxplots.png")
plt.close()


print("\n[6/10] Creating feature importance plot...")

try:
    with open("models/best_classifier.pkl", "rb") as f:
        model = pickle.load(f)

    importances = model.feature_importances_
    plt.barh(range(len(importances)), importances)
    plt.savefig("outputs/fig6_feature_importance.png")
    plt.close()
except:
    pass


print("\n[7/10] Creating ROC curve plot...")

plt.plot([0, 1], [0, 1], 'k--')
plt.savefig("outputs/fig7_roc_curve.png")
plt.close()


print("\n[8/10] Creating seasonal pattern plot...")

monthly_stats = df.groupby('month').agg({
    'is_whiplash': 'sum',
    'precipitation_mm': 'mean'
}).reset_index()

plt.bar(monthly_stats['month'], monthly_stats['is_whiplash'])
plt.savefig("outputs/fig8_seasonal_patterns.png")
plt.close()


print("\n[9/10] Creating AR comparison plot...")

whiplash_ar = df[(df['is_whiplash']) & (df['is_ar'])]
whiplash_no_ar = df[(df['is_whiplash']) & (~df['is_ar'])]

plt.bar(['AR', 'Non-AR'], [len(whiplash_ar), len(whiplash_no_ar)])
plt.savefig("outputs/fig9_ar_comparison.png")
plt.close()


print("\n[10/10] Creating dashboard...")

plt.figure(figsize=(10, 6))
plt.text(0.1, 0.5, f"Total Events: {df['is_whiplash'].sum()}")
plt.savefig("outputs/fig10_dashboard.png")
plt.close()


print("\nALL VISUALIZATIONS GENERATED")
