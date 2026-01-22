import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("menstrual_cycle_dataset_with_factors.csv")
print(df.head())
print(df.columns)
print(df.info())

#converting date columns
df['Cycle Start Date'] = pd.to_datetime(df['Cycle Start Date'])
df['Next Cycle Start Date'] = pd.to_datetime(df['Next Cycle Start Date'])
print(df.tail())

#observing
print(
    df[['Age', 'BMI', 'Stress Level', 'Sleep Hours',
        'Cycle Length', 'Period Length']].describe()
)

#converting data to one row one user
user_cycle_stats = (
    df
    .groupby('User ID')['Cycle Length']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
print(user_cycle_stats.head())

#adding other factors along with cycle length
user_agg = df.groupby("User ID").agg({
    "Cycle Length": ["mean", "std"],
    "Sleep Hours": "mean",
    "Stress Level": "mean",
    "BMI": "mean"
})

# Flatten multi-level columns
user_agg.columns = ['Cycle Length_mean', 'Cycle Length_std', 'Sleep Hours_mean', 'Stress Level_mean', 'BMI_mean']

# Reset index to have 'User ID' as a column
user_agg = user_agg.reset_index()

#checking distributions
numerical_features = ['Cycle Length_mean','Cycle Length_std','Sleep Hours_mean','Stress Level_mean','BMI_mean']

for feature in numerical_features:
    plt.figure(figsize=(6,4))
    plt.hist(user_agg[feature], bins=15, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Number of Users')
    plt.show()

#graph for std of cycle_length
plt.figure(figsize=(6,4))
plt.hist(user_agg['Cycle Length_std'], bins=15, edgecolor='black')
plt.title('Distribution of Cycle Length Variability')
plt.xlabel('Cycle Length Standard Deviation')
plt.ylabel('Number of Users')
plt.show()

#graph for sleep cycle
plt.figure(figsize=(6,4))
plt.hist(user_agg['Sleep Hours_mean'], bins=15, edgecolor='black')
plt.title('Distribution of Average Sleep Hours')
plt.xlabel('Average Sleep Hours')
plt.ylabel('Number of Users')
plt.show()

#graph for stress level
plt.figure(figsize=(6,4))
plt.hist(user_agg['Stress Level_mean'], bins=15, edgecolor='black')
plt.title('Distribution of Average Stress Level')
plt.xlabel('Average Stress Level')
plt.ylabel('Number of Users')
plt.show()

#graph for bmi
plt.figure(figsize=(6,4))
plt.hist(user_agg['BMI_mean'], bins=15, edgecolor='black')
plt.title('Distribution of Average BMI')
plt.xlabel('Average BMI')
plt.ylabel('Number of Users')
plt.show()

#outliers
features = [
    'Cycle Length_mean',
    'Cycle Length_std',
    'Sleep Hours_mean',
    'Stress Level_mean',
    'BMI_mean'
]

for feature in features:
    plt.figure(figsize=(6,2))
    sns.boxplot(x=user_agg[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

#correlation analysis
corr_features = [
    'Cycle Length_mean',
    'Cycle Length_std',
    'Sleep Hours_mean',
    'Stress Level_mean',
    'BMI_mean'
]

corr_matrix = user_agg[corr_features].corr()
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix (User-Level Features)')
plt.show()

#adding more variability features(Cycle Length_std,cycle Length_cv,Cycle Length_range,Cycle Length_mad,Cycle Length_diff_std)
user_agg['Cycle Length_cv'] = (
    user_agg['Cycle Length_std'] / user_agg['Cycle Length_mean']
)

cycle_range = (
    df.groupby('User ID')['Cycle Length']
    .agg(lambda x: x.max() - x.min())
    .reset_index(name='Cycle Length_range')
)

user_agg = user_agg.merge(cycle_range, on='User ID')

cycle_mad = (
    df.groupby('User ID')['Cycle Length']
    .agg(lambda x: np.median(np.abs(x - np.median(x))))
    .reset_index(name='Cycle Length_mad')
)

user_agg = user_agg.merge(cycle_mad, on='User ID')

cycle_diff_std = (
    df.sort_values(['User ID', 'Cycle Start Date'])
      .groupby('User ID')['Cycle Length']
      .apply(lambda x: x.diff().std())
      .reset_index(name='Cycle Length_diff_std')
)

user_agg = user_agg.merge(cycle_diff_std, on='User ID')

variability_features = [
    'Cycle Length_std',
    'Cycle Length_cv',
    'Cycle Length_range',
    'Cycle Length_mad',
    'Cycle Length_diff_std'
]

print(user_agg[variability_features].head())
print(user_agg[variability_features].describe())

for feature in variability_features:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=user_agg[feature])
    plt.title(f'Distribution of {feature}')
    plt.show()

#correlating variablity features
var_corr = user_agg[variability_features].corr()

plt.figure(figsize=(8,6))
sns.heatmap(var_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Variability Features')
plt.show()

#starting pca
from sklearn.decomposition import PCA

pca_features = [
    'Cycle Length_std',
    'Cycle Length_cv',
    'Cycle Length_range',
    'Cycle Length_mad',
    'Cycle Length_diff_std'
]

X = user_agg[pca_features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

for i, var in enumerate(explained_variance, 1):
    print(f"PC{i}: {var:.2f}")

loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_features,
    columns=[f'PC{i}' for i in range(1, len(pca_features)+1)]
)
print(loadings)

#converting pca to 2D
user_agg['PC1'] = X_pca[:, 0]
user_agg['PC2'] = X_pca[:, 1]

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(7,6))
plt.scatter(user_agg['PC1'], user_agg['PC2'], alpha=0.7)
plt.xlabel('PC1 (Overall Instability)')
plt.ylabel('PC2 (Instability Type)')
plt.title('PCA Projection of Cycle Variability (User-Level)')
plt.axhline(0)
plt.axvline(0)
plt.show()

#hdbscan
import hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=8,
    min_samples=4,
    metric='euclidean'
)
user_agg['cluster'] = clusterer.fit_predict(
    user_agg[['PC1', 'PC2']]
)
print(user_agg['cluster'].value_counts())

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=user_agg['PC1'],
    y=user_agg['PC2'],
    hue=user_agg['cluster'],
    palette='tab10',
    legend='full'
)

plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)

plt.title('HDBSCAN Clusters on PCA Space')
plt.xlabel('PC1 (Overall Instability)')
plt.ylabel('PC2 (Instability Type)')
plt.show()

#doing lifestyle comparison with other variability features those were left before
clustered_data = user_agg[user_agg['cluster'] != -1]

#sleep hours vs cluster
plt.figure(figsize=(6,4))
sns.boxplot(
    x='cluster',
    y='Sleep Hours_mean',
    data=clustered_data
)
plt.title('Average Sleep Hours Across Variability Clusters')
plt.xlabel('Cluster')
plt.ylabel('Average Sleep Hours')
plt.show()

#stress level vs clsuter
plt.figure(figsize=(6,4))
sns.boxplot(
    x='cluster',
    y='Stress Level_mean',
    data=clustered_data
)
plt.title('Average Stress Level Across Variability Clusters')
plt.xlabel('Cluster')
plt.ylabel('Average Stress Level')
plt.show()

#bmi vs cluster
plt.figure(figsize=(6,4))
sns.boxplot(
    x='cluster',
    y='BMI_mean',
    data=clustered_data
)
plt.title('Average BMI Across Variability Clusters')
plt.xlabel('Cluster')
plt.ylabel('Average BMI')
plt.show()

#summary table
cluster_summary = (
    clustered_data
    .groupby('cluster')[['Sleep Hours_mean', 'Stress Level_mean', 'BMI_mean']]
    .agg(['mean', 'std'])
)

print(cluster_summary)
