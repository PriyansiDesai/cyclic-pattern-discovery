# Pattern Discovery in Noisy Human Cyclic Data
This project explores whether noisy human cyclic data contains meaningful structure when analyzed beyond simple averages.

Menstrual cycle data is used as a proxy for complex, behavior-influenced, multivariate time-series.  
**This is not a medical or diagnostic project.**

## Core Idea
Two individuals can share similar averages yet differ fundamentally in how stable or unstable their behavior is over time.
Mean values alone often obscure variability patterns that are central to understanding real-world human behavior.

## Methodology
- User-level aggregation and variability-focused feature engineering  
- Dimensionality reduction using PCA to capture dominant modes of instability  
- Density-based unsupervised clustering using HDBSCAN  
- Post-hoc comparison with lifestyle-related factors (sleep, stress, BMI)  

## Observations
- Individuals with similar average cycle lengths exhibited markedly different temporal stability profiles  
- Variability-based features separated individuals more clearly than mean-based features  
- PCA revealed multiple dimensions of instability rather than a single scalar axis  
- HDBSCAN identified distinct density-based groupings without predefined labels  
- A subset of individuals was classified as noise, indicating atypical or extreme instability patterns  
- Lifestyle-related variables showed weak but non-random associations with certain instability profiles
  
## Interpreting the Discovered Groups

Unsupervised clustering revealed distinct **instability archetypes**:
- **Compact / Low-Variability Group**  
  Individuals with relatively stable behavior across cycles and lower overall variability  
- **Moderate Instability Group**  
  Individuals showing higher variability across selected features, but still forming a dense cluster  
- **Noise / Outliers (HDBSCAN label = -1)**  
  Individuals with irregular or extreme variability patterns that do not align with dominant groups and are better treated as noise rather than forced into clusters  
These groupings emerge purely from data structure and density, without predefined medical thresholds or labels.

## Key Takeaways
- Instability is multi-dimensional, not a single scalar quantity  
- Natural groupings can emerge without labels, rules, or domain-specific cutoffs  
- Averages alone often hide meaningful behavioral structure  
- Exploratory data analysis is not just a setup step — it is where hidden structure often first appears  

## Notes
This repository contains exploratory research code focused on analysis and reasoning rather than production-level engineering.
