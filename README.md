#Pattern Discovery in Noisy Human Cyclic Data
This project explores whether noisy human cyclic data contains meaningful structure when analyzed beyond simple averages.

Menstrual cycle data is used as a proxy for complex, behavior-influenced, multivariate time-series.
This is not a medical or diagnostic project.

#Core Idea
-Two individuals can share similar averages yet differ fundamentally in how stable or unstable their behavior is over time.
-Mean values alone often obscure variability patterns that are central to understanding real-world human behavior.

#Methodology
-User-level aggregation and variability-focused feature engineering
-Dimensionality reduction using PCA to capture dominant modes of instability
-Density-based unsupervised clustering using HDBSCAN
-Post-hoc comparison with lifestyle-related factors (sleep, stress, BMI)

#Observations
-Individuals with similar average cycle lengths exhibited markedly different temporal stability profiles
-Variability-based features (variance, coefficient of variation, spread) separated individuals more clearly than mean-based features
-PCA revealed multiple dimensions of instability rather than a single “more vs less” axis
-HDBSCAN identified distinct density-based groupings without predefined labels, along with a set of non-conforming outliers
-Lifestyle-related variables showed weak but non-random associations with certain instability patterns

#Interpreting the Discovered Groups
-The unsupervised clustering surfaced distinct regions in variability feature space, which can be interpreted as different instability archetypes:
-Dense Core Cluster
Individuals showing moderate overall instability but diverse instability structures across features
-Compact Low-Variability Cluster
Individuals with relatively lower overall instability and more homogeneous variability patterns
-Noise / Outliers (HDBSCAN -1)
Individuals with atypical or extreme variability patterns that do not align with dominant groups and are better treated as noise rather than forced into clusters
These groupings emerge purely from data density and structure, without medical thresholds or behavioral labels.

#Key Takeaways
-Instability is multi-dimensional, not a single scalar quantity
-Natural groupings can emerge without labels, rules, or domain-specific cutoffs
-Averages alone often hide meaningful behavioral structure
-Exploratory analysis is not just a preliminary step — it is where hidden structure often first becomes visible

#Notes
-This repository contains exploratory research code focused on analysis, reasoning, and pattern discovery rather than production-level engineering.
