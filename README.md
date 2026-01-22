# Pattern Discovery in Noisy Human Cyclic Data

This project explores whether noisy human cyclic data contains meaningful structure
when analyzed beyond simple averages.

Menstrual cycle data is used as a proxy for complex, behavior-influenced,
multivariate time-series. This is not a medical or diagnostic project.

## Core Idea
Two individuals can share similar averages but differ fundamentally
in how stable or unstable their behavior is over time.

## Methodology
- User-level aggregation and variability feature engineering
- PCA to identify dominant modes of instability
- HDBSCAN for density-based, unsupervised clustering
- Post-hoc comparison of lifestyle factors (sleep, stress, BMI)

## Observations
- Individuals with similar average cycle lengths showed markedly different temporal stability
- Variability-based features separated individuals more clearly than mean-based features
- Unsupervised clustering revealed distinct instability profiles without predefined labels
- Lifestyle-related variables showed weak but non-random associations with instability patterns

## Key Takeaways
- Instability is multi-dimensional, not a single scalar quantity
- Natural groupings emerge without labels or medical thresholds
- Averages alone hide meaningful behavioral structure

## Notes
This repository contains exploratory research code focused on analysis and reasoning,
not production-level engineering.

