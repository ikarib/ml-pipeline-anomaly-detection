# Limitations

This project is stronger than a minimal demo, but it is still a portfolio sample.

## Main limitations

1. **Synthetic data**
   The data structure is useful for experimentation, but it does not capture every messiness factor from a live industrial system such as sensor outages, maintenance flags, missing intervals, and label ambiguity.
2. **Point-level metrics**
   The current evaluation is row-based. In operational settings, it is usually better to measure event detection: was the anomaly episode caught early enough, and how noisy were the alerts?
3. **Small feature space**
   Only local rolling features are used. Real systems often benefit from domain features, lag stacks, external context, and asset metadata.

4. **No deployment layer**
   There is no model serving, alert routing, or retraining logic in this repository. The focus is experimentation and explanation.
5. **Random holdout for the supervised model**
   The Random Forest benchmark uses a random split. That is fine for a compact demo, but a production evaluation should use a time-based split to avoid leakage from nearby patterns.
