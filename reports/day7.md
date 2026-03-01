# Day 7 — Calibration + threshold selection (production-style)

## Goal
Make probability outputs more reliable and choose a decision threshold based on a target objective (precision vs recall), not just default 0.5.

## Model
- LogisticRegression v3 (with Title) + preprocessing pipeline (impute + one-hot).

## Calibration
- Compared raw (uncalibrated) vs calibrated probabilities using Platt scaling (sigmoid).
- Example outputs (first 5):
  - Uncalibrated: [0.0714, 0.9589, 0.6063, 0.9466, 0.0733]
  - Calibrated:   [0.0785, 0.9468, 0.5940, 0.9349, 0.0828]
- Observation: calibrated probabilities are slightly less extreme and more stable for threshold decisions.

## Threshold selection
- Planned: search thresholds in a range (e.g., 0.1–0.9) and pick the best threshold for:
  - F1 (balanced precision/recall)
  - Recall (minimize false negatives for Survived=1)
- Results:
  - Best F1 threshold (uncalibrated): t=...
  - Best F1 threshold (calibrated):   t=...
  - Best Recall threshold (uncalibrated): t=...
  - Best Recall threshold (calibrated):   t=...

## Takeaways
- Calibration is useful when we care about probability quality and consistent thresholding.
- Threshold tuning should be guided by the business goal:
  - prioritize recall to avoid missing positives (FN),
  - prioritize precision to avoid too many false positives (FP).

## Next steps
1) Rerun threshold search and record chosen threshold + metrics (precision/recall/F1 and confusion matrix).
2) Compare sigmoid vs isotonic calibration (optional).
3) Apply the chosen threshold to the best-performing model and report error tradeoffs.
