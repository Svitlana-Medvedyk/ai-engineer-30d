## Threshold tuning (v2 LogisticRegression)
- t=0.3: precision=0.667, recall=0.812, F1=0.732, CM=[[82,28],[13,56]]
- t=0.5: precision=0.783, recall=0.681, F1=0.729, CM=[[97,13],[22,47]]
Chosen threshold: t=0.3 (goal: reduce FN / increase recall for Survived=1).

## Baseline v3 (RandomForest)
- Default t=0.5: Acc=0.816, ROC-AUC=0.833, CM=[[97,13],[20,49]]
- Threshold sweep highlights:
  - t=0.4: precision=0.726, recall=0.768, F1=0.746, CM=[[90,20],[16,53]]
  - t=0.5: precision=0.790, recall=0.710, F1=0.748, CM=[[97,13],[20,49]]

## Takeaways
- RandomForest improved accuracy on this split but had lower ROC-AUC than LogisticRegression v2.
- For maximizing recall (catching survivors), LogisticRegression v2 at t=0.3 performed best (FN=13, recall=0.812).
