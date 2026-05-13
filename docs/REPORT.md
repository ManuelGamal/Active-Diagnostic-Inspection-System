# Engineer 5 Evaluation Report  
## Supervised Industrial Defect Detection on MVTec AD

---

## 1. Project Context

This project evaluates a supervised industrial defect detection system on six categories from the MVTec AD dataset: **bottle, capsule, carpet, hazelnut, leather, and pill**. The goal is to classify each industrial product image as either **normal** or **defective**, then provide both quantitative and qualitative evidence of model performance.

Although the original repository title mentions self-supervised industrial defect detection, the completed experimental scope for this stage is a **supervised ResNet-50 binary classification baseline**. The model is trained and evaluated using 100% of the available labeled data for the selected six categories, with three cross-validation folds per category.

Engineer 5 is responsible for the evaluation infrastructure. This includes metric computation, bootstrap confidence intervals, checkpoint evaluation, qualitative Grad-CAM galleries, failure-case analysis, the final evaluation notebook, and this report.

---

## 2. Evaluation Scope

The evaluation covers six MVTec AD categories:

- bottle
- capsule
- carpet
- hazelnut
- leather
- pill

The evaluation uses the best selected checkpoint per category based on Engineer 4's best-fold analysis. The task is image-level binary classification:

- class 0: normal
- class 1: defective

The evaluation metrics are:

- AUROC
- AUPR
- F1 score at the optimal threshold
- Accuracy
- 95% bootstrap confidence intervals

The qualitative outputs are:

- ROC curves
- Precision-Recall curves
- Confusion matrices
- Grad-CAM galleries
- Failure-case galleries

The final evaluation notebook used for this report is:

`notebooks/04_Evaluation.ipynb`

The notebook runs the complete evaluation pipeline end-to-end on Kaggle using the trained checkpoints and the real MVTec test dataloaders.

---

## 3. Methodology

### 3.1 Model Architecture

The evaluated model is a supervised binary classifier based on a ResNet-50 backbone.

The model pipeline is:

Input image → ResNet-50 encoder → Global Average Pooling → Dropout → Linear classification head

The ResNet-50 backbone is loaded using `timm`, and the final classification head outputs two logits:

- logit 0: normal
- logit 1: defective

During evaluation, the defect probability is computed using softmax over the output logits:

defect probability = softmax(logits)[class 1]

This defect probability is then used to compute AUROC, AUPR, F1, and accuracy.

---

### 3.2 Training Context

The models were trained using:

- ResNet-50 feature extractor
- binary classification head
- focal loss
- AdamW optimizer
- cosine learning-rate schedule
- PyTorch Lightning training loop

Focal loss was used because industrial defect datasets often contain class imbalance between normal and defective samples. This helps reduce the dominance of easy majority-class examples during training.

Training was completed by Engineer 3 using:

6 categories × 3 folds = 18 supervised training runs

Engineer 4 then selected the best-performing fold checkpoint for each category. Engineer 5 used these selected checkpoints for the final evaluation.

---

### 3.3 Data Protocol

The project uses MVTec AD data prepared into train, validation, and test splits. For each category, three cross-validation folds were trained and evaluated.

The final evaluation uses the best fold checkpoint per category selected from Engineer 4's fold analysis.

The final evaluation therefore represents:

- 100% labeled supervised evaluation
- best selected fold per category
- real test-set predictions
- sample-level bootstrap confidence intervals

This is important because the reported results are not based on random or mocked predictions. They are generated using real checkpoints and real dataloaders.

---

## 4. Evaluation Implementation

The evaluation code is organized under:

`src/evaluation/`

The main files are:

- `src/evaluation/metrics.py`
- `src/evaluation/bootstrap.py`
- `src/evaluation/evaluator.py`
- `src/evaluation/qualitative.py`

---

### 4.1 Metrics

The following functions are implemented in `src/evaluation/metrics.py`:

- `compute_auroc`
- `compute_aupr`
- `compute_f1`
- `compute_f1_optimal`
- `compute_accuracy`
- `compute_pixel_iou`
- `evaluate_detector`

The metrics module was updated to include strict input validation. The validation checks for:

- empty `y_true`
- empty `y_score`
- shape mismatch between labels and scores
- single-class input for metrics where single-class data is undefined

This is important because AUROC, AUPR, and optimal-threshold F1 are not reliable when the ground-truth labels contain only one class. Instead of silently producing misleading values, the functions now raise clear `ValueError` messages.

---

### 4.2 Bootstrap Confidence Intervals

Bootstrap confidence intervals are implemented in:

`src/evaluation/bootstrap.py`

The final evaluation uses:

- 10,000 bootstrap resamples
- 95% confidence intervals
- random seed = 42

Bootstrap confidence intervals are used to estimate uncertainty around each metric. This is especially important because some categories have relatively small test sets, so a single metric value can be misleading without an uncertainty range.

---

### 4.3 Checkpoint Evaluation

Checkpoint evaluation is implemented in:

`src/evaluation/evaluator.py`

The evaluator performs the following steps:

1. Load the selected model checkpoint.
2. Run inference on the test dataloader.
3. Extract defect probabilities using softmax.
4. Compute AUROC, AUPR, F1, accuracy, and threshold.
5. Compute 95% bootstrap confidence intervals.
6. Save diagnostic plots.
7. Save a `metrics.json` file for each evaluated category.

For each category, the evaluator saves:

- `metrics.json`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`

---

### 4.4 Qualitative Evaluation

Qualitative analysis is implemented in:

`src/evaluation/qualitative.py`

It produces:

- Grad-CAM galleries
- failure-case galleries

The Grad-CAM galleries help inspect whether the model is focusing on meaningful product regions when making predictions.

The failure-case galleries show the most problematic examples, such as false positives, false negatives, or highly uncertain samples. These outputs help explain why some categories perform worse than others.

---

## 5. Final Quantitative Results

The following results are from the final Kaggle evaluation run using the best selected checkpoint for each category.

| Category | Fold | Threshold | AUROC (95% CI) | AUPR (95% CI) | F1 (95% CI) | Accuracy (95% CI) |
|---|---:|---:|---|---|---|---|
| bottle | 3 | 0.3255 | 0.9746 [0.9235, 1.0000] | 0.9094 [0.7120, 1.0000] | 0.8571 [0.6667, 1.0000] | 0.9318 [0.8409, 1.0000] |
| capsule | 1 | 0.4617 | 0.8834 [0.7745, 0.9615] | 0.8093 [0.6200, 0.9384] | 0.7407 [0.5000, 0.9000] | 0.8679 [0.7736, 0.9434] |
| carpet | 1 | 0.3225 | 0.9574 [0.9021, 0.9926] | 0.8726 [0.6984, 0.9759] | 0.8125 [0.6316, 0.9412] | 0.9000 [0.8167, 0.9667] |
| hazelnut | 1 | 0.2554 | 0.9846 [0.9531, 1.0000] | 0.9330 [0.7925, 1.0000] | 0.8696 [0.6667, 1.0000] | 0.9605 [0.9079, 1.0000] |
| leather | 3 | 0.5226 | 0.9966 [0.9837, 1.0000] | 0.9911 [0.9556, 1.0000] | 0.9630 [0.8571, 1.0000] | 0.9821 [0.9464, 1.0000] |
| pill | 3 | 0.4983 | 0.9360 [0.8669, 0.9837] | 0.9048 [0.7979, 0.9739] | 0.8085 [0.6667, 0.9167] | 0.8636 [0.7727, 0.9394] |
| **Overall Mean** | — | — | **0.9554** | **0.9034** | **0.8419** | **0.9177** |

Total evaluated test samples across the six categories:

**355 samples**

---

## 6. Per-Category Discussion

### 6.1 Leather

Leather is the strongest category in the final evaluation.

It achieved:

- AUROC = 0.9966
- AUPR = 0.9911
- F1 = 0.9630
- Accuracy = 0.9821

This indicates that the model separates normal and defective leather images very effectively. Leather defects are often visually distinct from the normal texture, allowing the ResNet-50 feature extractor to learn discriminative patterns.

The AUROC confidence interval is also narrow, which suggests that the result is stable across bootstrap samples.

---

### 6.2 Hazelnut

Hazelnut also performs strongly.

It achieved:

- AUROC = 0.9846
- AUPR = 0.9330
- F1 = 0.8696
- Accuracy = 0.9605

The model is able to detect defective hazelnut samples with high confidence. However, the F1 confidence interval is still relatively wide, which suggests that the exact classification threshold and class distribution affect the final binary predictions.

---

### 6.3 Bottle

Bottle achieved:

- AUROC = 0.9746
- AUPR = 0.9094
- F1 = 0.8571
- Accuracy = 0.9318

This is a strong result overall. However, the AUPR and F1 confidence intervals are wider than expected. This is likely due to the smaller number of test samples and the effect of threshold selection on the positive defect class.

The model ranks defective bottle images well, but the final binary classification performance still depends on the selected threshold.

---

### 6.4 Carpet

Carpet achieved:

- AUROC = 0.9574
- AUPR = 0.8726
- F1 = 0.8125
- Accuracy = 0.9000

The AUROC is high, meaning the model generally ranks defective images above normal images. However, the F1 score is lower than the strongest categories.

This suggests that choosing a single operating threshold is harder for carpet. Carpet textures contain repeated patterns and local variations that may visually resemble defects, making classification more challenging.

---

### 6.5 Pill

Pill achieved:

- AUROC = 0.9360
- AUPR = 0.9048
- F1 = 0.8085
- Accuracy = 0.8636

This is a reasonable result, but weaker than leather, hazelnut, and bottle. Pill defects can be subtle, including small cracks, color changes, or shape irregularities. These defects may not dominate the image, making them harder to detect using only image-level labels.

---

### 6.6 Capsule

Capsule is the weakest category in the final test evaluation.

It achieved:

- AUROC = 0.8834
- AUPR = 0.8093
- F1 = 0.7407
- Accuracy = 0.8679

This suggests that capsule is the most difficult category among the six evaluated categories.

Possible reasons include:

- subtle visual differences between normal and defective capsules
- reflections or lighting changes that resemble anomalies
- small defect regions relative to the full image
- possible validation/test distribution gap

Engineer 3's earlier validation-level results suggested stronger capsule performance, but the final selected test result shows a lower AUROC. This indicates that validation performance may not fully represent test difficulty for this category.

---

## 7. Per-Fold Consistency Discussion

Engineer 4's fold analysis is important because it shows whether model performance is stable across different data splits.

The three-fold cross-validation setup is useful because a single split can hide instability. In this project, fold-level differences are especially important for categories with subtle defect patterns.

Categories such as leather, hazelnut, and bottle show strong performance and appear relatively stable. Their high AUROC values suggest that the model consistently learns useful defect-discriminative features.

Categories such as capsule, carpet, and pill are more challenging. Capsule in particular shows weaker final test performance compared with its validation performance, which suggests either overfitting to validation data or a harder selected test fold.

This justifies using three-fold cross-validation instead of a single random split. The additional compute cost provides better evidence about whether a model is genuinely robust or only performs well on a favorable split.

---

## 8. Qualitative Analysis

The qualitative evaluation produces Grad-CAM galleries and failure-case galleries for each category.

The Grad-CAM galleries are intended to answer the question:

Is the model looking at meaningful product regions when making its decision?

For strong categories such as leather and hazelnut, the Grad-CAM overlays generally provide useful evidence that the model focuses on relevant product regions. This supports the quantitative results because the model is not only achieving high AUROC but also appears to attend to meaningful visual areas.

The failure-case galleries are useful for understanding where the model struggles. These examples are especially important for capsule, carpet, and pill, where subtle defects or texture variations may cause false positives or false negatives.

The qualitative outputs are generated by:

- `src/evaluation/qualitative.py`
- `notebooks/04_Evaluation.ipynb`

The generated artifacts include:

- `gradcam_gallery.png`
- `failure_cases.png`

for each evaluated category.

---

## 9. Negative Results and Limitations

This section is mandatory because strong headline metrics alone do not fully describe model behavior.

### 9.1 Capsule is the weakest category

The weakest final result is capsule:

- AUROC = 0.8834
- F1 = 0.7407

This suggests that the model struggles more with capsule defects than with the other evaluated categories. Capsule images may contain small or subtle defects, and normal visual variations may look similar to real anomalies.

This is important because the system may not be equally reliable across all industrial object types.

---

### 9.2 Wide confidence intervals on smaller test sets

Several categories have wide confidence intervals, especially for F1 and AUPR. This is expected because the test-set sizes are relatively small.

For example, bottle has:

F1 = 0.8571 [0.6667, 1.0000]

The wide interval means that the exact value of F1 is uncertain. The model may still be strong, but the limited number of test examples makes the final estimate less precise.

This is why confidence intervals are necessary in the report.

---

### 9.3 Threshold-dependent metrics are less stable than AUROC

AUROC measures ranking quality across all possible thresholds. F1 and accuracy depend on a selected threshold.

Some categories, such as carpet and pill, have good AUROC but weaker F1. This means the model may rank defective examples reasonably well, but selecting a single operational threshold is still difficult.

For deployment, the threshold should be selected carefully based on the real industrial cost of false positives and false negatives.

---

### 9.4 The project does not yet include full label-fraction ablation

The original project plan included experiments with different label fractions, such as:

- 10% labels
- 50% labels
- 100% labels

However, the completed evaluation scope focuses on 100% labels only.

This should be considered a scope reduction. The current results show supervised full-label performance but do not prove how well the model performs in low-label regimes.

---

### 9.5 The current evaluation is image-level, not true pixel-level segmentation

Although Grad-CAM visualizations provide qualitative localization evidence, this is not the same as a fully supervised segmentation evaluation.

The current system evaluates whether an image is defective, not whether every defective pixel is correctly segmented.

A future version should include pixel-level mask evaluation using ground-truth defect masks where available.

---

### 9.6 The dataset split differs from the standard one-class MVTec protocol

The current project behaves as a supervised binary classification benchmark. This differs from the standard MVTec anomaly detection protocol, where models are usually trained only on normal images and tested on normal plus defective images.

This limitation should be clearly stated so that the results are interpreted as supervised classification results rather than pure one-class anomaly detection results.

---

## 10. GPU-Hour Cost-Benefit Discussion

The training phase involved:

6 categories × 3 folds = 18 training runs

Assuming each Kaggle T4 run took approximately 30 minutes, the total training cost is approximately:

18 × 0.5 GPU-hours = 9 GPU-hours

The final evaluation phase was much cheaper. The quantitative evaluation took approximately 4 GPU-minutes, and the qualitative Grad-CAM galleries took less than 1 additional minute.

This means the evaluation cost was small compared with the training cost.

The use of three-fold cross-validation increased training compute by roughly 3× compared with a single split. However, this cost is justified because it reveals performance variability across folds. Without cross-validation, the team might incorrectly trust a single favorable split.

For example, the weaker capsule test performance shows that some categories are more sensitive to split difficulty. The extra compute cost therefore provided useful evidence about robustness.

Overall, the cost-benefit tradeoff is acceptable:

Higher compute cost → better confidence in model stability and category-level reliability

---

## 11. Test Coverage and Code Quality

Engineer 5 added and updated tests for:

- `tests/test_metrics.py`
- `tests/test_bootstrap.py`
- `tests/test_engineer5_coverage.py`

The final local coverage result for the main evaluation files was:

| File | Coverage |
|---|---:|
| `src/evaluation/bootstrap.py` | 92% |
| `src/evaluation/evaluator.py` | 94% |
| `src/evaluation/metrics.py` | 95% |
| `src/evaluation/qualitative.py` | 90% |
| **Total** | **93%** |

This satisfies the required coverage threshold of more than 85%.

The coverage tests use lightweight mocked or synthetic inputs to exercise the evaluator and qualitative modules without requiring the full MVTec dataset or large trained checkpoints during unit testing.

---

## 12. Final Deliverables

Engineer 5 deliverables completed:

| Deliverable | Status |
|---|---|
| `src/evaluation/metrics.py` | Completed |
| `src/evaluation/bootstrap.py` | Completed |
| `src/evaluation/evaluator.py` | Completed |
| `src/evaluation/qualitative.py` | Completed |
| `tests/test_metrics.py` | Completed |
| `tests/test_bootstrap.py` | Completed |
| `tests/test_engineer5_coverage.py` | Completed |
| `notebooks/04_Evaluation.ipynb` | Completed |
| `docs/REPORT.md` | Completed |

---

## 13. Conclusion

The evaluation confirms that the supervised ResNet-50 defect classifier performs strongly on several MVTec AD categories, especially leather, hazelnut, and bottle. The overall mean AUROC is 0.9554, showing strong image-level ranking performance across the six selected categories.

However, the results also show that performance is category-dependent. Capsule is the weakest category, and threshold-dependent metrics such as F1 are less stable than AUROC. Confidence intervals are necessary because some category test sets are small.

The final evaluation infrastructure is complete, tested, and suitable for reporting. It provides quantitative metrics, uncertainty estimates, diagnostic plots, Grad-CAM visualizations, and failure-case analysis.

---

## References

1. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

2. Bergmann P, Fauser M, Sattlegger D, Steger C. MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

3. Lin TY, Goyal P, Girshick R, He K, Dollár P. Focal Loss for Dense Object Detection. Proceedings of the IEEE International Conference on Computer Vision. 2017.

4. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Proceedings of the IEEE International Conference on Computer Vision. 2017.

5. Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall/CRC. 1993.