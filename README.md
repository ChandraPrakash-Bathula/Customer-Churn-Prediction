# Customer-Churn-Prediction
 "Customer Churn Prediction: A Comparative Analysis of Models with and without Sentiment Analysis ":

```markdown
# Customer Churn Prediction: Comparative Analysis with Sentiment Analysis (Phase 2)

## Overview of the Study
This research aims to evaluate the impact of sentiment analysis on predicting customer churn within the e-commerce sector using various predictive modeling techniques. The study particularly assesses which model most accurately predicts customer churn rates.

## Process Involved in the Study
The project is split into two phases:
1. **Phase 1:** Selection of relevant variables believed to influence customer churn and development of predictive models without customer feedback.
2. **Phase 2:** Incorporation of customer feedback to extract sentiment scores, combined with variables from EDA, to enhance the predictive models.

The effectiveness of these models, with and without sentiment analysis, is compared to understand if sentiment analysis aids in better predicting why customers may discontinue transactions with the organization.

## Research Question
"How does sentiment analysis impact predicting the customer churn of an organization?"

## Current Implementation (Phase 2)
- **Models Implemented:** In addition to previous models (Logistic Regression, SVM, etc.), a Naive Bayes model is introduced.
- **Enhancements:**
  - Increased cross-validation folds from 5 to 10.
  - Optimized training time and memory consumption.
  - Added additional hyperparameters for SVM tuning.
  - Evaluated models using Sensitivity, Specificity, and ROC_AUC scores.
  - Plotted AUC-ROC curves for all models.
  - Created pickle files for model preservation.

### Sentiment Analysis Integration
- **Hugging Face API Connection:**
  ```python
  from huggingface_hub import notebook_login
  notebook_login()
  ```
- **IMDB Dataset Pre-training:**
  ```python
  from datasets import load_dataset
  imdb = load_dataset("imdb")
  ```

### Model Performance Metrics
- **Cross-Validation Scores:**
  - Accuracy, Recall, Precision, F1 Score, ROC_AUC.
- **Random Forest Model Metrics:**
  ```python
  import pandas as pd
  # DataFrame setup
  rf_output = pd.DataFrame({
      'Training': [train_cv_acc_rf, train_cv_recall_rf, train_cv_precision_rf, train_cv_f1_rf, roc_auc_train_rf],
      'Testing': [test_cv_acc_rf, test_cv_recall_rf, test_cv_precision_rf, test_cv_f1_rf, roc_auc_test_rf]
  }, index=['Accuracy', 'Recall', 'Precision', 'F1', 'ROC_AUC'])
  print(rf_output)
  ```

### Feature Importance Analysis
- **Variable Importances for Random Forest and LightGBM:**
  ```python
  import matplotlib.pyplot as plt
  # Plot feature importance
  plt.barh(sorted_feature_names, sorted_feature_importances)
  plt.xlabel('Feature Importance')
  plt.title('Variable Importance for Random Forest')
  plt.show()
  ```

### Next Steps
- Interpretation and comparison of model results to determine the efficacy of including sentiment analysis in predicting customer churn.

## Support
Support our work by starring our GitHub repository.

## Conclusion
This phase of the study explores how incorporating customer sentiment analysis affects churn prediction accuracy across different models. The findings from this phase will guide future research directions and potential improvements in customer retention strategies.
```

