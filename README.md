# Customer-Churn-Prediction
 "Customer Churn Prediction: A Comparative Analysis of Models with and without Sentiment Analysis ":

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

### IMDB Dataset Pre-training
from datasets import load_dataset
imdb = load_dataset("imdb")

### Model Performance Metrics
#### Cross-Validation Scores:
Accuracy, Recall, Precision, F1 Score, ROC_AUC.
Random Forest Model Metrics:

