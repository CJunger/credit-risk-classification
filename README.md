# credit-risk-classification
Train and evaluate a model based on loan risk.

## Overview of the Analysis


In this analysis we take data about the health of loans and the outcomes to train a model to predict wether loans will be healthy or default using a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
The dataset includes the following features: loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, total debt, loan status (wether it is considered a health or high-risk loan).  
The variable we were trying to predict was the loan status which was categorized as either "healthy" or high-risk". 
First we split the data into a training and a testing datasets.  Then we used a Logistic Regression model with the original data, first fitting the model using the training data (X_train and y_train), then saving the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model, and finally evalutating the model’s performance by calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.
This model predicted the healthy loans very well with high precision and recall. It predicts the high-risk loans less well as evident in the lower precision and recall. This could be explained by the severe class imbalance.
To counteract this we then used the RandomOverSampler module from the imbalanced-learn library to resample the data ensuring that the labels have an equal number of data points. Then using the resampled data ran the Logistic Regression classifier a second time to fit the model and make predictions.  With the resampled data it predicted very well at over 99% recall but it does not predict signifcantly better with the balanced data in comparison to the unbalanced data.


## Results

Accuracy shows how often a classification ML model is correct overall. 
Precision shows how often an ML model is correct when predicting the target class. 
Recall shows whether an ML model can find all objects of the target class. 
(source https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall#:~:text=Accuracy%20shows%20how%20often%20a,when%20choosing%20the%20suitable%20metric.)

* Machine Learning Model 1:

  * Accuracy 99% 
  * Precision are 100% for healthy and 85% for unhealthy
  * Recall scores are 99% for healthy and  91% for unhealthy

* Machine Learning Model 2:

  * Accuracy 99%
  * Precision are 100% for healthy and 84% for unhealthy
  * Recall scores are 99% for healthy and  99% for unhealthy
  
  Classification Report
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384

## Summary

The models perform pretty similarly. Even though we resampled the data to make it more balanced we did not see a marked improvement in both accuracy and precision. One model really isn't better than the other. 

Performance of the model depends on what we are trying to target, in this case we are targeting credit worthiness of borrowers by seeing if they are at risk of default. The goal is to minimize false positives (maximize precision) because we don't want to lend if they are going to default.  
