# Fraud Detection Model

This repository contains code and resources for a fraud detection model. The model is designed to classify transactions as either fraudulent or non-fraudulent.

## Data Dictionary:
step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

amount - amount of the transaction in local currency.

nameOrig - customer who started the transaction

oldbalanceOrg - initial balance before the transaction

newbalanceOrig - new balance after the transaction

nameDest - customer who is the recipient of the transaction

oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

## Data Cleaning
The initial data cleaning process includes handling missing values, outliers, and multi-collinearity. This involves imputing missing values, removing duplicates, and more.

## Exploratory Data Analysis (EDA)
The exploratory data analysis (EDA) is performed to understand the underlying patterns and trends in the data. This involves visualizing the distribution of different features, identifying correlations between features, and identifying any anomalies or unusual patterns in the data.

##Model Building
The fraud detection model is built using a supervised learning approach, specifically a binary classification model. Techniques such as logistic regression and decision trees are used to classify transactions as either fraudulent or non-fraudulent. The model's performance is optimized using techniques such as over-sampling.

## Model Evaluation
The performance of the model is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. A confusion matrix, precision-recall curve, and ROC curve are used to evaluate the performance of the model.

## Key Factors for Fraud Detection
The key factors that predict fraudulent customers include transaction amount, customer demographics, and previous transaction history. Large transactions and transactions made by customers with a history of fraud are both indicators that a transaction may be fraudulent.

## Prevention Measures
Prevention measures that can be adopted include implementing fraud detection systems, monitoring transactions for suspicious activity, and educating customers on how to spot and report fraud. Companies can also update their infrastructure to include advanced security measures such as multi-factor authentication and encryption.

## Determining Effectiveness
To determine if the prevention measures are effective, companies can monitor the number of fraudulent transactions and compare it to the number of fraudulent transactions before the prevention measures were implemented. Additionally, customer complaints about fraud can be tracked and monitored for changes. A decrease in the number of fraudulent transactions and customer complaints about fraud would indicate that the prevention measures are working.
