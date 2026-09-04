# Credit Card Fraud Detection using Logistic Regression


### 📌 Project Overview

This project detects whether a credit card transaction is **Normal** or **Fraudulent** using Machine Learning.

The project uses **Logistic Regression**, a **classification algorithm**.

- ```0``` → Normal Transaction

- ```1``` → Fraud Transaction


#


### 📂 Dataset

The project uses a synthetic, beginner-friendly practice dataset named ```creditcard_balanced.csv```.

The dataset contains 1000 transactions:

- 500 Normal transactions (```Class = 0```)

- 500 Fraud transactions (```Class = 1```)


### Logistic Regression

Logistic Regression is used for classification problems where the output belongs to categories.

Here:

- 0 = Normal

- 1 = Fraud



**Features**

- ```Time``` – Transaction time

- ```Amount``` – Transaction amount

- ```V1 to V5``` – Transaction-related numerical features

- ```Class``` – Target/output column

<br />


> [!NOTE]
> This is a synthetic practice dataset, not the original real-world credit card fraud dataset. Model results should therefore be treated as practice results.


#



### Logistic Regression

Logistic Regression is used for classification problems where the output belongs to categories.

**HERE :**

- 0 = Normal

- 1 = Fraud


# 


### 🧠 Model Training


The Logistic Regression model is trained using the training data.

*from sklearn.linear_model import LogisticRegression*

*model = LogisticRegression()*
*model.fit(X_train, y_train)*


#


### 🔮 Prediction

The trained model is used to predict whether a transaction is Normal or Fraudulent.

*prediction = model.predict(new_transaction)*

**Output:**

- ```[0]``` → Normal Transaction

- ```[1]``` → Fraud Transaction


#

