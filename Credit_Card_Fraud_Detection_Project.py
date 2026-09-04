import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


database = pd.read_csv('creditcard_balanced.csv')


database


database.shape


database.isnull().sum()


plt.scatter(database['Time'],database['Amount'], c=database['Class'], cmap='bwr')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Normal vs Fraud Transactions')
plt.show()


X = database.iloc[:,0:7].values
y = database.iloc[:,7].values


X


y


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)


X_train.shape


y_train.shape


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)


X_train


X_test = scaler.transform(X_test)


X_test


from sklearn.linear_model import LogisticRegression


model = LogisticRegression()


model.fit(X_train,y_train)


y_pred = model.predict(X_test)


y_pred


from sklearn.metrics import accuracy_score


accuracy = accuracy_score(y_test, y_pred)


accuracy


from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred)


cm


plt.imshow(cm)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.colorbar()
plt.show()


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


model.predict([[598, -2.3, 1.2, -0.5, 0.8, 1.1, -0.3]])

