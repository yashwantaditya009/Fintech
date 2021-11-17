#!/usr/bin/env python
# coding: utf-8

# #                                 Credit Card Fraud Detection

# The project is to recognize fraudulent credit card transactions so that the customers of credit card companies are not charged for items that they did not purchase.

# # 1. Importing all the necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# # 2. Loading the Data

# In[7]:


data = pd.read_csv(r"C:\Users\CG-DTE\Downloads\microsoft project\Fintech\creditcard.csv")


# #  3. Understanding the Data

# In[8]:


data.head()


# # 4. Describing the Data

# In[9]:


print(data.shape)
print(data.describe())


# #  5. Imbalance in the data

# In[10]:


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# Only 0.17% fraudulent transaction out all the transactions. The data is highly Unbalanced. Lets first apply the models without balancing it and if we don’t get a good accuracy then we can find a way to balance this dataset.

# #  6. Print the amount details for Fraudulent Transaction

# In[18]:


print("Amount details of the fraudulent transaction")
fraud.Amount.describe()


# Here we can clearly see from this, the average Money transaction for the fraudulent ones is more. This makes this problem crucial to deal with.

# # 7.  Print the amount details for Normal Transaction

# In[24]:


print("details of valid transaction")
valid.Amount.describe()


# # 8. Plotting the Correlation Matrix

# In[25]:


corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In the HeatMap we can clearly see that most of the features do not correlate to other features but there are some features that either has a positive or a negative correlation with each other. For example, V2 and V5 are highly negatively correlated with the feature called Amount. We also see some correlation with V20 and Amount. This gives us a deeper understanding of the Data available to us.

# # 9. Separating the X and the Y values

# Dividing the data into inputs parameters and outputs value format

# In[26]:


X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
xData = X.values
yData = Y.values


# # 10. Training and Testing Data Bifurcation

# We will be dividing the dataset into two main groups. One for training the model and the other for Testing our trained model’s performance.

# In[27]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)


# In[32]:


from sklearn.ensemble import RandomForestClassifier as rfc


# In[34]:


rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)


# # 11. Building all kinds of evaluating parameters

# In[35]:


from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
  
n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest classifier")
  
acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))
  
prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))
  
rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))
  
f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))
  
MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is{}".format(MCC))


# # 12. Visulalizing the Confusion Matrix

# In[36]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# As we can see with our Random Forest Model we are getting a better result even for the recall which is the most tricky part.

# In[ ]:




