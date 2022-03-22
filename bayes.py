#Import scikit-learn dataset library
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
#Load dataset
data = np.load("vec.npy")
label = np.ones(np.shape(data)[0])
label[:15000] = 0
print("Datashape",np.shape(data))
#print("MEAN",np.mean(data,axis=0))
#print("VAR",np.var(data,axis=0))
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.3,random_state=109) # 70% training and 30% test
print("Train and test shape",np.shape(X_train),np.shape(X_test))
from sklearn.naive_bayes import GaussianNB

#Create a svm Classifier
logisticRegr =  GaussianNB()

#Train the model using the training sets
logisticRegr.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = logisticRegr.predict(X_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
print("f1 score:",metrics.f1_score(y_test, y_pred))
#plot_confusion_matrix(logisticRegr, X_test, y_test)  
#plt.show()