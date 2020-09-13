#import libraries

from core.data import Feature
import pandas as pd

#model selected for classification - refer to notebook on exploration
from sklearn.linear_model import LogisticRegression

#For label encoding
from sklearn import preprocessing

#splitting for traing and testing
from sklearn.model_selection import train_test_split

#to evaluate the model
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

#load data
data = pd.read_csv('data/lang_data.csv')

#text cleaning
feature = Feature()
data = feature.text_cleaning(data)

#Convert features text to vector count
X = feature.feature_engineering(data)

#convert target text to int
encoder = preprocessing.LabelEncoder()
Y = encoder.fit_transform(data['language'])

#split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# instantiate model. refer to the 
#notebook on dat exploaration on how we arrive at the conclusion to choose logistic regression
logreg = LogisticRegression()
# fit model
logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)

# calculate accuracy
print(metrics.accuracy_score(y_test, y_pred_class))

#predicting the whole dataset
y_pred_class = logreg.predict(X)

#reverse label encoding
inverted = encoder.inverse_transform(y_pred_class)

#attaching the predicted column to the original dataset
data['predicted_language'] = inverted

#save the predictions to excel for futher evaluations
data.to_csv('data/language_prediction.csv')

#view data sample
print(data.tail())

print ("Done...")
	
