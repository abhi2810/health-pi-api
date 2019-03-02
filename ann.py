#ANN


#Data Preprocessing 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 4:7].values
Y = dataset.iloc[:, 7].values


# Encoding categorical data
# Encoding the Independent Variable
# =============================================================================
# 
# from sklearn.preprocessing import LabelEncoder
# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y)
# 
# =============================================================================
# =============================================================================
# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features = [1])
# X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]
# to unlock ctrl + 5
# =============================================================================
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)






# Feature Scaling

# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# 
# =============================================================================

#making ANN

import keras
from keras.models import Sequential  #to initialize Neural network
from keras.layers import Dense #for the layers



#Dropout regularization is the solution for overfitting i.e. when the model learns too much, when overfitting
#occurs there is much higher accuracy in training set than the test set
#high variance means overfitting
#solution for overfitting i.e. when we see high variance


      
#initializing ANN
classifier = Sequential()

#Adding input layer and first hidden layer with dropout
classifier.add(Dense(units = 9, activation = 'relu', input_dim = 3))
# =============================================================================
# #classifier.add(Dropout(p= 0.1))  #output dim or units is the number of nodes in the
# #hidden layer and input_dim is the number of nodes in input layer we need to specify the number of input nodes as
# #the hidden layer do not know how many inputs to expect relu is for rectifier activation function
# #output_dim is most of the times average of all the independent and dependnt variable.
# 
# #Adding the second hidden layer
# 
# classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) #here we dont need to specify the input
# #classifier.add(Dropout(p= 0.1))
# #nodes as it know the number of inputs to expect
# classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
# #classifier.add(Dropout(p= 0.1))
# 
# classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
# #classifier.add(Dropout(p= 0.1))
# =============================================================================



#Adding the  layers
classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
classifier.add(Dense(1)) 

#Compiling the ANN



classifier.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'] ) #optimizer takes in the algorithm that we want to use in our ANN
 
 
#fitting the ANN to the training set

classifier.fit(X_train, Y_train, batch_size = 1, epochs = 100)

classifier.fit(X_test, Y_test, batch_size = 1, epochs = 100)
# =============================================================================
# getting 80% accuracy for both training and test dataset
# =============================================================================

#making the prediction

# Predicting the Test set results

Y_pred = classifier.predict(X_test) 
#Y_pred = (Y_pred > 0.5) 

classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("classifier.h5")









































# =============================================================================
# 
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_test, Y_pred)
# acc = (85+27)/200 # to check if our model is correct bcoz acc value and the accuracy we got from training our model is same.
# 
# # =============================================================================
# # # =============================================================================
# # # #predicvtion for a particular individual
# # # new_predict = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 6000, 2, 1, 1, 50000]]))) #2 brackets will help to create a 2d array
# # # #the order should be same as in the dataset
# # # new_predict = (new_predict > 0.5)
# # # =============================================================================
# # 
# # 
# # #Evaluating the ANN using K-fold cross validation it divides the ANN in K-folds i.e. 10 here, 
# # #and then using 10 different combination it trains the data on 9 folds and tests it on the 10th fold 
# # #hence we get the variance or accuracy of our model i.e. it will be trained 10 tinmes to get the accuracy 
# # #hence n_jobs will decide the number of CPU's to use for fast computation but not able to use paralle computing 
# # #so 1 otherwise it should be -1
# # 
# # import keras
# # from keras.models import Sequential  #to initialize Neural network
# # from keras.layers import Dense #for the layers
# # from keras.wrappers.scikit_learn import KerasClassifier
# # from sklearn.model_selection import cross_val_score
# # def build_classifier():
# #     classifier = Sequential()
# #     classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# #     classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# #     classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) 
# #     classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
# #     return classifier
# # classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# # accuracies = cross_val_score(classifier, X = X_train, y = Y_train, cv = 10, n_jobs = 1)
# # mean = accuracies.mean()
# # variance = accuracies.std()
# # 
# # 
# # #Dropout regularization is the solution for overfitting i.e. when the model learns too much, when overfitting
# # #occurs there is much higher accuracy in training set than the test set
# # #high variance means overfitting
# # #solution for overfitting i.e. when we see high variance
# # 
# # # Tuning of ANN 
# # 
# # 
# # import keras
# # from keras.models import Sequential  #to initialize Neural network
# # from keras.layers import Dense #for the layers
# # from keras.wrappers.scikit_learn import KerasClassifier
# # from sklearn.model_selection import GridSearchCV
# # def build_classifier(optimizer):
# #     classifier = Sequential()
# #     classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# #     classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# #     classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) 
# #     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'] )
# #     return classifier
# # classifier = KerasClassifier(build_fn = build_classifier)
# # param = {'batch_size' : [25,32], 
# #          'epochs' : [100,500],
# #          'optimizer' : ['adam', 'rmsprop']} #tuning different parameters
# # 
# # gridsearch = GridSearchCV(estimator = classifier,
# #                           param_grid = param,
# #                           scoring = 'accuracy',
# #                           cv = 10)
# # gridsearch = gridsearch.fit(X_train, Y_train)
# # best_param = gridsearch.best_params_
# # best_accuracy = gridsearch.best_score_
# # 
# # 
# # 
# # # Making the Confusion Matrix
# # 
# # from sklearn.metrics import confusion_matrix
# # cm = confusion_matrix(Y_test, Y_pred)
# # 
# # =============================================================================
# 
# =============================================================================
