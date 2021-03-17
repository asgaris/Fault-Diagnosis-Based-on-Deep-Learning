from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
import numpy as np
import pandas as pd
from keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

## Import training data
def load_train_data ():
    df1 = pd.read_csv(r'C:\Users\Hamidreza\Desktop\Sahar\Final_Data\Classification.csv')
    return df1

## Import testing data
def load_test_data ():
    df2 = pd.read_csv(r'C:\Users\Hamidreza\Desktop\Sahar\Final_Data\Test.csv')
    return df2

Test = load_test_data ()
Temp = load_train_data ()
################

X_train = Temp.iloc[:,1:501].to_numpy()
y_train = Temp.iloc[:,533:540].to_numpy() 
X_train = X_train.reshape(156,20,25)    # X_train.reshape(samples, timesteps, features)


X_test = Test.iloc[:,1:501].to_numpy()
X_test = X_test.reshape(60,20,25)

## RNN 
def create_model(dropout=0.2, learn_rate =0.1):
	# create model
    RNN = Sequential()
    RNN.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(20,25)))
    RNN.add(Dropout(dropout))
    RNN.add(LSTM(units=40, activation='relu'))
    RNN.add(Dropout(dropout))
    RNN.add(Dense(7, activation='softmax'))
    opt = keras.optimizers.Adamax(learning_rate=learn_rate)
    # Compile model
    RNN.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return RNN

RNN = KerasClassifier(build_fn=create_model, verbose=1)
grid = {'nb_epoch': [100, 150, 200, 300],
        'batch_size': [5, 10, 20],
        'dropout': [0.2, 0.25, 0.3],
        'learn_rate': [0.0001, 0.001, 0.01]
       }

kfold = KFold(n_splits=5, random_state=None, shuffle=True)
search = GridSearchCV(estimator = RNN, param_grid = grid, cv=kfold)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
grid_result = search.fit(X_train, y_train, verbose=True, callbacks=[es]) 
# Best solution
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


## Refit the same RNN with the best parameters from above GridSearchCV
model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(20,25)))
model.add(LSTM(units=40, activation='relu'))
model.add(Dropout(grid_result.best_params_['dropout']))
model.add(Dense(units=7, activation='softmax'))

opt = keras.optimizers.Adamax(learning_rate=grid_result.best_params_['learn_rate'])
# Compile model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=grid_result.best_params_['nb_epoch'], 
          batch_size=grid_result.best_params_['batch_size'])  #validation_data=(X_val, y_val), 
#########################################


# Prediction for two simultaneous failures
print ('\n Failures prediction')
result = model.predict(X_test)
print(result)
result = pd.DataFrame(result)
# Export the results to a csv file
export_csv = result.to_csv('Results.csv')
