from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Flatten
import pandas as pd
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.layers import GaussianNoise
from keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM

## Import training data
def load_temp_data ():
    df1 = pd.read_csv(r'C:\Users\Hamidreza\Desktop\Sahar\Final_Data\Classification.csv')
    return df1

## Import testing data
def load_test_data ():
    df2 = pd.read_csv(r'C:\Users\Hamidreza\Desktop\Sahar\Final_Data\Test.csv')
    return df2

Test = load_test_data ()
Temp = load_temp_data ()
################

X_train = Temp.iloc[:,1:526]
y_train = Temp.iloc[:,533:540]


## Train Validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)
#################


## Reshape train and validation into 5*5 images
X_train = X_train.values.reshape(-1, 1, 5, 5, 21)  ## (number of samples, time_steps, frame_row, frame_col, number of channels)
X_val = X_val.values.reshape(-1, 1, 5, 5, 21)
################

## CNN 
def create_model(dropout=0.2, learn_rate =0.1):
	# create model
    model = Sequential()
    #Add Gaussian white noise
    model.add(GaussianNoise(0.01, input_shape=X_train.shape[1:]))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(70, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(7, activation='softmax'))

    
    opt = keras.optimizers.Adamax(learning_rate=learn_rate)
    # Compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter study to find the best parameters
model = KerasClassifier(build_fn=create_model, verbose=1)
grid = {'nb_epoch': [150, 200, 300],
        'batch_size': [5, 10, 20],
        'dropout': [0.2, 0.25],
        'learn_rate': [0.0001, 0.001, 0.01]
       }

kfold = KFold(n_splits=5, random_state=None, shuffle=True)
search = GridSearchCV(estimator = model, param_grid = grid, cv=kfold)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
grid_result = search.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=True, callbacks=[es])   
## Best solution
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


#Refit the same CNN-LSTM with the best parameters from above GridSearchCV
model = Sequential()
#Add Gaussian white noise
model.add(GaussianNoise(0.01, input_shape=X_train.shape[1:]))
model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Conv2D(filters=32, kernel_size=2, activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=2)))
model.add(Dropout(grid_result.best_params_['dropout']))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(70, activation='relu'))
model.add(Dropout(grid_result.best_params_['dropout']))
model.add(Dense(7, activation='softmax'))

opt = keras.optimizers.Adamax(learning_rate=grid_result.best_params_['learn_rate'])
# Compile model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=grid_result.best_params_['nb_epoch'], 
          batch_size=grid_result.best_params_['batch_size'])
#########################################


# Prediction for two simultaneous  failures
# Fan 1 and 2
X_Test = Test.iloc[0:15,1:526]
X_Test = X_Test.values.reshape(15, 1, 5, 5, 21)
print ('\n fan 1 and 2')
print(model.predict(X_Test))
##############

# Fan 1 and 6
X_Test = Test.iloc[15:30,1:526]
X_Test = X_Test.values.reshape(15, 1, 5, 5, 21)
print ('\n fan 1 and 6')
print(model.predict(X_Test))
##############

# Fan 2 and 4
X_Test = Test.iloc[30:45,1:526]
X_Test = X_Test.values.reshape(15, 1, 5, 5, 21)
print ('\n fan 2 and 4')
print(model.predict(X_Test))
##############

# Pump and Fan 1
X_Test = Test.iloc[45:60,1:526]
X_Test = X_Test.values.reshape(15, 1, 5, 5, 21)
print ('\n Pump and Fan 1')
print(model.predict(X_Test))
