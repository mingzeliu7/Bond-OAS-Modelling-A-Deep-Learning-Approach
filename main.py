import util
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import tensorflow as tf

'''Data Preprocessing'''
dataset_original = pd.read_csv(
    r"C:\Users\Michael\OneDrive\Desktop\Stanford AI\CS229 - Machine Learning\Bond Price Predictions Using Machine Learning Models\dataset_original.csv")
dataset_original = dataset_original.dropna()
dataset_original.to_csv('dataset_dropna.csv')
x_train, y_train = util.load_dataset("dataset_dropna.csv")  # Load dataset # np.shape(x_train) : (n,d)
x_train = np.delete(x_train, 0, axis=1)  # Remove id column
x_train = np.delete(x_train, 0, axis=1)  # Remove bond id column
scaler = StandardScaler()  # Standardize data
scaler.fit(x_train)  # Calculate mean and variance of entire dataset used for scaling
# Perform standardization by centering and scaling by subtracting mean and dividing standard deviation
x_train = scaler.transform(x_train)
pca = PCA(0.95)  # Dimension reduction using principal component analysis (PCA) by retaining 95% variance
x_train = pca.fit_transform(x_train)  # 25 features were selected by PCA: np.shape(x_train) = (n,26)

'''Load updated dataset containing manually selected 25 features'''
x_train, y_train = util.load_dataset("dataset_25features.csv")
# Standardization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

'''Baseline linear regression by minimizing the squared error (LMS) using stochastic gradient descent (SGD)'''
# Split data 70/30 training/testing with shuffling
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=0.3, train_size=0.7, shuffle=True)
# Split the above training and testing set into 5 equally sized training and testing sets without shuffling
train_fold_increment = np.shape(x_train1)[0] // 5
test_fold_increment = np.shape(x_test1)[0] // 5
# Perform linear regression using SGD with batch size of 1 to minimize lost function
generalized_error = []  # Vector containing all the mean absolute errors of all the different models w.r.t learning rate
for learning_rate in [0.00001, 0.00002, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
    # Hyperparameter tuning for learning rate
    sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=learning_rate, max_iter=1000000000)

    y_pred_temp = []
    for fold in range(5):  # Train and test each model on the 5 training and 5 test sets
        if fold != 4:
            sgd.fit(x_train1[train_fold_increment * fold: train_fold_increment * (fold + 1)],
                    y_train1[train_fold_increment * fold: train_fold_increment * (fold + 1)])
            y_pred_temp.append(
                (sgd.predict(x_test1[test_fold_increment * fold: test_fold_increment * (fold + 1)])).tolist())
        else:
            sgd.fit(x_train1[train_fold_increment * fold: np.shape(x_train)[0]],
                    y_train1[train_fold_increment * fold: np.shape(x_train)[0]])
            y_pred_temp.append((sgd.predict(x_test1[test_fold_increment * fold: np.shape(x_test1)[0]])).tolist())
    y_pred = []
    # Mean absolute error computation
    for i in range(5):
        for j in y_pred_temp[i]:
            y_pred.append(j)
    y_pred = np.array(y_pred)
    abs_error = np.abs(np.subtract(y_test1, y_pred))
    generalized_error.append(np.sum(abs_error) / np.shape(y_pred)[0])
print(generalized_error)
# Running the above numerous times, we decided alpha of 0.00002 is the most suitable for the base regression model
# Train the entire 70% split training set with alpha = 0.00002
sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=0.00002, max_iter=1000000000)
sgd.fit(x_train1, y_train1)
# Test the entire 30% split testing set
model_y_pred = sgd.predict(x_test1)
# MAE computation
model_abs_error = np.abs(np.subtract(y_test1, model_y_pred))
model_mean_absolute_error = np.sum(model_abs_error) / np.shape(model_y_pred)[0]
print(model_mean_absolute_error)

'''Neural networks'''
generalized_error = []  # Vector containing all the mean absolute errors of all the different models w.r.t hyperparams
for learning_rate in [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]:
    for batch_size in [1, 16, 32, 64, 128, 256]:
        for num_neurons in [[64, 32, 16, 1], [5, 5, 5, 1]]:
            ann = tf.keras.Sequential()
            input_shape = [np.shape(x_train1)[1]]
            y_pred_temp = []
            for fold in range(5):  # Train and test each model on the 5 training and 5 test sets
                if fold != 4:
                    x = x_train1[train_fold_increment * fold: train_fold_increment * (fold + 1)]
                    y = y_train1[train_fold_increment * fold: train_fold_increment * (fold + 1)]
                    x_t = x_test1[test_fold_increment * fold: test_fold_increment * (fold + 1)]
                else:
                    x = x_train1[train_fold_increment * fold: np.shape(x_train)[0]]
                    y = y_train1[train_fold_increment * fold: np.shape(x_train)[0]]
                    x_t = x_test1[test_fold_increment * fold: np.shape(x_test1)[0]]
                ann.add(tf.keras.layers.Dense(units=num_neurons[0], activation='relu', input_shape=input_shape))
                ann.add(tf.keras.layers.Dense(units=num_neurons[1], activation='relu'))
                ann.add(tf.keras.layers.Dense(units=num_neurons[2], activation='relu'))
                ann.add(tf.keras.layers.Dense(units=num_neurons[3], activation='sigmoid'))
                ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mae')
                ann.fit(x, y, batch_size=batch_size, epochs=25)
                y_pred = ann.predict(x_t)
                y_pred = y_pred.reshape((-1,))
                y_pred_temp.append(y_pred.tolist())
            # Mean absolute error computation
            y_pred = []
            for i in range(5):
                for j in y_pred_temp[i]:
                    y_pred.append(j)
            y_pred = np.array(y_pred)
            abs_error = np.abs(np.subtract(y_test1, y_pred))
            generalized_error.append(np.sum(abs_error) / np.shape(y_pred)[0])
# Running the above numerous times, we decided learning rate of 0.01 is the most suitable for the ANN model
# Batch size = 256
# Neurons in each layer: [64, 32, 16, 1]
# Train the entire 70% split training set with alpha = 0.01
ann = tf.keras.Sequential()
input_shape = [np.shape(x_train1)[1]]
# Test the entire 30% split testing set
ann.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape))
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae')
ann.fit(x, y, batch_size=batch_size, epochs=100)
y_pred = ann.predict(x_test1)
y_pred = y_pred.reshape((-1,))
y_pred_temp.append(y_pred.tolist())
# MAE computation
model_abs_error = np.abs(np.subtract(y_test1, y_pred))
model_mean_absolute_error = np.sum(model_abs_error) / np.shape(y_pred)[0]
print(model_mean_absolute_error)