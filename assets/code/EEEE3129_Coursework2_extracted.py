# ===== Cell 5 =====
# DO NOT CHANGE THIS PART

import pandas as pd # pandas module required for dataset manipulation
import numpy as np # numpy module required for numerical calculation, signal processing and training.
from sklearn.linear_model import LinearRegression # the model LinearRegression from sklearn.linear_model is required for Part 2 
from sklearn.model_selection import train_test_split # the function train_test_split is required from sklearn.model_selection for training and testing dataset preparation.
import matplotlib.pyplot as plt
###################################################
## Write your code for Task 1.1 here. 
def UKdata(file_name = 'UKLoad2023.json'):
    df = pf.read_json(file_name)
    return df



###################################################
## Write the set-up Code for the whole notebook. Write necessary codes including module import, self-defined supporting functions and general configurations for the notebook.

# ===== Cell 6 =====
# these two functions are mandatory, and you should NOT change it. Modifying these two functions will lead to 0 marks for any assessment calling these two functions.
def performance_indicator_relative(mse_train, mse_test, mse_validation):
    return abs(mse_validation - mse_train) / abs(mse_train) + abs(mse_validation - mse_test) / abs(mse_test)

def performance_indicator_rmse(mse_validation, Y_labels):
    return mse_validation/(np.mean(Y_labels)**2)

# ===== Cell 8 =====
## Code for Part 1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Load the training dataset
df = pd.read_json('UKLoad2023.json')

###################################################
## Write your code for Task 1.2 here. 
duplicated_rows = df.duplicated()
print(duplicated_rows.sum())

df.drop_duplicates(inplace = True)
df.sort_values('publishTime', inplace = True)
df.drop_duplicates(subset = ['settlementDate','settlementPeriod'], keep = 'last', inplace = True)

###################################################
## Write code to validate the integrity of the dataset by checking for duplicate data and applying appropriate solutions.
#
###################################################
## Write your code for Task 1.3 here. 
missingVal = df.isnull().sum()
df['quantity'].fillna(df['quantity'].median(), inplace = True) ##ffill method to replace missing values
df['startTime'].fillna(method = 'ffill', inplace = True)
df['settlementDate'].fillna(method = 'ffill', inplace = True)
df['settlementPeriod'].fillna(df['settlementPeriod'].median(), inplace = True)
###################################################
## Write code to validate the integrity of the dataset by checking missing values (NaNs) and applying appropriate solutions.
#
###################################################
## Write your code for Task 1.5 here. 
###################################################
# Write code to prepare the following two dataset: the feature dataset with the name of "feature_dataset", which should be 
# a 2-dimensional numpy ndarray format, each row is a different day, and each row contains 48 values from the field "quantity" 
# in that day; the label dataset with the name of "label_dataset", which is with a similar format of the "feature_dataset", 
# but the date for each row should be one day later. For example, if row 3 in "feature_dataset" is the data for 2023-8-20, 
# then row 3 in "label_dataset" should be the data for 2023-8-21. This will prepare the "feature_dataset" and "label_dataset"
# ready for the day-ahead model training and validation purpose.
df.sort_values(['settlementDate', 'settlementPeriod'], inplace = True)
uniqueDay = df['settlementDate'].unique()
 ## checking for nans
dailyData = []
for day in uniqueDay:
    dayRows = df[df['settlementDate'] == day]
    quantities = dayRows['quantity'].values   
    if len(quantities) == 48:
        dailyData.append(quantities)
        
dailyData = np.array(dailyData)

feature_dataset = dailyData[:-1, :]  ##creates an array to hold feature and label
label_dataset = dailyData[1:, :]

# dataset initialization - you might need to rewrite/remove this part if necessary
#feature_dataset = np.ones((10,10)) # - you might need to rewrite/remove this part if necessary
#label_dataset   = np.ones((10,10)) # - you might need to rewrite/remove this part if necessary

#

###################################################
## Write your code for Task 1.6 here. 
day_idx = 1
plt.figure() ##plot showing the feature and label dataset
plt.plot(feature_dataset[day_idx, :], label = 'featuredata')
plt.plot(label_dataset[day_idx, :], label = 'labeldata')
plt.xlabel('settlement 48')
plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()

###################################################
# Write code to visualize the day-ahead forecasting problem, by plotting the day-ahead data from  "feature_dataset" and the corresponding output data from "label_dataset".

#

# ===== Cell 11 =====
# Example code for dataset splitting, training, and testing. You may need to rewrite/reuse this routine for this or later parts.
# you should have this part working for the first run, to check your variables are properly named in Part 1.
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Training and Testing dataset preparation. Splitting ratio between training and testing is set as 50% vs 50% as an example.
X_train, X_test, Y_train, Y_test = train_test_split(feature_dataset, label_dataset, test_size=1)

# Configuring the models to fit/train. 
cwk_model_part2 = LinearRegression() ## using the linear regression model

# Fitting the models with the training data.
cwk_model_part2.fit(X_train, Y_train)

# Making predictions with the fitted model
Y_train_output = cwk_model_part2.predict(X_train)

# Making predictions with the fitted model
Y_test_output= cwk_model_part2.predict(X_test)

###################################################
## Write your code for Task 2.1 here. 
###################################################
# Write code to calculate the Mean Squared Error (MSE) performance for both the training part and the testing part, with the name of mse_train and mse_test, correspondingly. You will need to follow this naming convention for Part 3.

# value initialization - you might need to rewrite/remove this part if necessary
mse_train = mean_squared_error(Y_train, Y_train_output) # - you might need to rewrite/remove this part if necessary
mse_test = mean_squared_error(Y_test, Y_test_output) # - you might need to rewrite/remove this part if necessary
 ## using MSE to improve generalisation and overfitting
#


###################################################
# the following code should run with no problem after your implementation of Task 2.1
print("---------------------------------------------------------------------------")
print("MSE Performance for Part 2:")
print(f"Linear Regression MSE Train: {mse_train}")
print(f"Linear Regression MSE Test: {mse_test}")
print("---------------------------------------------------------------------------")

###################################################
## Write your code for Task 2.2 here. 
#plt.figure()
#plt.plot(Y_train[0], label = 'Y_Train')
#plt.plot(Y_train_output[0], label = 'Y_train_output')
#plt.xlabel('settlement 1-48')
#plt.ylabel('')
#plt.legend()
#plt.tight_layout()
#plt.show()

#plt.figure()
#plt.plot(Y_test[0], label = 'Y_test')
#plt.plot(Y_test_output[0], label = 'Y_test_output')
#plt.xlabel('settlement 1-48')
#plt.ylabel('')
#plt.legend()
#plt.tight_layout()
#plt.show()
###################################################
# Write code to visualize the model fitting performance. You should plot the first row of Y_train and Y_train_output, first row of Y_test_output and Y_test.
plt.figure()
plt.plot(Y_test[0], label = 'Y_test')
plt.plot(Y_test_output[0], label = 'Y_test_output')
plt.plot(Y_train[0], label = 'Y_Train')
plt.plot(Y_train_output[0], label = 'Y_train_output')
plt.xlabel('settlement 1-48')
plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()
#

# ===== Cell 14 =====
###################################################
## Write your code for Task 3.1 here. 
###################################################
# Write your code to implement a different model (any model except LinearRegression model) and evaluate its performance. You should rewrite part of the following code.
# They are provided to give you hints about the model names to be expected in Part 4.
# The following code should be rewrite completely. 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(feature_dataset, label_dataset, test_size=1, random_state = 0)

scaler = StandardScaler() ## standard scaler comes with MLP package and iomproves generalisation
xTrainScaled = scaler.fit_transform(X_train)
xTestScaled = scaler.transform(X_test)

# Configuring the models to fit/train. 
cwk_model_part3 = MLPRegressor(hidden_layer_sizes = (100, 500), activation = 'relu', solver = 'adam', learning_rate_init = 0.001, max_iter = 2000, random_state = 42)  ## can change random state but 42 causes the least overfitting
## learning rate and max iterations are standard values
# Fitting the models with the training data.
cwk_model_part3.fit(xTrainScaled, Y_train)

# Making predictions with the fitted model
Y_train_output = cwk_model_part3.predict(xTrainScaled)

# Making predictions with the fitted model
Y_test_output= cwk_model_part3.predict(xTestScaled)


print("train", mse_train)
print("test", mse_test)

print("Ytrain", Y_train_output.shape)
print("Ytest", Y_test_output.shape)

#plt.figure()
#plt.plot(Y_train[0], label = 'Y_Train')
#plt.plot(Y_train_output[0], label = 'Y_train_output')
#plt.xlabel('settlement 1-48')
#plt.ylabel('')
#plt.legend()
#plt.tight_layout()
#plt.show()

#plt.figure()
#plt.plot(Y_test[0], label = 'Y_test')
#plt.plot(Y_test_output[0], label = 'Y_test_output')
#plt.xlabel('settlement 1-48')
#plt.ylabel('')
#plt.legend()
#plt.tight_layout()
#plt.show()

plt.figure()
plt.plot(Y_test[0], label = 'Y_test')
plt.plot(Y_test_output[0], label = 'Y_test_output')
plt.plot(Y_train[0], label = 'Y_Train')
plt.plot(Y_train_output[0], label = 'Y_train_output')
plt.xlabel('settlement 1-48')
plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()
#

# ===== Cell 17 =====
# To load the evaluation dataset
df_evaluation = pd.read_json('UKLoad2023_test.json')

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
###################################################
## Write your code for Task 4.1 here. 
###################################################
df_evaluation.drop_duplicates(inplace = True)  ## same code as part 1 however uses a different df
df_evaluation.sort_values('publishTime', inplace = True)
df_evaluation.drop_duplicates(subset = ['settlementDate','settlementPeriod'], keep = 'last', inplace = True)

###################################################
## Write code to validate the integrity of the dataset by checking for duplicate data and applying appropriate solutions.
#
###################################################
## Write your code for Task 1.3 here. 
df_evaluation['quantity'].fillna(df_evaluation['quantity'].median(), inplace = True)
df_evaluation['startTime'].fillna(method = 'ffill', inplace = True)
df_evaluation['settlementDate'].fillna(method = 'ffill', inplace = True)
df_evaluation['settlementPeriod'].fillna(df_evaluation['settlementPeriod'].median(), inplace = True)

df_evaluation.sort_values(['settlementDate', 'settlementPeriod'], inplace = True)
uniqueDay = df_evaluation['settlementDate'].unique()

dailyData = []
for day in uniqueDay:
    dayRows = df_evaluation[df_evaluation['settlementDate'] == day]
    quantities = dayRows['quantity'].values
    if len(quantities) == 48:
        dailyData.append(quantities)
        
dailyData = np.array(dailyData)

feature_dataset = dailyData[:-1, :]
label_dataset = dailyData[1:, :]

# dataset initialization - you might need to rewrite/remove this part if necessary
feature_dataset_validation = feature_dataset # - you might need to rewrite/remove this part if necessary
label_dataset_validation   = label_dataset # - you might need to rewrite/remove this part if necessary

# initialize the mse_validation - you might need to rewrite/remove this part if necessary
YValid = cwk_model_part2.predict(feature_dataset_validation)
mse_validation = mean_squared_error(label_dataset_validation, YValid) # - you might need to rewrite/remove this part if necessary


print("mse", mse_validation)

# ===== Cell 18 =====
#################################################################################################################
## Performance Evaluation Part
# Please leave this part un-touched. 
# They are provided for you to test if your code can be evaluated - the performance reported below is for
# your reference only and not the ones in marking. The dataset for marking is NOT available to you so don't ask for it.
overall_performance = performance_indicator_relative(mse_train, mse_test, mse_validation)
print('overall_performance', overall_performance)

rmse_performance = performance_indicator_rmse(mse_validation, label_dataset_validation)
print('rmse_performance', rmse_performance)
#################################################################################################################

