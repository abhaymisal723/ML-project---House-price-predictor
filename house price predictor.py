
# # Information about data

import pandas as pd
housing = pd.read_csv("data.csv")
housing.head()
housing.info()
housing["CHAS"].value_counts()
housing.describe()



# # Splitting data into train and test

from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(housing, test_size=0.2, random_state=42)
print(f"No. of rows in train set: {len(trainset)}\nNo. of rows in test set:  {len(testset)}")

from sklearn.model_selection import StratifiedShuffleSplit
spt = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in spt.split(housing, housing["CHAS"]):
    strat_trainset = housing.loc[train_index]
    strat_testset = housing.loc[test_index]
strat_trainset["CHAS"].value_counts()
strat_testset["CHAS"].value_counts()
housing = strat_trainset.copy()
housing.shape
housing_labels = strat_trainset["MEDV"].copy()


# # Looking for correlations

corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending = False)


# # Creating pipeline

housing_data = housing.drop("MEDV", axis = 1)
housing_data.shape
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline ([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
housing_tr = my_pipeline.fit_transform(housing_data)
housing_tr.shape


# # Choosing the best model

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
model_rfr = RandomForestRegressor()
model_lr = LinearRegression()
model_dtr = DecisionTreeRegressor()

model_rfr.fit(housing_tr, housing_labels)
model_lr.fit(housing_tr, housing_labels)
model_dtr.fit(housing_tr, housing_labels)

import numpy as np
from sklearn.metrics import mean_squared_error
housing_predictions_rfr = model_rfr.predict(housing_tr)
mse_rfr = mean_squared_error(housing_labels, housing_predictions_rfr)
rmse_rfr = np.sqrt(mse_rfr)
print(f"mse_rfr: {mse_rfr}\nrmse_rfr: {rmse_rfr}")

housing_predictions_lr = model_lr.predict(housing_tr)
mse_lr  = mean_squared_error(housing_labels, housing_predictions_lr )
rmse_lr  = np.sqrt(mse_lr )
print(f"mse_lr: {mse_lr}\nrmse_lr: {rmse_lr}")

housing_predictions_dtr = model_dtr.predict(housing_tr)
mse_dtr = mean_squared_error(housing_labels, housing_predictions_dtr)
rmse_dtr = np.sqrt(mse_dtr)
print(f"mse_dtr: {mse_dtr}\nrmse_dtr: {rmse_dtr}")  # this is known as over fitting

from sklearn.model_selection import cross_val_score
import numpy as np
scores_lr = cross_val_score(model_lr, housing_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores_lr = np.sqrt(-scores_lr)

scores_dtr = cross_val_score(model_dtr, housing_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores_dtr = np.sqrt(-scores_dtr)

scores_rfr = cross_val_score(model_rfr, housing_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores_rfr = np.sqrt(-scores_rfr)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("RMS: ", np.sqrt((scores**2).mean()))
    print("Standard deviation: ", scores.std())
print_scores(rmse_scores_lr)
print_scores(rmse_scores_dtr)
print_scores(rmse_scores_rfr) # chosing this model because of low mean and std


# # Creating a suitable model

from joblib import dump, load
dump(model_rfr, "house_predictor.joblib")


# # Testing the test data

test_data = strat_testset.drop("MEDV", axis = 1)
actual_test_output = strat_testset["MEDV"].copy()
prepared_test_data = my_pipeline.transform(test_data )

from sklearn.metrics import mean_squared_error
housing_predictions = model_rfr.predict(housing_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

our_test_output = model_rfr.predict(prepared_test_data)
final_mse = mean_squared_error(actual_test_output, our_test_output)
final_rmse = np.sqrt(final_mse)
print(f"final rmse: {final_rmse}") # final rmse is less than RMS:3.3632

a = len(our_test_output)
print(a)
j =1;
for i in our_test_output:
    print(f"{j}:{i}", end= ', ')
    j = j+1
print("\n")

a = len(actual_test_output)
print(a)
j =1;
for i in actual_test_output:
    print(f"{j}:{i}", end= ', ')
    j = j+1


# # Using the model for unknown data and predicting the price

from joblib import dump, load
model = load("house_predictor.joblib")
features = np.array([[0.02187,  60.00,   2.930, 0,  0.4010,  66.8000,   34.90 , 6.2196,   1,  265.0,  15.60, 393.37,   5.03]])
print("\n")
print(model.predict(features))

