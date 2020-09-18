# %%
# dataframe and plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
# scaling and dataset split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# OLS, Ridge
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# model evaluation
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%
df_BHA = df[(df['Race/Ethnicity_Asian'] == 0) & (df['Race/Ethnicity_Other'] == 0)]
# %%
cols = [
    # 'Disabling Condition: Drug or Alcohol',
    #    'Disabling Condition: Emotionally Disturbed',
    #    'Disabling Condition: Learning Disability',
    #    'Disabling Condition: Medically Involved', 'Disabling Condition: Other',
    #    'Disabling Condition: Physical', 'Children in Adoption Placements',
    #    'Children Waiting on Adoption as of 8/31',
    #    'Total Months Since Termination',
    #    'Average Months Since Termination of Parental Rights',
    #    'Removals',
    #    'Victims',
    #    'In Substitute Care with Non-Relative',
    #    'In Substitute Care with Relative', 
    #    'Fiscal Year_2010',
    #    'Fiscal Year_2011', 'Fiscal Year_2012', 'Fiscal Year_2013',
    #    'Fiscal Year_2014', 'Fiscal Year_2015', 'Fiscal Year_2016',
    #    'Fiscal Year_2017', 'Fiscal Year_2018', 'Fiscal Year_2019',
    #    'Region_1-Lubbock', 'Region_10-El Paso', 'Region_11-Edinburg',
    #    'Region_2-Abilene', 'Region_3-Arlington', 'Region_4-Tyler',
    #    'Region_5-Beaumont', 'Region_6-Houston', 'Region_7-Austin',
    #    'Region_8-San Antonio', 'Region_9-Midland',
       'Race/Ethnicity_African American', 'Race/Ethnicity_Anglo',
       'Race/Ethnicity_Asian',
        'Race/Ethnicity_Hispanic',
       'Race/Ethnicity_Other', 
        'Age Group_Birth to Five Years Old',
       'Age Group_Six to Twelve Years Old',
       'Age Group_Thirteen to Seventeen Years Old', 'Gender_Female',
       'Gender_Male' ]
X = df[cols].values
y = df['Percent of Kids in Consummated Adoptions'].values
# %%
len(df[cols].values)
len(df['Percent of Kids in Consummated Adoptions'].values)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %%
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)
plt.scatter(y_test, y_pred_linear, color='gray')
plt.scatter
plt.show()
linear_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_linear))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_linear))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_linear))  
print('Root Mean Squared Error:', linear_rmse)
# %%
rf = RandomForestRegressor(n_estimators = 400, max_depth=5, random_state = 1, oob_score=True)
rf.fit(X_train, y_train)
rf.oob_score_
rf.feature_importances_
# Plot the feature importance
feat_scores = pd.Series(rf.feature_importances_,
                           index=cols)
feat_scores = feat_scores.sort_values()
ax = feat_scores.plot(kind='barh', 
                      figsize=(10,8),
                      color='b')
ax.set_title('Average Gini Importance')
ax.set_xlabel('Average contribution to information gain');
# Use the forest's predict method on the test data
y_pred_rf = rf.predict(X_test)
rf_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
print('Root Mean Squared Error:', rf_rmse)
# %%
# # GridSearchCV

# hyperparameters = {'max_depth': [2, 3, 4, 5],  
#                    'max_features': ['sqrt', 'log2', None], 
#                    'random_state': [0, 1]}
# gridsearch_rf = GridSearchCV(rf, hyperparameters, verbose=True, scoring='neg_mean_squared_error')
# gridsearch_rf.fit(X_train, y_train)

# print("best parameters:", gridsearch_rf.best_params_)
# best_rf_model = gridsearch_rf.best_estimator_
# %%
xgb_reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1)
xgb_reg.fit(X_train, y_train)
# Plot the feature importance
feat_scores = pd.Series(xgb_reg.feature_importances_,
                           index=cols)
feat_scores = feat_scores.sort_values()
ax = feat_scores.plot(kind='barh', 
                      figsize=(10,8),
                      color='b')
ax.set_title('Feature Importances')
ax.set_xlabel('Average contribution to information gain');
y_pred_xgb = xgb_reg.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print('Root Mean Squared Error:', xgb_rmse)
# %%
# # GridSearchCV

# param_grid = {
#     'n_estimators': [1000],
#     # 'colsample_bytree': [0.7, 0.8],
#     'max_depth': [15,20,25],
#     # 'reg_alpha': [1.1, 1.2, 1.3],
#     # 'reg_lambda': [1.1, 1.2, 1.3],
#     # 'subsample': [0.7, 0.8, 0.9],
#     # 'learning_rate': [0.1, 0.5, 0.9]
# }

# gridsearch_xgb = GridSearchCV(xgb_reg, param_grid, verbose=True, scoring='neg_mean_squared_error')
# gridsearch_xgb.fit(X_train, y_train)

# print("best parameters:", gridsearch_xgb.best_params_)
# best_xgb_model = gridsearch_xgb.best_estimator_
# %%
