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


# %%
len(df[df['Percent of Kids in Consummated Adoptions']==0])
# %%
df_BHA = df[(df['Race/Ethnicity_Asian'] == 0) & (df['Race/Ethnicity_Other'] == 0)]
# %%
df_BHA.info()
# %%
df_BHA.columns
# %%

cols = ['Disabling Condition: Drug or Alcohol',
       'Disabling Condition: Emotionally Disturbed',
       'Disabling Condition: Learning Disability',
       'Disabling Condition: Medically Involved', 'Disabling Condition: Other',
       'Disabling Condition: Physical', 'Children in Adoption Placements',
       'Children Waiting on Adoption as of 8/31',
       'Total Months Since Termination',
       'Average Months Since Termination of Parental Rights', 'Removals',
       'Victims',
       'In Substitute Care with Non-Relative',
       'In Substitute Care with Relative', 
       'Fiscal Year_2010',
       'Fiscal Year_2011', 'Fiscal Year_2012', 'Fiscal Year_2013',
       'Fiscal Year_2014', 'Fiscal Year_2015', 'Fiscal Year_2016',
       'Fiscal Year_2017', 'Fiscal Year_2018', 'Fiscal Year_2019',
       'Region_1-Lubbock', 'Region_10-El Paso', 'Region_11-Edinburg',
       'Region_2-Abilene', 'Region_3-Arlington', 'Region_4-Tyler',
       'Region_5-Beaumont', 'Region_6-Houston', 'Region_7-Austin',
       'Region_8-San Antonio', 'Region_9-Midland',
       'Race/Ethnicity_African American', 'Race/Ethnicity_Anglo',
       'Race/Ethnicity_Asian', 'Race/Ethnicity_Hispanic',
       'Race/Ethnicity_Other', 'Age Group_Birth to Five Years Old',
       'Age Group_Six to Twelve Years Old',
       'Age Group_Thirteen to Seventeen Years Old', 'Gender_Female',
       'Gender_Male' ]
X = df_BHA[cols].values
y = df_BHA['Percent of Kids in Consummated Adoptions'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
linear = LinearRegression()
linear.fit(X_train, y_train)

# %%
# coeff_df = pd.DataFrame(linear.coef_, X)  
# coeff_df

# %%
y_pred = linear.predict(X_test)

# %%
plt.scatter(y_test, y_pred, color='gray')
plt.scatter
plt.show()

# %%
actual_vs_pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_vs_pred_df1 = actual_vs_pred_df.head(25)
actual_vs_pred_df1
# %%
linear_mse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# %%
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# %%
# df_above_0.mean()
# %%
# create a random forest regressor

rf = RandomForestRegressor(n_estimators = 400, max_depth=5, random_state = 1, oob_score=True)
rf.fit(X_train, y_train)
rf.oob_score_

# %%
rf.feature_importances_

# %%
# Plot the feature importance
feat_scores = pd.Series(rf.feature_importances_,
                           index=cols)
feat_scores = feat_scores.sort_values()
ax = feat_scores.plot(kind='barh', 
                      figsize=(10,8),
                      color='b')
ax.set_title('Average Gini Importance')
ax.set_xlabel('Average contribution to information gain');
# %%
# Use the forest's predict method on the test data
y_pred = rf.predict(X_test)
# %%
# calculate rmse

np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# %%
GridSearchCV

hyperparameters = {'max_depth': [2, 3, 4, 5],  
                   'max_features': ['sqrt', 'log2', None], 
                   'random_state': [0, 1]}
gridsearch = GridSearchCV(rf, hyperparameters, verbose=True, scoring='neg_mean_squared_error')
gridsearch.fit(X_train, y_train)

print("best parameters:", gridsearch.best_params_)
best_rf_model = gridsearch.best_estimator_
# %%

X = df_BHA[cols].values
y = df_BHA['Percent of Kids in Consummated Adoptions'].values

# scaler = StandardScaler()

# #transform data
# scaled = scaler.fit_transform(X)
# scaled
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %%
# ridge = Ridge(alpha=0, normalize=)
ridge.fit(X_train, y_train)
ridge_df = pd.DataFrame({'variable': df_BHA[cols].columns, 'estimate': ridge.coef_})
ridge_train_pred = []
ridge_test_pred = []

# iterate lambdas

for alpha in np.arange(0.001, 10, .1):
    # training
    ridge_reg = Ridge(alpha=alpha, normalize=True)
    ridge_reg.fit(X_train, y_train)
    var_name = 'estimate' + str(alpha)
    ridge_df[var_name] = ridge_reg.coef_
    # prediction
    ridge_train_pred.append(ridge_reg.predict(X_train))
    ridge_test_pred.append(ridge_reg.predict(X_test))

# organize dataframe
# ridge_df = ridge_df.set_index('variable').T.rename_axis('estimate', 1).rename_axis(None, 1).set_index('variable')
ridge_df = ridge_df.set_index('variable')
# %%
ridge_df.head()
# %%
ridge_test_pred
# %%
ridge_df.iloc[1,:].values
# %%
fig, ax = plt.subplots(figsize=(10, 5))
# counter = 0
for i in range(len(ridge_df)):
    ax.plot(np.arange(0.001,10,.1), ridge_df.iloc[i,1:])
    # counter += 1
    # print(counter)
# %%
# plot betas by lambda
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ridge_df.iloc[0,:], 'r', ridge_df.iloc[1,:], 'g', ridge_df.iloc[2,:], 'b', ridge_df.iloc[3,:], 'c', ridge_df.iloc[4,:], 'y')
ax.axhline(y=0, color='black', linestyle='--')
ax.set_xlabel("Lambda")
ax.set_ylabel("Beta Estimate")
ax.set_title("Ridge Regression Trace", fontsize=16)
ax.legend(labels=['Different Race/Ethnicity as Adoptive Parent','Same Race/Ethnicity as Adoptive Parent','Disabling Condition: Drug or Alcohol','Disabling Condition: Emotionally Disturbed','Disabling Condition: Learning Disability'])
ax.grid(True)
# %%

# MSE of Ridge and OLS
ridge_mse_test = [mean_squared_error(y_test, p) for p in ridge_test_pred]

# plot mse
plt.plot(ridge_mse_test[:25], 'ro')
plt.axhline(y=linear_mse, color='g', linestyle='--')
plt.title("Ridge Test Set MSE", fontsize=16)
plt.xlabel("Model Simplicity$\longrightarrow$")
plt.ylabel("MSE")
# %%

# LASSO REG
lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)

# %%
print("training score:", train_score)
print("test score: ", test_score)
print("number of features used: ", coeff_used)
# %%
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)
train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)

# %%
print("training score for alpha=0.01:", train_score001)
print("test score for alpha =0.01: ", test_score001)
print("number of features used: for alpha =0.01:", coeff_used001)
# %%
lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
# %%
print("training score for alpha=0.0001:", train_score00001)
print("test score for alpha =0.0001: ", test_score00001)
print("number of features used: for alpha =0.0001:", coeff_used00001)
# %%
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
# %%
print "LR training score:", lr_train_score 
print "LR test score: ", lr_test_score
# %%
plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()
# %%
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)
y_pred_linear = pipeline.predict(X_test)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_linear))))

# %%
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)
y_pred_ridge = ridge_pipe.predict(X_train)

print('Training Score: {}'.format(ridge_pipe.score(X_train, y_train)))
print('Test Score: {}'.format(ridge_pipe.score(X_test, y_test)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))))
# %%

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.3, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)

lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test, y_test)))
# %%
xgb = xgb.XGBRegressor(n_estimators=1000, reg_lambda=1, gamma=0, max_depth=3, learning_rate=0.1)
xgb.fit(X_train, y_train)
# %%
pd.DataFrame(xgb.feature_importances_.reshape(1, -1), columns=cols)
# %%
y_pred_xgb = xgb.predict(X_test)
# %%
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_rmse
# %%
# def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
#                        model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
#                        do_probabilities = False):
#     gs = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid, 
#         cv=cv, 
#         n_jobs=-1, 
#         scoring=scoring_fit,
#         verbose=2
#     )
#     fitted_model = gs.fit(X_train_data, y_train_data)
    
#     # if do_probabilities:
#     #   pred = fitted_model.predict_proba(X_test_data)
#     # else:
#     pred = fitted_model.predict(X_test_data)
    
#     return fitted_model, pred
# %%

# %%
model = xgb
param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9]
}

model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, param_grid, cv=5, scoring_fit='neg_mean_squared_error')

print(model.best_score_)
print(model.best_params_)

# hyperparameters = {'max_depth': [2, 3, 4, 5],  
#                    'max_features': ['sqrt', 'log2', None], 
#                    'random_state': [0, 1]}
# gridsearch = GridSearchCV(rf, hyperparameters, verbose=True, scoring='neg_mean_squared_error')
# gridsearch.fit(X_train, y_train)

# print("best parameters:", gridsearch.best_params_)
# best_rf_model = gridsearch.best_estimator_
# %%
