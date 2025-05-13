import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
import numpy as np

#https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption
df = pd.read_csv("WorldEnergyConsumption.csv")
df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')


reviewed = df
reviewed.dropna(inplace=True)
reviewed.reset_index(drop=True, inplace=True)


#Normalize
reviewed.iloc[:, 3:] -= np.average(reviewed.iloc[:, 3:], axis=0)
reviewed.iloc[:, 3:] /= np.std(reviewed.iloc[:, 3:], axis=0)
print(reviewed.info())


X = reviewed.iloc[:, 3:-1].to_numpy()
y = reviewed.iloc[:, -1].to_numpy()

print(f"Absolute largest feature value: {np.max(np.abs(X))}")
print(f"Absolute largest label value: {np.max(np.abs(y))}")

test_size = 0.85

#For HW3#

# for i in range(20):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#     reg = LinearRegression().fit(X_train, y_train)
#     arg_max = np.argmax(np.abs(reg.coef_))
#     rmse = np.average((reg.predict(X_test)-y_test)**2.0)**0.5
#     print(f"Linear: R^2 = {reg.score(X_test, y_test):.3f}, RMSE = {rmse:.3f}, largest weight occurred at feature {arg_max} ({df.columns[arg_max+2]}) with value {reg.coef_[arg_max]}")
#     reg = Ridge(alpha=1.0).fit(X_train, y_train)
#     arg_max = np.argmax(np.abs(reg.coef_))
#     rmse = np.average((reg.predict(X_test) - y_test) ** 2.0) ** 0.5
#     print(f"Ridge: R^2 = {reg.score(X_test, y_test):.3f}, RMSE = {rmse:.3f}, largest weight occurred at feature {arg_max} ({df.columns[arg_max+2]}) with value {reg.coef_[arg_max]}")
#     print("-" * 120)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
reg = Lasso()
parameters = {"alpha": np.linspace(0.005, 0.2, num=10)}
grid_search = GridSearchCV(reg, param_grid=parameters, cv=5, scoring="r2")
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_alpha', 'mean_test_score', 'rank_test_score']])

alpha = grid_search.best_params_['alpha']
reg = Lasso(alpha=alpha)
cv_results = cross_validate(reg, X_train, y_train, cv=5, scoring = "r2")
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))