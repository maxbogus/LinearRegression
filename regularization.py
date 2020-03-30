import pandas
import numpy
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
train_data = pandas.read_csv('data.csv', header = None)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

lasso_reg = Lasso()

lasso_reg.fit(X, y)

reg_coef = lasso_reg.coef_
print(reg_coef)