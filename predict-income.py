# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
''' Removes Irrelevant columns from Dataset '''
def dropIrrelevantColumns(data) :
    data = data.drop('Wears Glasses', axis = 1)
    data = data.drop('Hair Color', axis = 1)
    data = data.drop('Instance', axis = 1)
    return data

''' Preprocesses Data as commented '''
def preprocessData(data, data_test) :
    #Split Data into Independent and Dependent Variable
    X = pd.DataFrame(data.iloc[:, :-1])
    X_test = pd.DataFrame(data_test.iloc[:, :-1])
    Y = pd.Series(data['Total Yearly Income [EUR]'])
    '''
    Z = pd.DataFrame(X['Yearly Income in addition to Salary (e.g. Rental Income)'])
    Z_test = pd.DataFrame(X_test['Yearly Income in addition to Salary (e.g. Rental Income)'])
    
    Z['Yearly Income in addition to Salary (e.g. Rental Income)'] = Z.apply(
            lambda row: float(row['Yearly Income in addition to Salary (e.g. Rental Income)'].split()[0]), axis=1
            )
    Z_test['Yearly Income in addition to Salary (e.g. Rental Income)'] = Z_test.apply(
            lambda row: float(row['Yearly Income in addition to Salary (e.g. Rental Income)'].split()[0]), axis=1
            )
    Z = pd.Series(Z['Yearly Income in addition to Salary (e.g. Rental Income)'])
    Z_test = pd.Series(Z_test['Yearly Income in addition to Salary (e.g. Rental Income)'])
    Y = Y.subtract(Z)
    del Z
    X = X.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis=1)
    X_test = X_test.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis=1)
    '''
    X['train'] = 1
    X_test['train'] = 0
    cmb = pd.concat([X, X_test])
    del X
    del X_test
    cmb['Yearly Income in addition to Salary (e.g. Rental Income)'] = cmb.apply(
            lambda row: float(row['Yearly Income in addition to Salary (e.g. Rental Income)'].split()[0]), axis=1
            )
    cmb['Work Experience in Current Job [years]'] = cmb['Work Experience in Current Job [years]'].replace('#NUM!', np.nan)   
    
    cmb['Gender'] = cmb['Gender'].fillna('unknown')
    #cmb['Gender'] = cmb['Gender'].replace('0', 'unknown')
    #cmb['Gender'] = cmb['Gender'].replace('f', 'female')
    cmb['University Degree'] = cmb['University Degree'].fillna('unknown')
    #cmb['University Degree'] = cmb['University Degree'].replace('0', 'zero')
    cmb['Profession'].fillna('unknown', inplace=True)
    cmb['Country'].fillna('unknown', inplace=True)
    cmb['Age'].fillna(cmb['Age'].median(), inplace=True)
    cmb['Year of Record'].fillna(cmb['Year of Record'].median(), inplace=True)
    cmb['Body Height [cm]'].fillna(cmb['Body Height [cm]'].mean(), inplace=True)
    cmb['Work Experience in Current Job [years]'] = pd.to_numeric(cmb['Work Experience in Current Job [years]'])
    cmb['Work Experience in Current Job [years]'].fillna(cmb['Work Experience in Current Job [years]'].median(), inplace=True)
    cmb['Satisfation with employer'].fillna('unknown', inplace=True)
    #cmb['Hair Color'].fillna('unknown', inplace=True)
    cmb['Housing Situation'].replace(0, 'zero')
    cat_Col_names = ['Gender', 'Country', 'Profession', 'University Degree', 'Housing Situation', 'Satisfation with employer']
    #for y in cat_Col_names:
    #   cmb[y] = LabelEncoder().fit_transform(cmb[y].astype(str))
    #te = ce.TargetEncoder()
    #cmb = te.fit_transform(cmb)
    X = cmb[cmb['train'] == 1]
    X_test = cmb[cmb['train'] == 0]
    del cmb
    X = X.drop('train', axis=1)
    X_test = X_test.drop('train', axis=1)
    #X = te.fit_transform(X, Y, verbose = 1000)
    #X_test = te.transform(X_test)
    
    return (X,Y, X_test)

data = pd.read_csv('data.csv')
data_test = pd.read_csv('data_test.csv')
data = dropIrrelevantColumns(data)
data_test = dropIrrelevantColumns(data_test)
X , Y , X_test = preprocessData(data, data_test)
cat_col = [1,4,5,7,9,10]
cat_Col_names = ['Gender', 'Country', 'Profession', 'University Degree', 'Housing Situation', 'Satisfation with employer']

te = ce.TargetEncoder(verbose=2, cols = cat_Col_names)
X = te.fit_transform(X, Y)
X_test = te.transform(X_test)
print(str(X.columns))

x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=0.2)

params = {
          'max_depth' : 20,
          'learning_rate' : 0.0015,
          "boosting" : "gbdt",
          "verbosity" : 2,
          "num_leaves" : 200,
          "n_jobs" : 12
         }

train_data = lgb.Dataset(X , label = Y)
test_data = lgb.Dataset(x_val, label = y_val)
l = lgb.train(params, train_data, 150000, verbose_eval=1000)
Y_pred = l.predict(X_test)
'''
#dMatrix = xgb.DMatrix(data=X,label=Y)
regressor = RandomForestRegressor(n_estimators=300, verbose=100, n_jobs = -1)
#regressor = xgb.XGBRFRegressor(max_depth=1500, learning_rate=0.3, n_estimators=500, verbosity=3, objective='reg:squarederror', n_jobs=-1, gpu_id=0)
regressor.fit(X, Y)


regressor = CatBoostRegressor(iterations = 100000, cat_features=cat_col, verbose=True)
model = regressor.fit(X, Y, cat_features=cat_col, verbose=True)
Y_pred = model.predict(X_test)

regressor = RandomForestRegressor(n_estimators=450, verbose=100, n_jobs = -1)
regressor.fit(X, Y)
Y_pred = regressor.predict(X_test)
'''


del X
del data
del data_test
del Y
del X_test

Y_pred = np.array(Y_pred)
#Z_test = np.array(Z_test)
#Y_pred = np.add(Y_pred, Z_test)
with open("ypredLastL2Try.csv", "w") as file:
    for i in np.array(Y_pred) :
        file.write(str(i) + "\n")
        
