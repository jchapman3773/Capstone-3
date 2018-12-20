from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats import outliers_influence
import statsmodels.api as sm
from joblib import dump
from math import sqrt
np.random.seed(42)

engine = create_engine('postgresql://banana:forscale@bananaforscale.ckaldwfguyw5.us-east-2.rds.amazonaws.com:5432/bananaforscale')

r_df = pd.read_sql_table('regression',con=engine)
h_df = pd.read_sql_table('heights',con=engine)

df = r_df.merge(h_df,left_on='filename',right_on='image',how='left')
df['banana_box'] = df[['banana_box_point1','banana_box_point2','banana_box_point3','banana_box_point4']].mean(axis=1)
df['person_box'] = df[['person_box_point2','person_box_point2','person_box_point3','person_box_point4']].mean(axis=1)
y = df.pop('height_inch')
print(f'Mean Height: {y.mean()//12}ft {round(y.mean()%12,1)}in')

def data(remove_cols):
    X = df.drop(columns=remove_cols)

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    dump(scaler,'models/scaler.joblib')

    X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X

def linear():
    print('---------- LINEAR ELASTIC NET MODEL ----------')
    remove_cols = ['filename','image','index_x','index_y','image_x','image_y',
                    'banana_box_point1','banana_box_point2','banana_box_point3','banana_box_point4',
                    'person_box_point1','person_box_point2','person_box_point3','person_box_point4']
    X_train, X_test, y_train, y_test, X = data(remove_cols)
    model = ElasticNetCV()
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    # [print(x,Y) for x, Y in zip(prediction,y_test)]

    print(f'R^2: {model.score(X_test,y_test)}')
    print(f'RMSE: {sqrt(mean_squared_error(y_test,prediction))}')
    print(f'MAE: {mean_absolute_error(y_test,prediction)}')

    for idx, col in enumerate(X.columns):
        print(f"{col}, {outliers_influence.variance_inflation_factor(X.values,idx)}")

    linear_model = sm.OLS(y_train, X_train).fit()
    print(linear_model.summary2())

    dump(model,'models/linearelasticnet.joblib')

def forest():
    print('---------- RANDOM FOREST MODEL ----------')
    remove_cols = ['filename','image','index_x','index_y']
    X_train, X_test, y_train, y_test, X = data(remove_cols)

    regr = RandomForestRegressor(random_state=42)

    # GRID/RANDOM SEARCH -----------------------------------------------------
    # max_depth = [int(x) for x in np.linspace(10,110, num=11)]
    # max_depth.append(None)
    # random_grid = {'n_estimators': [int(x) for x in np.linspace(start=200,stop=2000,num=10)],
    #                'max_features': ['auto','sqrt'],
    #                'max_depth': max_depth,
    #                # 'min_samples_split': [2,5,10],
    #                # 'min_samples_leaf': [1,2,4],
    #                'bootstrap': [True,False]}
    #
    # rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid,
    #                                 n_iter = 500, cv = 3, verbose=1,
    #                                 random_state=42, n_jobs = -1)
    # rf_random.fit(X_train,y_train)
    # pprint(rf_random.best_params_)
    #
    # def evaluate(model, test_features, test_labels):
    #     predictions = model.predict(test_features)
    #     errors = sqrt(mean_squared_error(test_labels,predictions))
    #     mape = 100 * np.mean(errors / test_labels)
    #     accuracy = 100 - mape
    #     print('Model Performance')
    #     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #     print('Accuracy = {:0.2f}%.'.format(accuracy))
    #
    #     return accuracy
    #
    # base_model = RandomForestRegressor(random_state=42)
    # base_model.fit(X_train,y_train)
    # base_accuracy = evaluate(base_model, X_test, y_test)
    #
    # best_random = rf_random.best_estimator_
    # random_accuracy = evaluate(best_random, X_test, y_test)
    #
    # dump(regr,'models/randomforest_bestsearch.joblib')
    # print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

    regr.fit(X_train,y_train)
    print(f'Feature Importances: {regr.feature_importances_}')
    prediction = regr.predict(X_test)
    [print(x,Y) for x, Y in zip(prediction,y_test)]

    print(f'R^2: {regr.score(X_test,y_test)}')
    print(f'RMSE: {sqrt(mean_squared_error(y_test,prediction))}')
    print(f'MAE: {mean_absolute_error(y_test,prediction)}')

    dump(regr,'models/randomforest.joblib')
    # print(X.info())

def boost():
    print('---------- GRADIENT BOOST MODEL ----------')
    remove_cols = ['filename','image','index_x','index_y']
    X_train, X_test, y_train, y_test, X = data(remove_cols)
    regr = GradientBoostingRegressor()
    regr.fit(X_train,y_train)
    print(f'Feature Importances: {regr.feature_importances_}')
    prediction = regr.predict(X_test)
    # [print(x,Y) for x, Y in zip(prediction,y_test)]

    print(f'R^2: {regr.score(X_test,y_test)}')
    print(f'RMSE: {sqrt(mean_squared_error(y_test,prediction))}')
    print(f'MAE: {mean_absolute_error(y_test,prediction)}')

    dump(regr,'models/gradientboost.joblib')
    print(X.info())

if __name__ == '__main__':
    linear()
    forest()
    boost()
