from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats import outliers_influence
import statsmodels.api as sm

engine = create_engine('postgresql://banana:forscale@bananaforscale.ckaldwfguyw5.us-east-2.rds.amazonaws.com:5432/bananaforscale')

r_df = pd.read_sql_table('regression',con=engine)
h_df = pd.read_sql_table('heights',con=engine)

df = r_df.merge(h_df,left_on='filename',right_on='image',how='left')
df['banana_box'] = df[['banana_box_point1','banana_box_point2','banana_box_point3','banana_box_point4']].mean(axis=1)
df['person_box'] = df[['person_box_point2','person_box_point2','person_box_point3','person_box_point4']].mean(axis=1)
y = df.pop('height_inch')

def data(remove_cols):
    X = df.drop(columns=remove_cols)

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X

def linear():
    remove_cols = ['filename','image','index_x','index_y',
                    'banana_box_point1','banana_box_point2','banana_box_point3','banana_box_point4',
                    'person_box_point1','person_box_point2','person_box_point3','person_box_point4']
    X_train, X_test, y_train, y_test, X = data(remove_cols)
    model = ElasticNetCV()
    model.fit(X_train,y_train)
    [print(x,Y) for x, Y in zip(model.predict(X_test),y_test)]

    print(model.score(X_test,y_test))

    for idx, col in enumerate(X.columns):
        print(f"{col}, {outliers_influence.variance_inflation_factor(X.values,idx)}")

    linear_model = sm.OLS(y_train, X_train).fit()
    print(linear_model.summary2())

def forest():
    remove_cols = ['filename','image']
    X_train, X_test, y_train, y_test, _ = data(remove_cols)
    regr = RandomForestRegressor(max_depth=2, random_state=42, n_estimators=100)
    regr.fit(X_train,y_train)
    print(regr.feature_importances_)
    print(regr.score(X_test,y_test))
    [print(x,Y) for x, Y in zip(regr.predict(X_test),y_test)]

if __name__ == '__main__':
    linear()
    forest()
