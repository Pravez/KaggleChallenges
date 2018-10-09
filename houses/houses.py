import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
import numpy as np
import scipy.stats as spstats
import seaborn as sns
import matplotlib.pyplot as plt

fields_to_dummies = ['Street', 'Alley', 'LandSlope', 'CentralAir', 'PavedDrive', 'BsmtExposure']

def to_dummy(data, field):
    dummies = pd.get_dummies(data[field])
    dummies.columns = ['%s-%s' % (field, x) for x in dummies.columns]
    data.drop([field], inplace=True, axis=1)
    print('Replacing %s with %s' % (field, ', '.join(dummies.columns)))
    data = data.join(dummies)
    return data

def to_integer(data, field):
    data[field] = data[field].astype('category')
    data[field] = data[field].cat.codes
    return data

def update_categorical_data(data, fields):
    dummied = []
    categoricalized = []
    for field in fields:
        if field in fields_to_dummies:
            data = to_dummy(data, field)
            dummied.append(field)
        else:
            data = to_integer(data, field)
            categoricalized.append(field)
    return data, dummied, categoricalized

def handle_nan(data, fields):
    for field in fields:
        if field in data.columns:
            data[field].fillna(0.0, inplace=True)
    return data

def convert_types(data):
    numeric_cols = [x for x in data.columns if ('Area' in x) | ('SF' in x)] + ['SalePrice', 'LotFrontage',
                                                                                   'MiscVal', 'EnclosedPorch',
                                                                                   'ScreenPorch',
                                                                                   'OverallQual', 'OverallCond',
                                                                                   'YearBuilt']

    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].astype(float)

    categorical_cols = [x for x in data.columns if x not in numeric_cols]

    for col in categorical_cols:
        data[col] = data[col].astype('category')

    return data, numeric_cols, categorical_cols

def correct_nulls(data, nan_cols, numerical_cols, categorical_cols):
    dummied = []
    categoricalized = []
    for col in nan_cols:
        if col in categorical_cols:
            data, d, c = update_categorical_data(data, [col])
            dummied.extend(d)
            categoricalized.extend(c)
        elif col in numerical_cols:
            data = handle_nan(data, [col])
    return data, dummied, categoricalized

def get_null_columns(data):
    nan = data.isnull().sum()
    return [data.columns[x] for x in range(len(nan)) if nan[x] > 0]

def apply_categorical_transform(data, dummied, categoricalized):
    for field in dummied:
        data = to_dummy(data, field)

    for field in categoricalized:
        data[field] = data[field].cat.codes

    return data

def normalize(data, field, inplace=False):
    if data[field].where(data[field] <= 0).count() > 0:
        if not inplace:
            data[field + "_log"] = np.log((1+data[field]))
        else:
            data[field] = np.log((1+data[field]))
    else:
        l, opt_lambda = spstats.boxcox(data[field])
        if not inplace:
            data[field + "_boxcox"] = spstats.boxcox(data[field], lmbda=opt_lambda)
        else:
            data[field] = spstats.boxcox(data[field], lmbda=opt_lambda)
    return data

def apply_other_modifications(data):
    to_normalize = ['TotalBsmtSF', 'GrLivArea', 'WoodDeckSF', 'BsmtUnfSF']

    for field in to_normalize:
        data = normalize(data, field, inplace=True)

    # sns.distplot(data['BsmtUnfSF_log'], kde=False)
    # plt.show()
    #
    # sns.distplot(data['BsmtUnfSF'], kde=False)
    # plt.show()
    #
    # sns.distplot(data['GrLivArea_boxcox'], kde=False)
    # plt.show()
    #
    # sns.distplot(data['GrLivArea'], kde=False)
    # plt.show()

    return data

def do_process(data, test=False, dum_n_cat=None):
    n_data, numeric_cols, categorical_cols = convert_types(data)

    if not test:
        nan_cols = get_null_columns(n_data)
        n_data, dum, cat = correct_nulls(n_data, nan_cols, numeric_cols, categorical_cols)
        n_data, d, c =  update_categorical_data(n_data, [x for x in categorical_cols if x not in nan_cols])
        dum.extend(d)
        cat.extend(c)
        return n_data, dum, cat
    else:
        n_data = handle_nan(n_data, numeric_cols)
        return apply_categorical_transform(n_data, dum_n_cat[0], dum_n_cat[1])


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def get_model(cv=False):
    if not cv:
        return RandomForestRegressor(n_estimators=288, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)
    else:
        random_forest = RandomForestRegressor()

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 50, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf_random = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid, n_iter=50, cv=3, verbose=2,
                                       random_state=42, n_jobs=4)

        return rf_random.best_estimator_

if __name__ == '__main__':
    path_train = "/home/pbreton/.kaggle/competitions/house-prices-advanced-regression-techniques/train.csv"
    path_test = "/home/pbreton/.kaggle/competitions/house-prices-advanced-regression-techniques/test.csv"

    data = pd.read_csv(path_train)
    data.drop(['Id'], axis=1, inplace=True)
    n_data, dum, cat = do_process(data)
    n_data = apply_other_modifications(n_data)


    x = n_data.drop(['SalePrice'], axis=1)
    y = n_data['SalePrice']

    data_t = pd.read_csv(path_test)
    ids = data_t['Id']
    data_t.drop(['Id'], axis=1, inplace=True)

    n_data_t = do_process(data_t, test=True, dum_n_cat=(dum, cat))
    n_data_t = apply_other_modifications(n_data_t)

    model = get_model(cv=False)

    base_model = RandomForestRegressor(n_estimators=10, random_state=42)
    base_model.fit(x, y)
    base_accuracy = evaluate(base_model, x, y)

    model = model.fit(x, y)
    model_accuracy = evaluate(model, x, y)


    results = model.predict(n_data_t)
    data_res = pd.DataFrame({
        "Id": ids,
        "SalePrice": results
    })

    data_res.to_csv('/home/pbreton/Documents/Workspace-BeTomorrow/workspace/miscellaneous/kaggle/houses/submission.csv', index=False)