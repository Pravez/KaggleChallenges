import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

def to_dummy(data, field):
    dummies = pd.get_dummies(data[field])
    dummies.columns = ['%s-%s' % (field, x) for x in dummies.columns]
    data.drop([field], inplace=True, axis=1)
    print('Replacing %s with %s' % (field, ', '.join(dummies.columns)))
    data = data.join(dummies)
    return data

def to_categorical(data, fields):
    dummied = []
    categoricalized = []
    for field in fields:
        #if data[field].isnull().sum() > 0 or len(data[field].cat.categories) < 3:
        data = to_dummy(data, field)
        dummied.append(field)
        #else:
        #    data[field] = data[field].cat.codes
        #    categoricalized.append(field)
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
            data, d, c = to_categorical(data, [col])
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

def do_process(data, test=False, dum_n_cat=None):
    n_data, numeric_cols, categorical_cols = convert_types(data)

    if not test:
        nan_cols = get_null_columns(n_data)
        n_data, dum, cat = correct_nulls(n_data, nan_cols, numeric_cols, categorical_cols)
        n_data, d, c =  to_categorical(n_data, [x for x in categorical_cols if x not in nan_cols])
        dum.extend(d)
        cat.extend(c)
        return n_data, dum, c
    else:
        n_data = handle_nan(n_data, numeric_cols)
        return apply_categorical_transform(n_data, dum_n_cat[0], dum_n_cat[1])


path_train = "/home/pbreton/.kaggle/competitions/house-prices-advanced-regression-techniques/train.csv"
path_test = "/home/pbreton/.kaggle/competitions/house-prices-advanced-regression-techniques/test.csv"

data = pd.read_csv(path_train)
data.drop(['Id'], axis=1, inplace=True)
n_data, dum, cat = do_process(data)


x = n_data.drop(['SalePrice'], axis=1)
y = n_data['SalePrice']

data_t = pd.read_csv(path_test)
ids = data_t['Id']
data_t.drop(['Id'], axis=1, inplace=True)

n_data_t = do_process(data_t, test=True, dum_n_cat=(dum, cat))

random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(x, y)
print(random_forest.score(x, y))

results = random_forest.predict(n_data_t)
results['Id'] = ids

kfold = KFold(4, True, 1)

#for train, test in kfold.split(data):
#    print('train: %s, test: %s' % (data[train], data[test]))