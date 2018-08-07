import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras_contrib.layers import InstanceNormalization
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random

from sklearn.neighbors import KNeighborsClassifier


def substrings_in_string(big_string, substrings):
    for substring in substrings:
        try:
            if big_string.find(substring) != -1:
                return substring
        except:
            return 'Unknown'
    return np.nan


def create_keras_model(input_size):
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_childs(person):
    age, sex = person
    return "child" if age < 18 else sex


train_file_path = "/home/pbreton/.kaggle/competitions/titanic/train.csv"
test_file_path = "/home/pbreton/.kaggle/competitions/titanic/test.csv"

df = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

print("Before transforming : ")
print(df.columns)

# Adding a column for family size
df['FamilySize'] = df['Parch'] + df['SibSp']
test['FamilySize'] = test['Parch'] + test['SibSp']

df['Fare per person'] = df['Fare'] / (df['FamilySize'] + 1)
test['Fare per person'] = test['Fare'] / (test['FamilySize'] + 1)

mean_ages = np.mean(df['Age'].dropna())
std_ages = np.std(df['Age'].dropna())

mean_ages_test = np.mean(df['Age'].dropna())
std_ages_test = np.std(df['Age'].dropna())

df['Age'].update(df.apply(lambda x: random.randrange(
    int(mean_ages - std_ages), int(mean_ages + std_ages)) if pd.isnull(x['Age']) else x['Age'], axis=1))
test['Age'].update(test.apply(lambda x: random.randrange(
    int(mean_ages_test - std_ages_test), int(mean_ages_test + std_ages_test)) if pd.isnull(x['Age']) else x['Age'], axis=1))

df['Age*Class'] = df['Age'] * df['Pclass']
test['Age*Class'] = test['Age'] * test['Pclass']

#df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)
#test['Sex'] = test['Sex'].map(lambda x: 1 if x == 'male' else 0)

df['Person'] = df[['Age', 'Sex']].apply(create_childs, axis=1)
test['Person'] = test[['Age', 'Sex']].apply(create_childs, axis=1)

df.drop(['Sex'], axis=1, inplace=True)
test.drop(['Sex'], axis=1, inplace=True)

person_dummies = pd.get_dummies(df['Person'])
person_dummies_test = pd.get_dummies(test['Person'])
person_dummies.columns = ["Child", "Female", "Male"]
person_dummies_test.columns = ["Child", "Female", "Male"]
person_dummies.drop(['Male'], axis=1, inplace=True)
person_dummies_test.drop(['Male'], axis=1, inplace=True)

df = df.join(person_dummies)
test = test.join(person_dummies_test)

df.drop(['Person'], axis=1, inplace=True)
test.drop(['Person'], axis=1, inplace=True)

mean_fare = np.mean(df['Fare per person'])
mean_fare_test = np.mean(test['Fare per person'])

df['Fare per person'].fillna(mean_fare)
test['Fare per person'].fillna(mean_fare_test)

test['Fare per person'].update(test.apply(lambda x: 0 if x['FamilySize'] == 0 else x['Fare per person'], axis=1))

df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

embarked_dummies = pd.get_dummies(df['Embarked'])
embarked_dummies_test = pd.get_dummies(test['Embarked'])
embarked_dummies.drop(['S'], inplace=True, axis=1)
embarked_dummies_test.drop(['S'], inplace=True, axis=1)

df = df.join(embarked_dummies)
test = test.join(embarked_dummies_test)

df.drop(['Embarked'], axis= 1 , inplace=True)
test.drop(['Embarked'], axis= 1 , inplace=True)

pclass_dummies = pd.get_dummies(df['Pclass'])
pclass_dummies_test = pd.get_dummies(test['Pclass'])
pclass_dummies.columns = ["Class_1", "Class_2", "Class_3"]
pclass_dummies_test.columns = ["Class_1", "Class_2", "Class_3"]
pclass_dummies.drop(['Class_3'], axis=1, inplace=True)
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

df.drop(['Pclass'], axis=1, inplace=True)
test.drop(['Pclass'], axis=1, inplace=True)

df = df.join(pclass_dummies)
test = test.join(pclass_dummies_test)

test["Fare"].fillna(test["Fare"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

df['Family'] = df.apply(lambda x: 1 if x['Parch'] + x['SibSp'] > 0 else 0, axis=1)
test['Family'] = test.apply(lambda x: 1 if x['Parch'] + x['SibSp'] > 0 else 0, axis=1)

df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

df.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)

print("After transforming : ")
print(df.columns)

X_train = df.drop(['Survived'], axis=1)
Y_train = df['Survived']
X_test = test.drop(['PassengerId', 'Ticket', 'Name'], axis=1).copy()

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)


random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))

# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = pd.DataFrame(df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print(coeff_df)

model = create_keras_model(X_train.shape[1])
Y_pred = model.predict(X_train)

#model.fit(X_train, Y_train, epochs=20, steps_per_epoch=500)

result = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": Y_pred
})

result.to_csv('/home/pbreton/submission_titanic.csv', index=False)