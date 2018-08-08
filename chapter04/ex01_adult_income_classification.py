from prettytable import PrettyTable
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def display(df):
    table = PrettyTable(field_names=list(df.columns))
    for row in df.values:
        table.add_row(row)
    print table


data = pd.read_csv('../data/adult.data', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race',
                          'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                          'native-country', 'income'])
chosen_fields = ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']
data = data[chosen_fields]
display(data.head())
print data.gender.value_counts()
print data.workclass.value_counts()
print "Original feature:\n", list(data.columns)
data_dummies = pd.get_dummies(data)
print "Dummy features:\n", list(data_dummies.columns)
display(data_dummies.head())
features = data_dummies.ix[:, : 'occupation_ Transport-moving']
display(features.head())
X = features.values
y = data_dummies['income_ >50K'].values
print "X.shape: {}, y.shape: {}".format(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape
model = LogisticRegression().fit(X_train, y_train)
print "Test score:", model.score(X_test, y_test)
