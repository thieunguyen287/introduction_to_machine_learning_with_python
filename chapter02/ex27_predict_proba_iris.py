import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

iris = load_iris()
named_target = iris.target_names[iris.target]
X_train, X_test, y_train, y_test, named_y_train, named_y_test = train_test_split(iris.data, iris.target, named_target)
log_reg = LogisticRegression().fit(X_train, named_y_train)
print 'classes:', log_reg.classes_
predictions = log_reg.predict(X_test)
parsed_df_predictions = log_reg.classes_[np.argmax(log_reg.decision_function(X_test), axis=1)]
for prediction, parsed_df_prediction in zip(predictions, parsed_df_predictions):
    print prediction, parsed_df_prediction

