from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_param=1, second_param=2):
        super(MyTransformer, self).__init__()
        self.first_param = first_param
        self.second_param = second_param

    def fit(self, X, y=None):
        print "Fitting the model right here"
        return self

    def transform(self, X):
        X_transformed = X + 1
        return X_transformed
