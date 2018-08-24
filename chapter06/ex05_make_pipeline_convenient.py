from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

pipe = make_pipeline(MinMaxScaler(), SVC(C=100))
print pipe.steps
print pipe.named_steps['svc']
