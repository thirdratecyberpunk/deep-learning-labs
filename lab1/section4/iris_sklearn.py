# sklearn provides some very common ML models, but is not good for complex or
# novel architectures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn.datasets as datasets

iris = datasets.load_iris()
# loading images
X = iris.data
# loading labels
T = iris.target
# calculating number of possible classes
C = len(set(list(T)))
# calculating number of features in each datapoint
F = X.shape[1]

estimator = LogisticRegression()
estimator.fit(X,T)
T_pred = estimator.predict(X)
acc = accuracy_score(T, T_pred)
print(acc)
