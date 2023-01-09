from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris = load_iris()

iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
iris_data['target'] = iris_data['target'].map({0: "setosa", 1: "versicolor", 2: "virginica"})

iris_data.to_csv("iris_data.csv")