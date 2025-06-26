import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load and prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and iris dataset object
with open("iris_model.pkl", "wb") as f:
    pickle.dump((model, iris), f)
