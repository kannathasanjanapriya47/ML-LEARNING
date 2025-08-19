from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model + metadata
joblib.dump({
    "model": model,
    "feature_names": feature_names,
    "target_names": target_names
}, "model.pkl")

print("Model trained and saved as model.pkl")
