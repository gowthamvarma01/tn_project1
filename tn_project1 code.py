import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Step 1: Generate Dataset
# -------------------------------
X, y = make_classification(n_samples=200,
                           n_features=5,
                           n_informative=3,
                           n_redundant=0,
                           n_classes=2,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 2: Custom Logistic Regression
# -------------------------------
class CustomLogisticRegression:
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X):
        y_pred_prob = self.predict_proba(X)
        return np.where(y_pred_prob >= 0.5, 1, 0)

# -------------------------------
# Step 3: Train Custom Model
# -------------------------------
custom_model = CustomLogisticRegression(learning_rate=0.1, iterations=2000)
custom_model.fit(X_train, y_train)

y_pred_custom = custom_model.predict(X_test)

# -------------------------------
# Step 4: Train Sklearn Model
# -------------------------------
sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)

y_pred_sk = sk_model.predict(X_test)

# -------------------------------
# Step 5: Evaluation Metrics
# -------------------------------
print("---- Custom Logistic Regression ----")
print("Weights:", custom_model.weights)
print("Bias:", custom_model.bias)
print("Accuracy:", accuracy_score(y_test, y_pred_custom))
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))

print("\n---- Sklearn Logistic Regression ----")
print("Weights:", sk_model.coef_)
print("Bias:", sk_model.intercept_)
print("Accuracy:", accuracy_score(y_test, y_pred_sk))
print("Precision:", precision_score(y_test, y_pred_sk))
print("Recall:", recall_score(y_test, y_pred_sk))
