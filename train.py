import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['hours_study', 'attendance', 'previous_marks']]
y = data['final_marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

best_model = None
best_score = 0

# Train and compare
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = model

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))

print("\nBest model saved!")