import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [95, 90, 40], 
    [90, 98, 50],
    [40, 50, 95],
    [88, 85, 45],  
    [30, 40, 90],
])

y = np.array([0, 1, 2, 0, 2])

model = LogisticRegression()
model.fit(X, y)

student = np.array([[92, 88, 40]])
prediction = model.predict(student)

majors = {0: "Engineering", 1: "Medicine", 2: "Arts"}
print("--- University Major AI Predictor ---")
print(f"Prediction for the new student is: {majors[prediction[0]]}")

print("--- University Major AI Predictor ---")
print(f"Prediction result: {majors[prediction[0]]}")

majors = {0: "Engineering", 1: "Medicine", 2: "Arts"}
print("--- University Major AI Predictor ---")
print(f"Prediction result: {majors[prediction[0]]}")