import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load the model
print("Loading model...")
with open('models/predicmodel.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
print(f"Feature names: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Unknown'}")
print(f"Feature importance: {model.feature_importances_ if hasattr(model, 'feature_importances_') else 'Unknown'}")

# Define parameter ranges for testing
progress_values = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.1, ..., 1.0
hours_values = np.linspace(0, 50, 11).astype(int)  # 0, 5, 10, ..., 50
priority_values = [0, 1, 2]  # Low, Medium, High
project_types = ['Renovation', 'Construction', 'Infrastructure', 'Maintenance', 'Innovation', 'Other']

# Create a grid of test cases
results = []

# Test progress vs hours with fixed priority and project type
print("\nTesting progress vs hours (with High priority and Renovation project type)...")
for progress in progress_values:
    for hours in hours_values:
        # Renovation project with High priority
        features = np.array([[progress, hours, 2, 1, 0, 0, 0, 0]])
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None
        
        result = {
            'Progress': progress,
            'Hours': hours,
            'Priority': 'High',
            'Project Type': 'Renovation',
            'Prediction': 'Delayed' if prediction[0] == 1 else 'On Track',
            'Probability': probability
        }
        results.append(result)

# Test priority with fixed progress, hours, and project type
print("\nTesting priority (with progress=0.3, hours=30, and Renovation project type)...")
for priority in priority_values:
    priority_name = ['Low', 'Medium', 'High'][priority]
    # Renovation project with fixed progress and hours
    features = np.array([[0.3, 30, priority, 1, 0, 0, 0, 0]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None
    
    result = {
        'Progress': 0.3,
        'Hours': 30,
        'Priority': priority_name,
        'Project Type': 'Renovation',
        'Prediction': 'Delayed' if prediction[0] == 1 else 'On Track',
        'Probability': probability
    }
    results.append(result)

# Test project types with fixed progress, hours, and priority
print("\nTesting project types (with progress=0.3, hours=30, and High priority)...")
for i, project_type in enumerate(project_types):
    # Create one-hot encoding for project type
    project_type_encoding = [0, 0, 0, 0, 0]
    if i < 5:  # Skip 'Other' which is all zeros
        project_type_encoding[i] = 1
    
    features = np.array([[0.3, 30, 2] + project_type_encoding])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None
    
    result = {
        'Progress': 0.3,
        'Hours': 30,
        'Priority': 'High',
        'Project Type': project_type,
        'Prediction': 'Delayed' if prediction[0] == 1 else 'On Track',
        'Probability': probability
    }
    results.append(result)

# Convert results to DataFrame for analysis
df_results = pd.DataFrame(results)

# Print summary of results
print("\nSummary of test results:")
print(f"Total test cases: {len(df_results)}")
print(f"Delayed predictions: {(df_results['Prediction'] == 'Delayed').sum()}")
print(f"On Track predictions: {(df_results['Prediction'] == 'On Track').sum()}")

# Find cases with highest probability of delay
if 'Probability' in df_results.columns and df_results['Probability'].notna().any():
    print("\nTop 10 cases with highest probability of delay:")
    top_delay = df_results.sort_values('Probability', ascending=False).head(10)
    print(top_delay[['Progress', 'Hours', 'Priority', 'Project Type', 'Prediction', 'Probability']])

# Find the decision boundary for progress
print("\nFinding decision boundary for progress (with hours=30, High priority, Renovation)...")
progress_boundary = None
for progress in np.linspace(0, 1, 101):  # More fine-grained search
    features = np.array([[progress, 30, 2, 1, 0, 0, 0, 0]])
    prediction = model.predict(features)
    result = 'Delayed' if prediction[0] == 1 else 'On Track'
    
    if result == 'Delayed':
        progress_boundary = progress
        print(f"Found decision boundary at progress = {progress:.2f}")
        break

if progress_boundary is None:
    print("No decision boundary found for progress - all predictions are 'On Track'")

# Find the decision boundary for hours
print("\nFinding decision boundary for hours (with progress=0.3, High priority, Renovation)...")
hours_boundary = None
for hours in np.linspace(0, 100, 101):  # More fine-grained search
    features = np.array([[0.3, hours, 2, 1, 0, 0, 0, 0]])
    prediction = model.predict(features)
    result = 'Delayed' if prediction[0] == 1 else 'On Track'
    
    if result == 'Delayed':
        hours_boundary = hours
        print(f"Found decision boundary at hours = {hours:.2f}")
        break

if hours_boundary is None:
    print("No decision boundary found for hours - all predictions are 'On Track'")

# Test extreme cases
print("\nTesting extreme cases...")
extreme_cases = [
    # Very low progress, very high hours
    [0.01, 100, 2, 1, 0, 0, 0, 0],  # Renovation, High priority
    [0.0, 100, 2, 1, 0, 0, 0, 0],   # Renovation, High priority, zero progress
    [0.0, 200, 2, 1, 0, 0, 0, 0],   # Renovation, High priority, zero progress, very high hours
    [0.0, 500, 2, 1, 0, 0, 0, 0],   # Renovation, High priority, zero progress, extremely high hours
]

for case in extreme_cases:
    features = np.array([case])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None
    result = 'Delayed' if prediction[0] == 1 else 'On Track'
    
    print(f"Progress: {case[0]:.2f}, Hours: {case[1]:.0f}, Priority: High, Type: Renovation -> {result}")
    if probability is not None:
        print(f"  Probability of delay: {probability:.4f}")

print("\nAnalysis complete!")
