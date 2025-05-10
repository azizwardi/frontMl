import pickle
import numpy as np
import pandas as pd

# Load the model
print("Loading model...")
with open('models/predicmodel.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a grid of test cases with varying progress and hours
progress_values = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.1, ..., 1.0
hours_values = np.linspace(0, 100, 21)  # 0, 5, 10, ..., 100

# Test all combinations of progress and hours with fixed priority and project type
results = []
for progress in progress_values:
    for hours in hours_values:
        # Renovation project with High priority
        features = np.array([[progress, hours, 2, 1, 0, 0, 0, 0]])
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None
        
        result = {
            'Progress': progress,
            'Hours': hours,
            'Prediction': 'Delayed' if prediction[0] == 1 else 'On Track',
            'Probability': probability
        }
        results.append(result)

# Convert results to DataFrame
df = pd.DataFrame(results)

# Find the threshold where predictions change from "On Track" to "Delayed"
delayed_cases = df[df['Prediction'] == 'Delayed']
if len(delayed_cases) > 0:
    print("\nFound cases with 'Delayed' prediction:")
    print(delayed_cases[['Progress', 'Hours', 'Probability']].head(10))
else:
    print("\nNo cases with 'Delayed' prediction found.")

# Find cases with highest probability of delay
print("\nTop 10 cases with highest probability of delay:")
top_delay = df.sort_values('Probability', ascending=False).head(10)
print(top_delay[['Progress', 'Hours', 'Probability', 'Prediction']])

# Find the threshold probability for "Delayed" prediction
if len(delayed_cases) > 0:
    min_delayed_prob = delayed_cases['Probability'].min()
    print(f"\nMinimum probability for 'Delayed' prediction: {min_delayed_prob:.4f}")
    
    # Find cases just below the threshold
    threshold_cases = df[(df['Prediction'] == 'On Track') & 
                         (df['Probability'] > min_delayed_prob * 0.9)]
    if len(threshold_cases) > 0:
        print("\nCases just below the threshold:")
        print(threshold_cases.sort_values('Probability', ascending=False).head(5)[['Progress', 'Hours', 'Probability']])

# Print summary statistics
print("\nProbability statistics:")
print(df['Probability'].describe())

# Identify parameter combinations that lead to high delay probabilities
print("\nParameter combinations for high delay probabilities:")

# Effect of progress
progress_effect = df.groupby('Progress')['Probability'].mean().reset_index()
print("\nAverage delay probability by progress:")
print(progress_effect.sort_values('Probability', ascending=False).head(5))

# Effect of hours
hours_effect = df.groupby('Hours')['Probability'].mean().reset_index()
print("\nAverage delay probability by hours:")
print(hours_effect.sort_values('Probability', ascending=False).head(5))

# Find the optimal parameter values for getting a "Delayed" prediction
print("\nOptimal parameter values for getting a 'Delayed' prediction:")
if len(delayed_cases) > 0:
    # Find the most reliable combination
    most_reliable = delayed_cases.sort_values('Probability', ascending=False).iloc[0]
    print(f"Progress: {most_reliable['Progress']:.2f}, Hours: {most_reliable['Hours']:.2f}, Probability: {most_reliable['Probability']:.4f}")
else:
    # Find the combination with highest probability
    highest_prob = df.sort_values('Probability', ascending=False).iloc[0]
    print(f"Progress: {highest_prob['Progress']:.2f}, Hours: {highest_prob['Hours']:.2f}, Probability: {highest_prob['Probability']:.4f}")
    print("Note: Even this combination doesn't result in a 'Delayed' prediction.")

print("\nAnalysis complete!")
