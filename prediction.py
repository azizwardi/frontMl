"""
Project Delay Prediction Module

This module handles the prediction logic for determining if a project will be delayed
based on various project parameters.
"""

import os
import pickle
import numpy as np

# Load the prediction model
model_path = os.path.join('models', 'predicmodel.pkl')
try:
    with open(model_path, 'rb') as f:
        predict_model = pickle.load(f)
    print("Prediction model loaded successfully!")
except Exception as e:
    print(f"Error loading prediction model: {e}")
    predict_model = None

# Project type mapping for one-hot encoding
PROJECT_TYPE_MAP = {
    'Renovation': [1, 0, 0, 0, 0],
    'Construction': [0, 1, 0, 0, 0],
    'Infrastructure': [0, 0, 1, 0, 0],
    'Maintenance': [0, 0, 0, 1, 0],
    'Innovation': [0, 0, 0, 0, 1],
    'Other': [0, 0, 0, 0, 0]  # Default/Other
}

# Priority mapping for encoding
PRIORITY_MAP = {
    'High': 2,
    'Medium': 1,
    'Low': 0
}

def get_project_delay_prediction(project_type, priority, hours_spent, progress):
    """
    Predict if a project will be delayed based on its parameters.

    Args:
        project_type (str): Type of the project (Renovation, Construction, etc.)
        priority (str): Priority of the project (High, Medium, Low)
        hours_spent (float): Hours already spent on the project
        progress (float): Current progress of the project (0.0 to 1.0)

    Returns:
        dict: Prediction results including delay status and probability
    """
    if predict_model is None:
        return {
            'result': 'Unknown',
            'probability': 0.0,
            'error': 'Prediction model not available'
        }

    try:
        # Encode priority
        priority_encoded = PRIORITY_MAP.get(priority, 0)

        # Encode project type (one-hot encoding)
        project_type_encoded = PROJECT_TYPE_MAP.get(project_type, [0, 0, 0, 0, 0])

        # Create feature array with 8 features in the expected order
        features = np.array([[
            progress,                  # Progress
            hours_spent,               # Hours Spent
            priority_encoded,          # Priority_Encoded
            project_type_encoded[0],   # Project_Type_0
            project_type_encoded[1],   # Project_Type_1
            project_type_encoded[2],   # Project_Type_2
            project_type_encoded[3],   # Project_Type_3
            project_type_encoded[4]    # Project_Type_4
        ]])

        # Get the probability of delay
        probability = None
        if hasattr(predict_model, 'predict_proba'):
            probability = float(predict_model.predict_proba(features)[0][1])
        else:
            # If predict_proba is not available, use the model's prediction
            prediction = predict_model.predict(features)
            probability = 1.0 if prediction[0] == 1 else 0.0

        # Use a custom threshold of 0.4 (40%) instead of the default 0.5 (50%)
        # This makes the model more sensitive to potential delays
        prediction_result = "Delayed" if probability >= 0.4 else "On Track"

        return {
            'result': prediction_result,
            'probability': probability,
            'threshold': 0.4,
            'features': {
                'progress': progress,
                'hours_spent': hours_spent,
                'priority': priority,
                'priority_encoded': priority_encoded,
                'project_type': project_type,
                'project_type_encoded': project_type_encoded
            }
        }

    except Exception as e:
        return {
            'result': 'Error',
            'probability': 0.0,
            'error': str(e)
        }

def analyze_model():
    """
    Analyze the model's behavior with different input parameters.
    
    Returns:
        dict: Analysis results for different parameter variations
    """
    if predict_model is None:
        return {'error': 'Model not loaded'}

    # Test with different progress values
    progress_tests = []
    for progress in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Renovation project with High priority and 30 hours
        features = np.array([[progress, 30, 2, 1, 0, 0, 0, 0]])
        probability = predict_model.predict_proba(features)[0][1] if hasattr(predict_model, 'predict_proba') else None

        progress_tests.append({
            'progress': progress,
            'prediction': 'Delayed' if probability >= 0.4 else 'On Track',
            'probability': float(probability) if probability is not None else None
        })

    # Test with different hours values
    hours_tests = []
    for hours in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        # Renovation project with High priority and 30% progress
        features = np.array([[0.3, hours, 2, 1, 0, 0, 0, 0]])
        probability = predict_model.predict_proba(features)[0][1] if hasattr(predict_model, 'predict_proba') else None

        hours_tests.append({
            'hours': hours,
            'prediction': 'Delayed' if probability >= 0.4 else 'On Track',
            'probability': float(probability) if probability is not None else None
        })

    # Return all test results
    return {
        'progress_tests': progress_tests,
        'hours_tests': hours_tests,
        'feature_importance': predict_model.feature_importances_.tolist() if hasattr(predict_model, 'feature_importances_') else None,
        'feature_names': predict_model.feature_names_in_.tolist() if hasattr(predict_model, 'feature_names_in_') else None
    }
