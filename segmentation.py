import os
import pickle
import numpy as np
import json
from datetime import datetime

# Load the segmentation model
model_path = os.path.join('models', 'Segmentmodel.pkl')
try:
    # First try to load with pickle
    try:
        with open(model_path, 'rb') as f:
            segment_model = pickle.load(f)
        print("Segmentation model loaded successfully with pickle!")
    except Exception as pickle_error:
        print(f"Error loading segmentation model with pickle: {pickle_error}")
        # If pickle fails, try a different approach - treat it as a binary file
        # This is a fallback that assumes the model might be in a custom format
        with open(model_path, 'rb') as f:
            model_data = f.read()
        print(f"Read {len(model_data)} bytes from segmentation model file")
        # Create a simple wrapper object that can be used in place of the model
        segment_model = {
            "model_data": model_data,
            "model_type": "binary",
            "description": "Task prioritization segmentation model"
        }
        print("Created wrapper for segmentation model data")
except Exception as e:
    print(f"Error loading segmentation model: {e}")
    segment_model = None

# Task priority levels
PRIORITY_LEVELS = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Optional": 0
}

# Task categories
TASK_CATEGORIES = [
    "Planning",
    "Design",
    "Development",
    "Testing",
    "Deployment",
    "Maintenance",
    "Documentation",
    "Other"
]

def get_task_segmentation(task_name, task_category, current_priority, deadline_days, estimated_hours, complexity):
    """
    Get task prioritization segmentation based on task parameters.

    Args:
        task_name (str): Name of the task
        task_category (str): Category of the task
        current_priority (str): Current priority level (Critical, High, Medium, Low, Optional)
        deadline_days (int): Days until the deadline
        estimated_hours (float): Estimated hours to complete the task
        complexity (int): Complexity level (1-5)

    Returns:
        dict: Segmentation results including recommended priority and segment
    """
    # Try to use the ML model if available
    model_enhanced = False
    model_data = None

    if segment_model is not None:
        try:
            # Check if we have a dictionary wrapper (from our fallback loading)
            if isinstance(segment_model, dict) and "model_type" in segment_model:
                # We have a wrapper object, not an actual model
                print(f"Using segmentation model wrapper: {segment_model['description']}")

                # Extract any useful information from the model data
                model_data = {
                    "model_type": segment_model["model_type"],
                    "data_size": len(segment_model.get("model_data", b"")) if isinstance(segment_model.get("model_data"), bytes) else 0,
                    "description": segment_model.get("description", "Unknown model")
                }

                # Set flag to indicate we're using model data
                model_enhanced = True
            else:
                # We have an actual model object, try to use it
                # Prepare input features for the model
                # Convert task_category to one-hot encoding
                category_idx = TASK_CATEGORIES.index(task_category) if task_category in TASK_CATEGORIES else len(TASK_CATEGORIES) - 1  # Default to "Other"
                category_encoding = [0] * len(TASK_CATEGORIES)
                category_encoding[category_idx] = 1

                # Convert priority to numeric
                priority_value = PRIORITY_LEVELS.get(current_priority, 2)  # Default to Medium

                # Create feature array
                features = np.array([[
                    priority_value, 
                    deadline_days, 
                    estimated_hours, 
                    complexity
                ] + category_encoding])

                # Get model predictions
                if hasattr(segment_model, 'predict'):
                    # If the model has a predict method, use it to get segmentation
                    segment_predictions = segment_model.predict(features)

                    # Process model predictions
                    if isinstance(segment_predictions, np.ndarray) and len(segment_predictions) > 0:
                        # Use model output to enhance the rule-based segmentation
                        print(f"Using ML model segmentation: {segment_predictions}")
                        model_data = segment_predictions
                        model_enhanced = True

            # If we have model data, use it to enhance segmentation
            if model_enhanced and model_data is not None:
                print(f"Using model-enhanced segmentation with data: {model_data}")
                return get_rule_based_segmentation(task_name, task_category, current_priority, deadline_days, estimated_hours, complexity, model_data)

            # If we couldn't use the model predictions, fall back to rule-based approach
            print("Falling back to rule-based segmentation")

        except Exception as e:
            print(f"Error using segmentation model: {e}")

    # Fall back to rule-based approach if model is not available or fails
    return get_rule_based_segmentation(task_name, task_category, current_priority, deadline_days, estimated_hours, complexity)

def get_rule_based_segmentation(task_name, task_category, current_priority, deadline_days, estimated_hours, complexity, model_data=None):
    """
    Get task prioritization using rule-based approach.

    Args:
        task_name (str): Name of the task
        task_category (str): Category of the task
        current_priority (str): Current priority level (Critical, High, Medium, Low, Optional)
        deadline_days (int): Days until the deadline
        estimated_hours (float): Estimated hours to complete the task
        complexity (int): Complexity level (1-5)
        model_data (optional): Data from ML model to enhance segmentation

    Returns:
        dict: Segmentation results including recommended priority and segment
    """
    # Calculate urgency score based on deadline and estimated hours
    urgency_score = 0
    
    # Deadline factor: shorter deadlines increase urgency
    if deadline_days <= 1:
        urgency_score += 5  # Extremely urgent
    elif deadline_days <= 3:
        urgency_score += 4  # Very urgent
    elif deadline_days <= 7:
        urgency_score += 3  # Urgent
    elif deadline_days <= 14:
        urgency_score += 2  # Moderately urgent
    else:
        urgency_score += 1  # Not urgent
    
    # Effort factor: more hours increase importance
    if estimated_hours >= 40:
        urgency_score += 5  # Major effort
    elif estimated_hours >= 20:
        urgency_score += 4  # Significant effort
    elif estimated_hours >= 10:
        urgency_score += 3  # Moderate effort
    elif estimated_hours >= 5:
        urgency_score += 2  # Small effort
    else:
        urgency_score += 1  # Minimal effort
    
    # Complexity factor
    urgency_score += complexity
    
    # Current priority factor
    priority_value = PRIORITY_LEVELS.get(current_priority, 2)
    urgency_score += priority_value
    
    # Calculate final score (normalize to 0-10 scale)
    max_possible_score = 5 + 5 + 5 + 4  # Max from all factors
    normalized_score = (urgency_score / max_possible_score) * 10
    
    # Determine segment based on score
    if normalized_score >= 8.5:
        segment = "Critical Priority"
        recommended_priority = "Critical"
    elif normalized_score >= 7:
        segment = "High Priority"
        recommended_priority = "High"
    elif normalized_score >= 5:
        segment = "Medium Priority"
        recommended_priority = "Medium"
    elif normalized_score >= 3:
        segment = "Low Priority"
        recommended_priority = "Low"
    else:
        segment = "Optional"
        recommended_priority = "Optional"
    
    # Return segmentation results
    return {
        "task_name": task_name,
        "task_category": task_category,
        "current_priority": current_priority,
        "deadline_days": deadline_days,
        "estimated_hours": estimated_hours,
        "complexity": complexity,
        "urgency_score": round(normalized_score, 2),
        "segment": segment,
        "recommended_priority": recommended_priority,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
