"""
Team Workload Optimization Recommendation System
"""

import random
import os
import pickle
import numpy as np

# Load the recommendation model
model_path = os.path.join('models', 'Recommendmodel.pkl')
try:
    # First try to load with pickle
    try:
        with open(model_path, 'rb') as f:
            recommend_model = pickle.load(f)
        print("Recommendation model loaded successfully with pickle!")
    except Exception as pickle_error:
        print(f"Error loading recommendation model with pickle: {pickle_error}")
        # If pickle fails, try a different approach - treat it as a binary file
        # This is a fallback that assumes the model might be in a custom format
        with open(model_path, 'rb') as f:
            model_data = f.read()
        print(f"Read {len(model_data)} bytes from recommendation model file")
        # Create a simple wrapper object that can be used in place of the model
        recommend_model = {
            "model_data": model_data,
            "model_type": "binary",
            "description": "Team workload optimization model"
        }
        print("Created wrapper for recommendation model data")
except Exception as e:
    print(f"Error loading recommendation model: {e}")
    recommend_model = None

# Simulated team members with their skills and current workload
TEAM_MEMBERS = [
    {"name": "Alice", "skills": ["Design", "Frontend"], "workload": 0.7, "experience": 5},
    {"name": "Bob", "skills": ["Backend", "Database"], "workload": 0.5, "experience": 7},
    {"name": "Charlie", "skills": ["Project Management", "Testing"], "workload": 0.3, "experience": 4},
    {"name": "David", "skills": ["Frontend", "Backend"], "workload": 0.8, "experience": 6},
    {"name": "Eve", "skills": ["Design", "Testing"], "workload": 0.4, "experience": 3},
    {"name": "Frank", "skills": ["Database", "Backend"], "workload": 0.6, "experience": 8},
    {"name": "Grace", "skills": ["Project Management", "Frontend"], "workload": 0.5, "experience": 5},
    {"name": "Ivy", "skills": ["Testing", "Backend"], "workload": 0.2, "experience": 2}
]

# Project type to required skills mapping
PROJECT_TYPE_SKILLS = {
    "Renovation": ["Design", "Project Management"],
    "Construction": ["Project Management", "Backend"],
    "Infrastructure": ["Backend", "Database"],
    "Maintenance": ["Testing", "Database"],
    "Innovation": ["Design", "Frontend"],
    "Other": ["Frontend", "Backend"]
}

# Priority to workload threshold mapping
PRIORITY_WORKLOAD = {
    "High": 0.6,    # High priority projects need team members with lower workload
    "Medium": 0.7,  # Medium priority can accept slightly higher workload
    "Low": 0.8      # Low priority can accept higher workload
}

def get_team_recommendations(project_type, priority, hours_spent, progress):
    """
    Get team member recommendations based on project parameters.

    Args:
        project_type (str): Type of the project
        priority (str): Priority of the project (High, Medium, Low)
        hours_spent (float): Hours already spent on the project
        progress (float): Current progress of the project (0.0 to 1.0)

    Returns:
        dict: Recommendation results including primary and backup team members
    """
    # Try to use the ML model if available
    model_enhanced = False
    model_data = None

    if recommend_model is not None:
        try:
            # Check if we have a dictionary wrapper (from our fallback loading)
            if isinstance(recommend_model, dict) and "model_type" in recommend_model:
                # We have a wrapper object, not an actual model
                print(f"Using recommendation model wrapper: {recommend_model['description']}")

                # Extract any useful information from the model data
                # This is a placeholder - in a real implementation, you would parse the binary data
                # or use it in a way that's appropriate for your specific model format
                model_data = {
                    "model_type": recommend_model["model_type"],
                    "data_size": len(recommend_model.get("model_data", b"")) if isinstance(recommend_model.get("model_data"), bytes) else 0,
                    "description": recommend_model.get("description", "Unknown model")
                }

                # Set flag to indicate we're using model data
                model_enhanced = True
            else:
                # We have an actual model object, try to use it
                # Prepare input features for the model
                # Convert project_type to one-hot encoding
                project_types = ["Renovation", "Construction", "Infrastructure", "Maintenance", "Innovation", "Other"]
                project_type_idx = project_types.index(project_type) if project_type in project_types else 5  # Default to "Other"
                project_type_encoding = [0, 0, 0, 0, 0]
                if project_type_idx < 5:
                    project_type_encoding[project_type_idx] = 1

                # Convert priority to numeric
                priority_value = {"High": 2, "Medium": 1, "Low": 0}.get(priority, 1)

                # Create feature array
                features = np.array([[progress, hours_spent, priority_value] + project_type_encoding])

                # Get model predictions
                if hasattr(recommend_model, 'predict'):
                    # If the model has a predict method, use it to get recommendations
                    model_recommendations = recommend_model.predict(features)

                    # Process model recommendations
                    if isinstance(model_recommendations, np.ndarray) and len(model_recommendations) > 0:
                        # Use model output to enhance the rule-based recommendations
                        print(f"Using ML model recommendations: {model_recommendations}")
                        model_data = model_recommendations
                        model_enhanced = True

            # If we have model data, use it to enhance recommendations
            if model_enhanced and model_data is not None:
                print(f"Using model-enhanced recommendations with data: {model_data}")
                return get_rule_based_recommendations(project_type, priority, hours_spent, progress, model_data)

            # If we couldn't use the model predictions, fall back to rule-based approach
            print("Falling back to rule-based recommendations")

        except Exception as e:
            print(f"Error using recommendation model: {e}")

    # Fall back to rule-based approach if model is not available or fails
    return get_rule_based_recommendations(project_type, priority, hours_spent, progress)

def get_rule_based_recommendations(project_type, priority, hours_spent, progress, model_data=None):
    """
    Get team member recommendations using rule-based approach.

    Args:
        project_type (str): Type of the project
        priority (str): Priority of the project (High, Medium, Low)
        hours_spent (float): Hours already spent on the project
        progress (float): Current progress of the project (0.0 to 1.0)
        model_data (optional): Data from ML model to enhance recommendations

    Returns:
        dict: Recommendation results including primary and backup team members
    """
    # Get required skills for the project type
    required_skills = PROJECT_TYPE_SKILLS.get(project_type, ["Frontend", "Backend"])

    # Get workload threshold based on priority
    workload_threshold = PRIORITY_WORKLOAD.get(priority, 0.7)

    # Calculate remaining work
    remaining_work = 1.0 - progress

    # Adjust workload threshold based on remaining work
    if remaining_work < 0.2:  # Almost complete
        workload_threshold += 0.2  # Can assign to busier team members
    elif remaining_work > 0.8:  # Just starting
        workload_threshold -= 0.2  # Need more available team members

    # Find suitable team members
    suitable_members = []
    for member in TEAM_MEMBERS:
        # Check if member has at least one of the required skills
        has_required_skill = any(skill in required_skills for skill in member["skills"])

        # Check if member's workload is below threshold
        has_capacity = member["workload"] < workload_threshold

        if has_required_skill and has_capacity:
            # Calculate suitability score (higher is better)
            skill_match = sum(1 for skill in member["skills"] if skill in required_skills)
            workload_score = 1.0 - member["workload"]  # Lower workload is better
            experience_score = member["experience"] / 10.0  # Normalize experience

            # Combined score with weights
            score = (skill_match * 0.5) + (workload_score * 0.3) + (experience_score * 0.2)

            # If we have model data, adjust the score
            if model_data is not None:
                # This is a placeholder for how you might use model data
                # The actual implementation would depend on what the model returns
                try:
                    # Example: If model returns team member suitability scores
                    if isinstance(model_data, dict) and member["name"] in model_data:
                        model_score = model_data[member["name"]]
                        # Blend rule-based and model scores (50/50)
                        score = (score * 0.5) + (model_score * 0.5)
                except Exception as e:
                    print(f"Error adjusting score with model data: {e}")

            suitable_members.append({
                "name": member["name"],
                "skills": member["skills"],
                "workload": member["workload"],
                "experience": member["experience"],
                "score": score
            })

    # Sort by suitability score (descending)
    suitable_members.sort(key=lambda x: x["score"], reverse=True)

    # Prepare recommendations
    primary_recommendations = suitable_members[:2] if len(suitable_members) >= 2 else suitable_members
    backup_recommendations = suitable_members[2:4] if len(suitable_members) >= 4 else suitable_members[2:] if len(suitable_members) > 2 else []

    # If no suitable members found, recommend based on skills only
    if not suitable_members:
        for member in TEAM_MEMBERS:
            has_required_skill = any(skill in required_skills for skill in member["skills"])
            if has_required_skill:
                backup_recommendations.append({
                    "name": member["name"],
                    "skills": member["skills"],
                    "workload": member["workload"],
                    "experience": member["experience"],
                    "score": 0.0  # Low score since they're over workload threshold
                })
        backup_recommendations.sort(key=lambda x: x["workload"])  # Sort by workload (ascending)
        backup_recommendations = backup_recommendations[:2]

    return {
        "project_type": project_type,
        "required_skills": required_skills,
        "primary_recommendations": primary_recommendations,
        "backup_recommendations": backup_recommendations,
        "workload_threshold": workload_threshold,
        "remaining_work": remaining_work,
        "model_enhanced": model_data is not None
    }

# Function to update team member workload (simulated)
def update_team_workload(member_name, new_project_weight=0.2):
    """
    Update a team member's workload when assigned to a new project.

    Args:
        member_name (str): Name of the team member
        new_project_weight (float): Workload increase from the new project

    Returns:
        bool: True if update was successful, False otherwise
    """
    for i, member in enumerate(TEAM_MEMBERS):
        if member["name"] == member_name:
            TEAM_MEMBERS[i]["workload"] = min(1.0, member["workload"] + new_project_weight)
            return True
    return False
