from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import json
import os
import pickle
import numpy as np
from datetime import datetime
from functools import wraps
from recommendation import get_team_recommendations, update_team_workload
from segmentation import get_task_segmentation

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key in production

# Database file path
DB_FILE = 'db.json'

# Initialize database if it doesn't exist
if not os.path.exists(DB_FILE):
    with open(DB_FILE, 'w') as f:
        json.dump({'users': []}, f)

# Load the prediction model
model_path = os.path.join('models', 'predicmodel.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Prediction model loaded successfully!")
except Exception as e:
    print(f"Error loading prediction model: {e}")
    model = None

# Note: We don't need to load the recommendation model here
# It's already handled in the recommendation.py module

# Helper functions for database operations
def get_db():
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_db(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=4)

def get_user_by_username(username):
    db = get_db()
    for user in db['users']:
        if user['username'] == username:
            return user
    return None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validation
        if not username or not password:
            flash('Username and password are required', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')

        # Check if username already exists
        if get_user_by_username(username):
            flash('Username already exists', 'danger')
            return render_template('register.html')

        # Add user to database
        db = get_db()
        user_id = len(db['users']) + 1
        db['users'].append({
            'id': user_id,
            'username': username,
            'password': password,  # In a real app, hash the password!
            'predictions': []
        })
        save_db(db)

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user_by_username(username)

        if user and user['password'] == password:  # In a real app, verify the hashed password
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    if request.method == 'POST':
        try:
            # Get form data
            project_type = request.form['project_type']
            priority = request.form['priority']
            hours_spent = float(request.form['hours_spent'])
            progress = float(request.form['progress'])

            # Prepare input for prediction
            # This should match the format expected by your model
            input_data = {
                'Project Type': project_type,
                'Priority': priority,
                'Hours Spent': hours_spent,
                'Progress': progress
            }

            # Make prediction
            if model:
                # Convert input data to the format expected by the model
                # Based on model.feature_names_in_: ['Progress' 'Hours Spent' 'Priority_Encoded' 'Project_Type_0' 'Project_Type_1' 'Project_Type_2' 'Project_Type_3' 'Project_Type_4']

                # Encode priority (High=2, Medium=1, Low=0)
                priority_encoded = 2 if priority == 'High' else (1 if priority == 'Medium' else 0)

                # Encode project type (one-hot encoding for 5 types)
                project_type_map = {
                    'Renovation': [1, 0, 0, 0, 0],
                    'Construction': [0, 1, 0, 0, 0],
                    'Infrastructure': [0, 0, 1, 0, 0],
                    'Maintenance': [0, 0, 0, 1, 0],
                    'Innovation': [0, 0, 0, 0, 1],
                    'Other': [0, 0, 0, 0, 0]  # Default/Other
                }
                project_type_encoded = project_type_map.get(project_type, [0, 0, 0, 0, 0])

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
                if hasattr(model, 'predict_proba'):
                    probability = float(model.predict_proba(features)[0][1])
                else:
                    # If predict_proba is not available, use the model's prediction
                    prediction = model.predict(features)
                    probability = 1.0 if prediction[0] == 1 else 0.0

                # Use a custom threshold of 0.4 (40%) instead of the default 0.5 (50%)
                # This makes the model more sensitive to potential delays
                prediction_result = "Delayed" if probability >= 0.4 else "On Track"

                # Save prediction to database
                user_id = session.get('user_id')
                db = get_db()

                # Find the user in the database
                for user in db['users']:
                    if user['id'] == user_id:
                        # Create prediction record
                        prediction_record = {
                            'id': len(user.get('predictions', [])) + 1,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'result': prediction_result,
                            'probability': probability,
                            'input_data': input_data
                        }

                        # Initialize predictions list if it doesn't exist
                        if 'predictions' not in user:
                            user['predictions'] = []

                        # Add prediction to user's predictions
                        user['predictions'].append(prediction_record)
                        save_db(db)
                        break

                flash('Prediction saved successfully!', 'success')
                return render_template('prediction.html', prediction=prediction_result, probability=probability, input_data=input_data)
            else:
                flash('Prediction model not available', 'danger')
                return render_template('prediction.html')

        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'danger')
            return render_template('prediction.html')

    return render_template('prediction.html')

@app.route('/history')
@login_required
def history():
    user_id = session.get('user_id')
    db = get_db()

    # Find the user and get their prediction history
    predictions = []
    for user in db['users']:
        if user['id'] == user_id:
            predictions = user.get('predictions', [])
            # Sort predictions by timestamp (newest first)
            predictions = sorted(predictions, key=lambda x: x.get('timestamp', ''), reverse=True)
            break

    return render_template('history.html', predictions=predictions)

@app.route('/analyze_model')
def analyze_model():
    """Special route to analyze the model's behavior."""
    if not model:
        return jsonify({'error': 'Model not loaded'})

    # Test with different progress values
    progress_tests = []
    for progress in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Renovation project with High priority and 30 hours
        features = np.array([[progress, 30, 2, 1, 0, 0, 0, 0]])
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

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
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        hours_tests.append({
            'hours': hours,
            'prediction': 'Delayed' if probability >= 0.4 else 'On Track',
            'probability': float(probability) if probability is not None else None
        })

    # Test with different priority values
    priority_tests = []
    for priority, priority_name in [(0, 'Low'), (1, 'Medium'), (2, 'High')]:
        # Renovation project with 30% progress and 30 hours
        features = np.array([[0.3, 30, priority, 1, 0, 0, 0, 0]])
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        priority_tests.append({
            'priority': priority_name,
            'prediction': 'Delayed' if probability >= 0.4 else 'On Track',
            'probability': float(probability) if probability is not None else None
        })

    # Test with different project types
    project_type_tests = []
    project_types = ['Renovation', 'Construction', 'Infrastructure', 'Maintenance', 'Innovation', 'Other']
    for i, project_type in enumerate(project_types):
        # Create one-hot encoding for project type
        project_type_encoding = [0, 0, 0, 0, 0]
        if i < 5:  # Skip 'Other' which is all zeros
            project_type_encoding[i] = 1

        # Project with High priority, 30% progress, and 30 hours
        features = np.array([[0.3, 30, 2] + project_type_encoding])
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        project_type_tests.append({
            'project_type': project_type,
            'prediction': 'Delayed' if probability >= 0.4 else 'On Track',
            'probability': float(probability) if probability is not None else None
        })

    # Test extreme cases
    extreme_tests = []
    extreme_cases = [
        {'progress': 0.01, 'hours': 100, 'priority': 'High', 'project_type': 'Renovation'},
        {'progress': 0.0, 'hours': 100, 'priority': 'High', 'project_type': 'Renovation'},
        {'progress': 0.0, 'hours': 200, 'priority': 'High', 'project_type': 'Renovation'},
        {'progress': 0.0, 'hours': 500, 'priority': 'High', 'project_type': 'Renovation'},
    ]

    for case in extreme_cases:
        # Create features based on the case
        project_type_idx = project_types.index(case['project_type'])
        project_type_encoding = [0, 0, 0, 0, 0]
        if project_type_idx < 5:
            project_type_encoding[project_type_idx] = 1

        priority_value = {'Low': 0, 'Medium': 1, 'High': 2}[case['priority']]

        features = np.array([[case['progress'], case['hours'], priority_value] + project_type_encoding])
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        extreme_tests.append({
            'case': case,
            'prediction': 'Delayed' if probability >= 0.4 else 'On Track',
            'probability': float(probability) if probability is not None else None
        })

    # Return all test results
    return jsonify({
        'progress_tests': progress_tests,
        'hours_tests': hours_tests,
        'priority_tests': priority_tests,
        'project_type_tests': project_type_tests,
        'extreme_tests': extreme_tests,
        'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
        'feature_names': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None
    })

@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    """Route for team workload optimization recommendations."""
    if request.method == 'POST':
        try:
            # Get form data
            project_type = request.form['project_type']
            priority = request.form['priority']
            hours_spent = float(request.form['hours_spent'])
            progress = float(request.form['progress'])

            # Get recommendations
            recommendations = get_team_recommendations(
                project_type=project_type,
                priority=priority,
                hours_spent=hours_spent,
                progress=progress
            )

            # Save recommendation to database
            user_id = session.get('user_id')
            db = get_db()

            # Find the user in the database
            for user in db['users']:
                if user['id'] == user_id:
                    # Create recommendation record
                    recommendation_record = {
                        'id': len(user.get('recommendations', [])) + 1 if 'recommendations' in user else 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'project_type': project_type,
                        'priority': priority,
                        'hours_spent': hours_spent,
                        'progress': progress,
                        'recommendations': recommendations
                    }

                    # Initialize recommendations list if it doesn't exist
                    if 'recommendations' not in user:
                        user['recommendations'] = []

                    # Add recommendation to user's recommendations
                    user['recommendations'].append(recommendation_record)
                    save_db(db)
                    break

            flash('Team recommendations generated successfully!', 'success')
            return render_template('recommend.html',
                                  recommendations=recommendations,
                                  project_type=project_type,
                                  priority=priority,
                                  hours_spent=hours_spent,
                                  progress=progress)

        except Exception as e:
            flash(f'Error generating recommendations: {str(e)}', 'danger')
            return render_template('recommend.html')

    return render_template('recommend.html')

@app.route('/recommendation_history')
@login_required
def recommendation_history():
    """Route to view recommendation history."""
    user_id = session.get('user_id')
    db = get_db()

    # Find the user and get their recommendation history
    recommendations = []
    for user in db['users']:
        if user['id'] == user_id:
            recommendations = user.get('recommendations', [])
            # Sort recommendations by timestamp (newest first)
            recommendations = sorted(recommendations, key=lambda x: x.get('timestamp', ''), reverse=True)
            break

    return render_template('recommendation_history.html', recommendations=recommendations)

@app.route('/assign_team_member', methods=['POST'])
@login_required
def assign_team_member():
    """Route to assign a team member to a project."""
    if request.method == 'POST':
        try:
            member_name = request.form['member_name']
            project_weight = float(request.form.get('project_weight', 0.2))

            # Update team member workload
            success = update_team_workload(member_name, project_weight)

            if success:
                flash(f'Successfully assigned {member_name} to the project!', 'success')
            else:
                flash(f'Failed to assign {member_name} to the project.', 'danger')

            return redirect(url_for('recommend'))

        except Exception as e:
            flash(f'Error assigning team member: {str(e)}', 'danger')
            return redirect(url_for('recommend'))

@app.route('/segment', methods=['GET', 'POST'])
@login_required
def segment():
    """Route for task prioritization segmentation."""
    if request.method == 'POST':
        try:
            # Get form data
            task_name = request.form['task_name']
            task_category = request.form['task_category']
            current_priority = request.form['current_priority']
            deadline_days = int(request.form['deadline_days'])
            estimated_hours = float(request.form['estimated_hours'])
            complexity = int(request.form['complexity'])

            # Get segmentation
            segmentation = get_task_segmentation(
                task_name=task_name,
                task_category=task_category,
                current_priority=current_priority,
                deadline_days=deadline_days,
                estimated_hours=estimated_hours,
                complexity=complexity
            )

            # Save segmentation to database
            user_id = session.get('user_id')
            db = get_db()

            # Find the user in the database
            for user in db['users']:
                if user['id'] == user_id:
                    # Create segmentation record
                    segmentation_record = {
                        'id': len(user.get('segmentations', [])) + 1 if 'segmentations' in user else 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'task_name': task_name,
                        'task_category': task_category,
                        'current_priority': current_priority,
                        'deadline_days': deadline_days,
                        'estimated_hours': estimated_hours,
                        'complexity': complexity,
                        'segmentation': segmentation
                    }

                    # Initialize segmentations list if it doesn't exist
                    if 'segmentations' not in user:
                        user['segmentations'] = []

                    # Add segmentation to user's segmentations
                    user['segmentations'].append(segmentation_record)
                    save_db(db)
                    break

            flash('Task prioritization completed successfully!', 'success')
            return render_template('segment.html',
                                  segmentation=segmentation,
                                  task_name=task_name,
                                  task_category=task_category,
                                  current_priority=current_priority,
                                  deadline_days=deadline_days,
                                  estimated_hours=estimated_hours,
                                  complexity=complexity)

        except Exception as e:
            flash(f'Error generating task prioritization: {str(e)}', 'danger')
            return render_template('segment.html')

    return render_template('segment.html')

@app.route('/segmentation_history')
@login_required
def segmentation_history():
    """Route to view segmentation history."""
    user_id = session.get('user_id')
    db = get_db()

    # Find the user and get their segmentation history
    segmentations = []
    for user in db['users']:
        if user['id'] == user_id:
            segmentations = user.get('segmentations', [])
            # Sort segmentations by timestamp (newest first)
            segmentations = sorted(segmentations, key=lambda x: x.get('timestamp', ''), reverse=True)
            break

    return render_template('segmentation_history.html', segmentations=segmentations)

if __name__ == '__main__':
    app.run(debug=True)
