from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import json
import os
import numpy as np
from datetime import datetime
from functools import wraps
from prediction import get_project_delay_prediction, analyze_model
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

# Note: Model loading is now handled in the respective modules:
# - prediction.py: Handles the project delay prediction model
# - recommendation.py: Handles the team recommendation model
# - segmentation.py: Handles the task prioritization model

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
            input_data = {
                'Project Type': project_type,
                'Priority': priority,
                'Hours Spent': hours_spent,
                'Progress': progress
            }

            # Get prediction using the prediction module
            prediction_data = get_project_delay_prediction(
                project_type=project_type,
                priority=priority,
                hours_spent=hours_spent,
                progress=progress
            )

            # Check if prediction was successful
            if 'error' in prediction_data:
                flash(f'Prediction error: {prediction_data["error"]}', 'danger')
                return render_template('prediction.html')

            # Extract prediction result and probability
            prediction_result = prediction_data['result']
            probability = prediction_data['probability']

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
def analyze_model_route():
    """Special route to analyze the model's behavior."""
    # Use the analyze_model function from prediction.py
    analysis_results = analyze_model()

    if 'error' in analysis_results:
        return jsonify({'error': analysis_results['error']})

    return jsonify(analysis_results)

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
