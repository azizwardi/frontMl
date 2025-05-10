# Project Management AI

A Flask-based web application that uses machine learning to help with project management tasks including delay prediction, team workload optimization, and task prioritization.

## Features

- **Project Delay Prediction**: Predict if your project will be delayed based on key metrics
- **Team Workload Optimization**: Get recommendations for team members based on skills and current workload
- **Task Prioritization**: Prioritize tasks based on urgency and importance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/project-management-ai.git
cd project-management-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure model files are in the `models/` directory:
   - `predicmodel.pkl` - For delay prediction
   - `Recommendmodel.pkl` - For team recommendations
   - `Segmentmodel.pkl` - For task prioritization

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://127.0.0.1:5000/`

3. Register a new account or login with existing credentials

4. Navigate to the dashboard to access all features

## Project Structure

- `app.py` - Main Flask application
- `prediction.py` - Project delay prediction logic
- `recommendation.py` - Team recommendation logic
- `segmentation.py` - Task prioritization logic
- `templates/` - HTML templates for the web interface
- `static/` - CSS, JavaScript, and other static files
- `models/` - Machine learning model files

## Database

The application uses a simple JSON file (`db.json`) as its database. This file is automatically created when the application runs for the first time.

## Security Note

This application is designed for demonstration purposes. In a production environment, you should:

1. Use a proper database system instead of JSON files
2. Implement proper password hashing
3. Use a more secure session management system
4. Generate a proper secret key for Flask

## License

[MIT License](LICENSE)