import os
import sys
from src.app import app

if __name__ == '__main__':
    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the Flask application
    print("Starting Bitcoin Price Prediction Web Application...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, port=5000) 