"""
Entry point for Streamlit Cloud deployment.
This file allows running the app from the repository root.
"""
import sys
import os

# Set up paths
repo_root = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(repo_root, 'frontend')

# Add frontend directory to path so imports work
sys.path.insert(0, frontend_dir)

# Change to frontend directory for relative file operations
os.chdir(frontend_dir)

# Import and run the main app
from main import main

if __name__ == "__main__":
    main()
