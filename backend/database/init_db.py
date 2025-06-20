import sys
from dotenv import load_dotenv
load_dotenv()
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import db
from app import app

with app.app_context():
    db.create_all()
    print("Database initialized.")
