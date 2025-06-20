from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mode = db.Column(db.String(20))
    message = db.Column(db.Text)
    response = db.Column(db.Text)
    emotion = db.Column(db.String(20))
    disclaimer = db.Column(db.Text, default="This is not a substitute for professional medical advice.")
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.Text)
    diagnosis = db.Column(db.Text)
    confidence = db.Column(db.Float)
    disclaimer = db.Column(db.Text, default="This is not a substitute for professional medical advice.")
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

class KnowledgeBase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    answer = db.Column(db.Text)

class UploadedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256))
    result = db.Column(db.String(256))
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
