from flask import Flask, request, jsonify
from groq_handler import GroqHandler
from models import db, Conversation, Diagnosis, UploadedImage, KnowledgeBase
import os
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
from werkzeug.utils import secure_filename
from flask_cors import CORS
import io
import logging
import json
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)
basedir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.join(basedir, 'database')
if not os.path.exists(db_dir):
    os.makedirs(db_dir)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(db_dir, 'medical_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

groq = GroqHandler(os.getenv("GROQ_API_KEY"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Initialize model with error handling
try:
    ct_model = models.resnet18(weights=None)  # Initialize without pretrained weights
    num_ftrs = ct_model.fc.in_features
    ct_model.fc = torch.nn.Linear(num_ftrs, 1)  # Output size 1
    ct_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ct_contrast_classifier.pth"), map_location="cpu"))
    ct_model.eval()
    model_status = 'loaded'
except Exception as e:
    logger.error(f"Model failed to load: {e}")
    ct_model = None
    model_status = f'error: {str(e)}'

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load drug data from local pickle
pkl_path = os.path.join(basedir, 'medicine_dataset.pkl')
with open(pkl_path, 'rb') as f:
    global_drug_data = pickle.load(f)

@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.path} - {request.remote_addr}")

@app.after_request
def log_response_info(response):
    logger.info(f"Response: {request.method} {request.path} - {response.status_code}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled Exception: {str(e)}")
    return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    mode = data.get('mode', 'emotional')
    message = data['message']
    response = groq.get_response(message, mode)
    new_convo = Conversation(
        mode=mode,
        message=message,
        response=response['content'],
        emotion=response.get('emotion', 'neutral')
    )
    db.session.add(new_convo)
    db.session.commit()
    return jsonify(response)

@app.route('/diagnose', methods=['POST'])
def handle_diagnosis():
    data = request.json
    symptoms = data['symptoms']
    response = groq.get_response(symptoms, 'medical')
    new_diag = Diagnosis(
        symptoms=symptoms,
        diagnosis=response['content'],
        confidence=response.get('confidence', 0.0)
    )
    db.session.add(new_diag)
    db.session.commit()
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess image
        image_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = ct_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        result = {
            'has_contrast': bool(predicted_class),
            'confidence': round(confidence * 100, 2),
            'status': 'success'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/ai_doctor', methods=['POST'])
def ai_doctor():
    symptoms = request.form.get('symptoms', '')
    history = request.form.get('history', '')
    files = request.files.getlist('files')
    image_findings = []
    for file in files:
        try:
            if file.filename.lower().endswith('.pdf'):
                # Placeholder: handle PDF/lab result extraction here
                finding = f"{secure_filename(file.filename)}: PDF/lab result support coming soon."
                image_findings.append(finding)
                continue
            image = Image.open(file.stream).convert("RGB")
            img = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = ct_model(img)
                prob = torch.sigmoid(output).item()
                prediction = 1 if prob > 0.5 else 0
            finding = f"{secure_filename(file.filename)}: {'Contrast' if prediction else 'No Contrast'} (prob: {prob:.2f})"
            image_findings.append(finding)
        except Exception as e:
            image_findings.append(f"{secure_filename(file.filename)}: Error - {str(e)}")
    kb_context = "Relevant medical knowledge base context here."
    prompt = f"""
    Patient symptoms: {symptoms}
    History: {history}
    Image findings: {'; '.join(image_findings)}
    Relevant KB: {kb_context}

    As an AI medical assistant, provide:
    - Most likely diagnosis (label as: Diagnosis: ...)
    - Recommended tests (label as: Recommended Tests: ...)
    - Possible medicines (label as: Medicines: ...). For each medicine, recommend both Indian and global options if available, and specify which is which. For each medicine, include the proper dosage and duration to be taken, if possible. Include disclaimers.
    - Explanation and references
    """
    response = groq.get_response(prompt, 'medical')
    disclaimer = "This is not a substitute for professional medical advice. Always consult a healthcare professional."
    # Log interaction
    for file in files:
        new_img = UploadedImage(filename=secure_filename(file.filename), result='; '.join(image_findings))
        db.session.add(new_img)
    db.session.commit()
    return jsonify({
        'diagnosis': response.get('content', ''),
        'image_findings': image_findings,
        'kb_context': kb_context,
        'disclaimer': disclaimer
    })

# Knowledge base management endpoint (add/search entries)
@app.route('/kb', methods=['POST'])
def add_kb_entry():
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    kb_entry = KnowledgeBase(question=question, answer=answer)
    db.session.add(kb_entry)
    db.session.commit()
    return jsonify({'status': 'added'})

@app.route('/kb_search', methods=['GET'])
def kb_search():
    query = request.args.get('q', '')
    results = KnowledgeBase.query.filter(KnowledgeBase.question.ilike(f"%{query}%")).all()
    return jsonify({'results': [{"question": r.question, "answer": r.answer} for r in results]})

@app.route('/prescription_safety', methods=['POST'])
def prescription_safety():
    data = request.json
    prescriptions = data.get('prescriptions', [])
    patient_context = data.get('patient_context', {})
    alerts = []
    for p in prescriptions:
        # Example rule: Agamree should not be IV
        if p['medication'].lower() == 'agamree' and p['route'].lower() == 'iv':
            alerts.append({
                'risk': 'High',
                'medication': p['medication'],
                'reason': 'IV route is inappropriate for Agamree (oral only).',
                'recommendation': 'Switch to oral administration using the provided oral syringe. Recommended dosage is 6 mg/kg once daily with a meal.'
            })
        # Example dosage check
        try:
            dose = float(p['dosage'])
            if dose < 1 or dose > 100:
                alerts.append({
                    'risk': 'Medium',
                    'medication': p['medication'],
                    'reason': f'Dosage {dose} mg is outside the safe range (1–100 mg).',
                    'recommendation': 'Check the recommended dosage for this medication.'
                })
        except Exception:
            pass
    prompt = f"""
Patient context: {patient_context}
Prescriptions: {prescriptions}
Alerts: {alerts}
Explain the risks, what to do next, and why.
"""
    response = groq.get_response(prompt, 'medical')
    return jsonify({'alerts': alerts, 'llm_explanation': response.get('content', '')})

@app.route('/drug_interactions', methods=['POST'])
def drug_interactions():
    data = request.json
    medicines = [m.lower().strip() for m in data.get('medicines', []) if m.strip()]
    allergies = [a.lower().strip() for a in data.get('allergies', []) if a.strip()] if 'allergies' in data else []
    dosages = data.get('dosages', {})  # {medicine: dose}
    if not medicines or len(medicines) < 2:
        return jsonify({'error': 'Please provide at least two medicines.'}), 400

    interactions = []
    cost_info = []
    allergy_warnings = []
    dosage_warnings = []
    medicine_info = {}
    for i, med1 in enumerate(medicines):
        info1 = global_drug_data.get(med1, {})
        # Cost info
        if 'price' in info1:
            cost_info.append({'medicine': med1, 'price': info1['price']})
        # Side effects, brand/generic info
        medicine_info[med1] = {
            'brand_name': med1.title(),
            'generic_name': info1.get('generic_name', ''),
            'price': info1.get('price', ''),
            'manufacturer': info1.get('manufacturer', ''),
            'type': info1.get('type', ''),
            'form': info1.get('form', ''),
            'strength': info1.get('strength', ''),
            'side_effects': info1.get('warnings', ''),
            'packaging': info1.get('packaging', ''),
            'schedule': info1.get('schedule', ''),
            'dosage': info1.get('dosage', {}),
            'contraindications': info1.get('contraindications', []),
            'interactions': info1.get('interactions', [])
        }
        # Allergy check
        for allergy in allergies:
            if allergy in med1:
                allergy_warnings.append({
                    'medicine': med1,
                    'allergy': allergy,
                    'warning': f"Allergy alert: {med1.title()} matches allergy '{allergy}'"
                })
        # Dosage check
        if med1 in dosages:
            try:
                dose = float(dosages[med1])
                min_dose = info1.get('min_dose')
                max_dose = info1.get('max_dose')
                if min_dose and dose < min_dose:
                    dosage_warnings.append({
                        'medicine': med1,
                        'dose': dose,
                        'warning': f"Dose {dose}mg is below minimum safe dose ({min_dose}mg) for {med1.title()}"
                    })
                if max_dose and dose > max_dose:
                    dosage_warnings.append({
                        'medicine': med1,
                        'dose': dose,
                        'warning': f"Dose {dose}mg is above maximum safe dose ({max_dose}mg) for {med1.title()}"
                    })
            except Exception:
                pass
        # Check interactions with other medicines
        for med2 in medicines[i+1:]:
            info2 = global_drug_data.get(med2, {})
            inter1 = info1.get('interactions', [])
            inter2 = info2.get('interactions', [])
            if isinstance(inter1, str):
                inter1 = [x.strip().lower() for x in inter1.split(',') if x.strip()]
            if isinstance(inter2, str):
                inter2 = [x.strip().lower() for x in inter2.split(',') if x.strip()]
            if med2 in inter1 or med1 in inter2:
                interactions.append({
                    'medicine_1': med1,
                    'medicine_2': med2,
                    'warning': f"Interaction found between {med1.title()} and {med2.title()}."
                })
    return jsonify({
        'interactions': interactions,
        'cost_info': cost_info,
        'allergy_warnings': allergy_warnings,
        'dosage_warnings': dosage_warnings,
        'medicine_info': medicine_info
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_status': model_status
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)
