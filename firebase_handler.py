import firebase_admin
from firebase_admin import credentials, auth, db
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz

load_dotenv()

def get_firebase_credentials():
    return {
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        "database_url": os.getenv("FIREBASE_DATABASE_URL")
    }

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(get_firebase_credentials())
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
        })

def login_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        user_data = db.reference(f'users/{user.uid}/info').get()
        if user_data:
            return user_data, None
        else:
            return None, "User data not found"
    except auth.UserNotFoundError:
        return None, "Invalid email or password"
    except Exception as e:
        return None, str(e)

def signup_user(email, password, department, interests, skills):
    try:
        user = auth.create_user(email=email, password=password)
        user_data = {
            "department": department,
            "interests": interests,
            "skills": skills,
            "uid": user.uid
        }
        db.reference(f'users/{user.uid}/info').set(user_data)
        return user_data, None
    except auth.EmailAlreadyExistsError:
        return None, "Email already exists"
    except Exception as e:
        return None, str(e)

def data_to_firebase(user_id, question, response, title):
    timestamp = datetime.now(pytz.utc).strftime("%Y-%m-%dT%H%M%S")
    log_data = {
        "question": question,
        "response": response,
        "title": title,
        "timestamp": timestamp
    }
    db.reference(f'users/{user_id}/chat/{timestamp}').set(log_data)

def get_user_data(user_id):
    return db.reference(f'users/{user_id}/info').get()

def get_conversation_titles(user_id):
    user_chat = db.reference(f'users/{user_id}/chat').get()
    if user_chat:
        return list(set([entry.get('title', 'Untitled') for entry in user_chat.values()]))
    return []

def get_recent_questions(user_id):
    user_chat = db.reference(f'users/{user_id}/chat').get()
    if user_chat:
        questions = [entry.get('question', '') for entry in user_chat.values()]
        return questions[-10:]
    return []

initialize_firebase()