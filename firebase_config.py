import firebase_admin
from firebase_admin import credentials, auth

# Load Firebase Admin SDK
cred = credentials.Certificate("E:\Techathon_672\cancer-cell-detection-1fa61-firebase-adminsdk-fbsvc-0c2de11f1a.json")  # Replace with actual path
firebase_admin.initialize_app(cred)

def register_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user.uid
    except Exception as e:
        return str(e)

def login_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except Exception as e:
        return None
