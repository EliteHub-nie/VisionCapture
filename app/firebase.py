import firebase_admin
from firebase_admin import credentials, auth, firestore

# Initialize Firebase Admin with your private key
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()