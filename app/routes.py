from flask import Blueprint, render_template, redirect, url_for, request, jsonify
from .firebase import auth  # Make sure this imports from your firebase.py
import csv
import os
import subprocess
import sys

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return redirect(url_for('main.dashboard'))

@main.route('/run-attendance', methods=['POST'])
def run_attendance_script():
    python_executable = sys.executable
    try:
        print(f"🧠 Running attendance script using: {python_executable}")
        result = subprocess.run(
            [python_executable, 'app/attendance_script.py'],
            capture_output=True,
            text=True
        )
        print("✅ Script output:", result.stdout)
        print("❌ Script error (if any):", result.stderr)

        return jsonify({"status": "success", "output": result.stdout})
    except Exception as e:
        print("❌ Error running script:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@main.route('/login')
def login():
    return render_template('login.html')

@main.route('/verify-token', methods=['POST'])
def verify_token():
    id_token = request.json.get('idToken')
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        print(f"✅ Authenticated user: {email} (UID: {uid})")
        return jsonify({"status": "success", "uid": uid, "email": email})
    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 401


@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

