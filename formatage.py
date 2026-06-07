import requests

# ✅ IMPORTANT: ajouter .json
FIREBASE_URL = "https://air-zando-default-rtdb.firebaseio.com/.json"

response = requests.delete(FIREBASE_URL)

print("Status code:", response.status_code)
print("Response:", response.text)