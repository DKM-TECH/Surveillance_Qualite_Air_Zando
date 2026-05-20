from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.templating import Jinja2Templates
#from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import firebase_admin
from firebase_admin import credentials

from firebase_admin import credentials, firestore
import networkx as nx
import matplotlib.pyplot as plt
import os

app = FastAPI()
from fastapi.staticfiles import StaticFiles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)
templates = Jinja2Templates(directory="templates")
print("SERVER STARTED")
SEUILS = {
    "pm25": 35,
    "pm10": 75,
    "co2": 800,
    "nox": 100,
    "sox": 50,
    "nhx": 25
}

# -------------------------
# Firebase Configuration
# -------------------------
import os
import firebase_admin
from firebase_admin import credentials

import os
import firebase_admin
from firebase_admin import credentials, firestore

import os

def get_env(key):
    value = os.getenv(key)
    if value is None:
        raise Exception(f"❌ Variable manquante: {key}")
    return value

firebase_config = {
    "type": get_env("FIREBASE_TYPE"),
    "project_id": get_env("FIREBASE_PROJECT_ID"),
    "private_key_id": get_env("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": get_env("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": get_env("FIREBASE_CLIENT_EMAIL"),
    "client_id": get_env("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/" + get_env("FIREBASE_CLIENT_EMAIL")
}

cred = credentials.Certificate(firebase_config)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
def get_mesures():

    docs = db.collection("mesures").stream()

    data = []

    for doc in docs:

        d = doc.to_dict()

        # Sécurisation champs
        for col in [
            "pm25",
            "pm10",
            "co2",
            "nox",
            "sox",
            "nhx",
            "lat",
            "lon"
        ]:
            d[col] = float(d.get(col, 0))

        d["timestamp"] = str(d.get("timestamp", ""))

        data.append(d)

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)

@app.get("/")
#@app.get("/", response_class=HTMLResponse)
async def root():
    return {"message": "API OK"}
#def home(request: Request):
   #from fastapi import FastAPI
 #   return {"message": "API OK"}

   #templates.TemplateResponse(
    #"home.html",
    #{"request": request}
#)