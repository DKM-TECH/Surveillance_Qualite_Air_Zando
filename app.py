from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import mysql.connector
from mysql.connector import Error

app = FastAPI()

# Connexion MySQL (fonction pour reconnexion automatique)
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="2024",
            database="air_quality",
            port=3307
        )
        return connection
    except Error as e:
        print(f"Erreur connexion MySQL: {e}")
        return None

db = get_db_connection()
cursor = db.cursor() if db else None

# Templates HTML
templates = Jinja2Templates(directory="templates")

# Endpoint pour recevoir les données PM
@app.post("/pm")
def receive_pm(data: dict):
    global db, cursor  # <-- déclarer global tout en haut de la fonction

    if not cursor or not db:
        return JSONResponse(status_code=500, content={"error": "Pas de connexion MySQL"})

    try:
        pm25 = float(data.get("pm25"))
        pm10 = float(data.get("pm10"))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "Données PM invalides"})

    try:
        # Reconnexion automatique si nécessaire
        if not db.is_connected():
            db = get_db_connection()
            cursor = db.cursor()

        cursor.execute(
            "INSERT INTO pm_measurements (pm25, pm10) VALUES (%s, %s)",
            (pm25, pm10)
        )
        db.commit()
    except Error as e:
        return JSONResponse(status_code=500, content={"error": f"MySQL Error: {e}"})

    return {"status": "ok"}

# Page web pour afficher les mesures
@app.get("/", response_class=HTMLResponse)
def read_pm(request: Request):
    if not cursor:
        return HTMLResponse(content="<h1>Erreur: Pas de connexion MySQL</h1>", status_code=500)

    try:
        cursor.execute("SELECT pm25, pm10, measure_time FROM pm_measurements ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
    except Error as e:
        return HTMLResponse(content=f"<h1>Erreur MySQL: {e}</h1>", status_code=500)

    return templates.TemplateResponse("index.html", {"request": request, "measurements": rows})
