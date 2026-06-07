import os
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import firebase_admin

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from supabase import create_client
import os

from mlxtend.frequent_patterns import apriori, association_rules

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)
print("BASE_DIR =", BASE_DIR)
print("TEMPLATES_DIR =", TEMPLATES_DIR)
print("EXISTS =", os.path.exists(TEMPLATES_DIR))
print("FILES =", os.listdir(TEMPLATES_DIR) if os.path.exists(TEMPLATES_DIR) else "NO DIR")
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
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_env(key):
    value = os.getenv(key)
    if value is None:
        raise Exception(f"❌ Variable manquante: {key}")
    return value

from collections import defaultdict

def clean_df(df):
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def convert_timestamp(ts):

    if ts is None or pd.isnull(ts):
        return pd.NaT

    try:
        # si déjà numérique (int/float)
        if isinstance(ts, (int, float)):
            ts = int(ts)

            # heuristique ms vs s
            if ts > 10**12:
                return pd.to_datetime(ts, unit="ms")
            else:
                return pd.to_datetime(ts, unit="s")

        ts = str(ts).strip()

        # string numérique
        if ts.isdigit():
            ts_int = int(ts)

            if len(ts) >= 13:
                return pd.to_datetime(ts_int, unit="ms")
            else:
                return pd.to_datetime(ts_int, unit="s")

        # fallback texte
        return pd.to_datetime(ts, errors="coerce")

    except Exception:
        return pd.NaT

def parse_timestamp(ts):
    try:
        ts = int(ts)
        return pd.to_datetime(ts, unit="s")
    except:
        return pd.NaT

def get_history_mesures():

    try:
        res = supabase.table("air_history") \
            .select("*") \
            .order("timestamp", desc=False) \
            .execute()

        if not res.data:
            return pd.DataFrame()

        df = pd.DataFrame(res.data)

        # sécurité types
        for col in ["pm25","pm10","co2","nox","sox","nhx"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        return df

    except Exception as e:
        print("SUPABASE ERROR:", e)
        return pd.DataFrame()

def get_mesures():

    try:
        res = supabase.table("air_history") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()

        if not res.data:
            return pd.DataFrame()

        data = res.data[0]

        return pd.DataFrame(res.data)

    except Exception as e:
        print("SUPABASE ERROR:", e)
        return pd.DataFrame()
   

from functools import lru_cache
import time

CACHE = {"data": None, "ts": 0}

def get_history_cached(ttl=10):
    global CACHE
    if time.time() - CACHE["ts"] < ttl:
        return CACHE["data"]

    data = get_history_mesures()
    CACHE["data"] = data
    CACHE["ts"] = time.time()
    return data

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

   return templates.TemplateResponse(
    "home.html",
    {"request": request}
)

@app.get("/api/live")
def api_live():
    df = get_mesures()

    if df.empty:
        return {}

    df = df.sort_values(by="timestamp", ascending=False)
    row = df.iloc[0]

    return {
        "pm25": float(row.get("pm25", 0)),
        "pm10": float(row.get("pm10", 0)),
        "co2": float(row.get("co2", 0)),
        "nox": float(row.get("nox", 0)),
        "sox": float(row.get("sox", 0)),
        "nhx": float(row.get("nhx", 0))
    }


@app.get("/test")
def test():
    return {
        "message": "API OK"
    }

@app.get("/gauges", response_class=HTMLResponse)
def gauges(request: Request):

    df = get_mesures()

    mesures = {
        "PM2.5": 0,
        "PM10": 0,
        "CO2": 0,
        "NOx": 0,
        "SOx": 0,
        "NHx": 0
    }

    seuils = {
        "PM2.5": 50,
        "PM10": 100,
        "CO2": 1000,
        "NOx": 200,
        "SOx": 150,
        "NHx": 100
    }

    etat = "AIR INCONNU"
    message = "Pas de données"

    if df is not None and not df.empty and "timestamp" in df.columns:

        try:

            df["timestamp"] = df["timestamp"].apply(convert_timestamp)
            df = df.sort_values(by="timestamp", ascending=False)
            row = df.iloc[0]

            correspondance = {
                "PM2.5": "pm25",
                "PM10": "pm10",
                "CO2": "co2",
                "NOx": "nox",
                "SOx": "sox",
                "NHx": "nhx"
            }

            for affichage, colonne in correspondance.items():

                if colonne in row:
                    mesures[affichage] = float(
                        row.get(colonne, 0)
                    )

            depassements = [

                pol for pol in mesures

                if mesures[pol] > seuils[pol]
            ]

            if len(depassements) >= 3:

                etat = "DANGEREUX"
                message = "⚠️ Pollution critique"

            elif len(depassements) > 0:

                etat = "AIR POLLUÉ"
                message = "Air dégradé"

            else:

                etat = "AIR SAIN"
                message = "Qualité normale"

        except Exception as e:

            print("GAUGES ERROR:", e)

    return templates.TemplateResponse(

        "gauges.html",

        {
            "request": request,
            "mesures": mesures,
            "seuils": seuils,
            "etat": etat,
            "message": message
        }
    )
#VOIR LES DONNEES DATASET (DONNEES BRUTES)

@app.get("/dataset", response_class=HTMLResponse)
def dataset_page(request: Request):

    df = get_history_mesures()

    if df is None:
        df = pd.DataFrame()

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)

        df = df.sort_values(
            "timestamp",
            ascending=False
        )

    stats = {}

    pollutants = [
        "pm25",
        "pm10",
        "co2",
        "nox",
        "sox",
        "nhx"
    ]

    for col in pollutants:

        if col in df.columns:

            stats[col] = {
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "avg": round(float(df[col].mean()), 2)
            }

    return templates.TemplateResponse(
        
        "dataset.html",
        {
            "request": request,
            "rows": df.to_dict("records"),
            "total": len(df),
            "stats": stats
        }
    )

@app.get("/api/dataset")
def dataset_api():

    df = get_history_mesures()

    if df is None or df.empty:
        return {"rows": 0, "data": []}

    df = df.sort_values("timestamp", ascending=True)

    # limite sécurité (IMPORTANT)
    df = df.tail(500)

    return {
        "rows": len(df),
        "data": df.to_dict(orient="records")
    }

from functools import lru_cache
@lru_cache(maxsize=1)
def cached_data():
    return get_history_mesures()

@app.get("/export-dataset")
def export_dataset():

    df = get_history_mesures()

    filename = "dataset_air.csv"

    df.to_csv(filename, index=False)

    return FileResponse(
        path=filename,
        filename=filename,
        media_type="text/csv"
    )

## --------------------
# Page Apriori
# --------------------
@app.get("/apriori", response_class=HTMLResponse)
def apriori_page(request: Request):

    df = get_history_mesures()

    # =========================
    # INIT VARIABLES
    # =========================
    dernieres_mesures = {k: 0.0 for k in SEUILS}
    polluant = None
    table_dataset = []
    rules_html = []

    # =========================
    # VALIDATION DATAFRAME
    # =========================
    if df is None or df.empty:
        return templates.TemplateResponse(
            "apriori.html",
            {
                "request": request,
                "dataset": [],
                "rules": [],
                "polluant_influent": None,
                "dernieres_mesures": dernieres_mesures,
                "seuils": SEUILS
            }
        )

    # =========================
    # TRI + DERNIÈRE MESURE
    # =========================
    df = df.sort_values(by="timestamp", ascending=False)

    latest = df.iloc[0]

    for k in SEUILS:
        try:
            dernieres_mesures[k] = float(latest.get(k, 0))
        except:
            dernieres_mesures[k] = 0.0

    # =========================
    # PRÉPARATION APRIORI
    # =========================
    if len(df) < 5:
        return templates.TemplateResponse(
            "apriori.html",
            {
                "request": request,
                "dataset": [],
                "rules": [],
                "polluant_influent": None,
                "dernieres_mesures": dernieres_mesures,
                "seuils": SEUILS
            }
        )

    # =========================
    # EXTRACTION POLLUANTS
    # =========================
    cols = list(SEUILS.keys())
    df_polluants = df[cols].fillna(0)

    # =========================
    # BINARISATION
    # =========================
    df_bin = pd.DataFrame()

    for col, seuil in SEUILS.items():

        df_bin[f"{col}_ELEVE"] = (df_polluants[col] > seuil)
        df_bin[f"{col}_TRES_ELEVE"] = (df_polluants[col] > seuil * 1.5)

    df_bin = df_bin.astype(int)

    # suppression colonnes vides
    df_bin = df_bin.loc[:, df_bin.sum(axis=0) > 0]

    # dataset pour frontend
    table_dataset = df_bin.to_dict(orient="records")[:50]

    print("DF BIN SHAPE:", df_bin.shape)

    # =========================
    # APRIORI
    # =========================
    try:

        if not df_bin.empty:

            frequent_itemsets = apriori(
                df_bin,
                min_support=0.3,
                use_colnames=True,
                max_len=3
            )

            if not frequent_itemsets.empty:

                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=0.8
                )

                if not rules.empty:

                    plot_apriori_network(rules)

                    # conversion sets -> list
                    rules["antecedents"] = rules["antecedents"].apply(list)
                    rules["consequents"] = rules["consequents"].apply(list)

                    # arrondir valeurs
                    for col in ["support", "confidence", "lift", "leverage", "conviction"]:
                        rules[col] = rules[col].round(3)

                    # tri
                    rules = rules.sort_values(by="lift", ascending=False).head(50)

                    metrics_cols = [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift",
                        "leverage",
                        "conviction"
                    ]

                    rules_html = rules[metrics_cols].to_dict(orient="records")

                    # polluant dominant
                    counts = rules["antecedents"].explode().value_counts()
                    if not counts.empty:
                        polluant = counts.idxmax()

    except Exception as e:
        print("Apriori error:", e)

    # =========================
    # DEBUG FINAL
    # =========================
    print("Dataset size =", len(table_dataset))
    print("Rules size =", len(rules_html))

    # =========================
    # TEMPLATE
    # =========================
    return templates.TemplateResponse(
        "apriori.html",
        {
            "request": request,
            "dataset": table_dataset,
            "rules": rules_html,
            "polluant_influent": polluant,
            "dernieres_mesures": dernieres_mesures,
            "seuils": SEUILS
        }
    )


def plot_apriori_network(rules):
    if rules.empty:
        return

    G = nx.DiGraph()

    # normalisation lift
    lift_min = rules["lift"].min()
    lift_max = rules["lift"].max()

    def normalize(x):
        if lift_max == lift_min:
            return 0
        return (x - lift_min) / (lift_max - lift_min)

    for _, row in rules.iterrows():

        confidence = row["confidence"]
        support = row["support"]
        lift = row["lift"]
        lift_norm = normalize(lift)

        # score global pondéré
        score = (
            0.4 * confidence +
            0.3 * lift_norm +
            0.3 * support
        )

        for a in row["antecedents"]:
            for c in row["consequents"]:
                G.add_edge(
                    a,
                    c,
                    weight=score,
                    confidence=confidence,
                    lift=lift
                )

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    
    node_weights = {}

    for _, row in rules.iterrows():
        for a in row["antecedents"]:
            node_weights[a] = node_weights.get(a, 0) + 1
        for c in row["consequents"]:
            node_weights[c] = node_weights.get(c, 0) + 1

    node_size = [node_weights.get(node, 1) * 800 for node in G.nodes()]
    # épaisseur = score
    edge_widths = [
        G[u][v]["weight"] * 6
        for u, v in G.edges()
    ]

    # couleur = lift (toujours interprétable scientifiquement)
    edge_colors = [
        G[u][v]["lift"]
        for u, v in G.edges()
    ]

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=11,
        edge_color=edge_colors,
        width=edge_widths,
        edge_cmap=plt.cm.viridis,
        arrowsize=18
    )

    ax.set_title(
        "Réseau pondéré des associations (Apriori)"
    )

    # colorbar lift
    if not edge_colors:
        return
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(
            vmin=min(edge_colors),
            vmax=max(edge_colors)
        )
    )
    sm.set_array([])

    fig.colorbar(sm, ax=ax, label="Lift")
    
    plt.tight_layout()

    plt.savefig(
    os.path.join(STATIC_DIR, "apriori_network.png")
    )
    #plt.savefig("static/apriori_network.png")
    plt.close()

 
@app.get("/api/historique")
def historique():

    df = get_history_mesures()

    if df.empty:
        return {
            "journalier": [],
            "mensuel": []
        }

    # Conversion robuste datetime
    df["timestamp"] = df["timestamp"].apply(convert_timestamp)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Suppression valeurs invalides
    df = df.dropna(subset=["timestamp"])

    # Vérification sécurité
    if df.empty:
        return {
            "journalier": [],
            "mensuel": []
        }

    # -------------------------
    # JOURNALIER
    # -------------------------

    df["jour"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    daily = df.groupby("jour")[
        ["pm25","pm10","co2","nox","sox","nhx"]
    ].mean().reset_index()

    # -------------------------
    # MENSUEL
    # -------------------------

    df["mois"] = df["timestamp"].dt.strftime("%Y-%m")

    monthly = df.groupby("mois")[
        ["pm25","pm10","co2","nox","sox","nhx"]
    ].mean().reset_index()
    
    daily = daily.sort_values("jour")
    monthly = monthly.sort_values("mois")

    return {
        "journalier": daily.to_dict(orient="records"),
        "mensuel": monthly.to_dict(orient="records")
    }

# Page Rapport

from datetime import datetime
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


# --- AQI Multi-pollutants OFFICIEL ---
AQI_BREAKPOINTS = {
    "PM2.5": [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300)],
    "PM10": [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300)],
    "NO2": [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300)],
    "SO2": [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300)]
}

POLLUTANT_DB_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "nox",
    "SO2": "sox"
}

def compute_aqi_generic(value, breakpoints):

    if value is None:
        return None

    for c_low, c_high, i_low, i_high in breakpoints:

        if c_low <= value <= c_high:

            aqi = (
                ((i_high - i_low) / (c_high - c_low))
                * (value - c_low)
            ) + i_low

            return round(aqi, 2)

    # Valeur hors plage
    if value > breakpoints[-1][1]:
        return 350

    return None

def aqi_category(aqi):
    if aqi is None:
        return "N/A", "#999999"
    if aqi <= 50:
        return "BON", "#00e400"
    elif aqi <= 100:
        return "MODÉRÉ", "#ffff00"
    elif aqi <= 150:
        return "MAUVAIS", "#ff7e00"
    elif aqi <= 200:
        return "TRÈS MAUVAIS", "#ff0000"
    else:
        return "DANGEREUX", "#8f3f97"

@app.get("/rapport", response_class=HTMLResponse)
def rapport(request: Request):
    df = get_mesures()

    if not df.empty:
        df = df.sort_values(by="timestamp", ascending=False).head(10).reset_index(drop=True)

    if df.empty:
        return templates.TemplateResponse(
        "rapport.html",
        {
            "request": request,
            "mesures": {},
            "depassements": {},
            "status": "Aucune donnée",
            "position": None,
            "history": {},
            "aqi_global": 0,
            "aqi_label": "N/A",
            "aqi_color": "#999999",
            "polluant_dominant": None,
            "aqi_parts": {}
        }
    )

    last_row = df.iloc[0]
    mesures = last_row[["pm25","pm10","co2","nox","sox","nhx"]].to_dict()
    history = {
    pol: df[pol].fillna(0).tolist()[::-1]
    for pol in ["pm25","pm10","co2","nox","sox","nhx"]
    if pol in df.columns
    }
    # --- Dépassements ---
    depassements = {k:v for k,v in mesures.items() if k in SEUILS and v>SEUILS[k]}
    status = "AIR POLUÉ" if depassements else "AIR SAIN"

    # --- Position ---
    position = None
    if "lat" in last_row and "lon" in last_row:
        if pd.notnull(last_row["lat"]) and pd.notnull(last_row["lon"]):
            position = {
            "lat": float(last_row["lat"]),
            "lon": float(last_row["lon"])
        }
   # if pd.notnull(last_row["lat"]) and pd.notnull(last_row["lon"]):
    #    position = {"lat": float(last_row["lat"]), "lon": float(last_row["lon"])}

    # --- AQI multi-polluants ---
    aqi_parts = {}
    for pol, bpts in AQI_BREAKPOINTS.items():
        db_key = POLLUTANT_DB_MAP[pol]
        if db_key in mesures:
            val = mesures.get(db_key)
            if val is not None:
                aqi_val = compute_aqi_generic(float(val), bpts)
                if aqi_val is not None:
                    aqi_parts[pol] = aqi_val

     
    if aqi_parts:
        aqi_global = max(aqi_parts.values())
        polluant_dominant = max(aqi_parts, key=aqi_parts.get)
        aqi_label, aqi_color = aqi_category(aqi_global)
    else:
        aqi_global = None
        polluant_dominant = None
        aqi_label, aqi_color = "N/A", "#999999"

    return templates.TemplateResponse(
    "rapport.html",
    {
        "request": request,
        "mesures": mesures,
        "depassements": depassements,
        "status": status,
        "position": position,
        "history": history,
        "aqi_global": aqi_global,
        "aqi_label": aqi_label,
        "aqi_color": aqi_color,
        "polluant_dominant": polluant_dominant,
        "aqi_parts": aqi_parts
    }
 )
@app.get("/api/dernieres_mesures")
def dernieres_mesures():

    df = get_mesures()

    if df.empty:
        return {}

    df = df.sort_values(
        by="timestamp",
        ascending=False
    )

    row = df.iloc[0]

    return {
        k: float(row.get(k, 0))
        for k in SEUILS
    }

#PREDICTION AVEC MACHINE LEARNING

@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(request: Request):
    return templates.TemplateResponse(
        "prediction.html",
        {"request": request}
    )
#import numpy as np

#import numpy as np

# scalers doivent être chargés globalement
# scaler_x = joblib.load(...)
# scaler_y = joblib.load(...)
import os
import joblib

model = None

MODEL_PATH = "air_xgb_model.pkl"
import xgboost as xgb
import traceback
@app.get("/predict")
def predict():

    global model
    print("PREDICT CALLED")

    try:
        print("===== PREDICT START =====")

        # =========================
        # LOAD MODEL
        # =========================
        if model is None:
            try:
                model = joblib.load("air_xgb_model.pkl")
                print("✅ Modèle chargé OK")
                print("TYPE =", type(model))
            except Exception as e:
                return {
                    "error": "model_load_failed",
                    "message": str(e)
                }

        print("MODEL OK")

        # =========================
        # LOAD DATA
        # =========================
        df = get_history_mesures().tail(100)

        if df is None or df.empty:
            return {
                "error": "no_data"
            }

        print("DF SHAPE =", df.shape)
        print(df.columns.tolist())

        # =========================
        # CLEAN DATA
        # =========================
        df["timestamp"] = df["timestamp"].apply(convert_timestamp)
        df = df.sort_values("timestamp")

        cols = [
            "pm25", "pm10", "co2", "nox", "sox", "nhx"
        ]

        # garder seulement colonnes existantes
        cols = [c for c in cols if c in df.columns]

        # convertir en numérique
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=cols)

        WINDOW = 10

        if len(df) < WINDOW:
            return {
                "error": "not_enough_data",
                "rows": len(df)
            }

        # =========================
        # BUILD INPUT
        # =========================

        X = df[cols].tail(WINDOW).values
        X = X.reshape(1, -1)

        print("X SHAPE =", X.shape)

        # =========================
        # PREDICT
        # =========================
        pred = model.predict(X)[0]

        def safe(v):
            try:
                return max(0.0, float(v))
            except:
                return 0.0

        prediction = {
            "pm25": safe(pred[0]),
            "pm10": safe(pred[1]),
            "co2": safe(pred[2]),
            "nox": safe(pred[3]),
            "sox": safe(pred[4]),
            "nhx": safe(pred[5])
        }

        last = df.iloc[-1]

        current = {
            "pm25": safe(last["pm25"]),
            "pm10": safe(last["pm10"]),
            "co2": safe(last["co2"]),
            "nox": safe(last["nox"]),
            "sox": safe(last["sox"]),
            "nhx": safe(last["nhx"])
        }

        return {
            "current": current,
            "prediction": prediction
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())

        return {
            "error": "runtime_error",
            "message": str(e)
        }

            
@app.get("/api/realtime")
def realtime():

    try:
        print("CALLED REALTIME")
        df = get_history_mesures().tail(100)

        if df is None or df.empty:
            return {"error": "no data"}

        df = df.tail(100)

        required = ["timestamp","pm25","pm10","co2","nox","sox","nhx"]

        missing = [c for c in required if c not in df.columns]
        if missing:
            return {"error": "missing columns", "message": missing}

        df = df.fillna(0)

        return {
            "labels": df["timestamp"].astype(str).tolist(),
            "pm25": df["pm25"].tolist(),
            "pm10": df["pm10"].tolist(),
            "co2": df["co2"].tolist(),
            "nox": df["nox"].tolist(),
            "sox": df["sox"].tolist(),
            "nhx": df["nhx"].tolist()
        }

    except Exception as e:
        return {"error": str(e)}
    
def pollutant_score(value, limit):

    if limit <= 0:
        return 0

    value = max(0, value)

    ratio = value / limit

    if ratio <= 1:
        return ratio * 50

    elif ratio <= 2:
        return 50 + (ratio - 1) * 50

    elif ratio <= 3:
        return 100 + (ratio - 2) * 50

    elif ratio <= 4:
        return 150 + (ratio - 3) * 50

    else:
        return 300
    
def compute_global_aqi(data):
    OMS_SEUILS = {
        "pm25": 15,
        "pm10": 45,
        "co2": 1000,
        "nox": 25,
        "sox": 40,
        "nhx": 10
    }

    scores = {}

    for pol, limit in OMS_SEUILS.items():
        if pol in data:
            scores[pol] = pollutant_score(data[pol], limit)

    return max(scores.values()), scores

def interpret_aqi(aqi):
    if aqi <= 50:
        return "BON", "Air sain 😊"
    elif aqi <= 100:
        return "MODÉRÉ", "Acceptable mais prudence"
    elif aqi <= 150:
        return "MAUVAIS", "Masque conseillé 😷"
    elif aqi <= 200:
        return "DANGEREUX", "Masque obligatoire ⚠️"
    else:
        return "TRÈS DANGEREUX", "Éviter toute sortie 🚨"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)