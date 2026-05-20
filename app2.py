from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import create_engine, text

app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SEUILS = {
    "pm25": 35,
    "pm10": 75,
    "co2": 800,
    "nox": 100,
    "sox": 50,
    "nhx": 25
}


# --------------------
# Connexion MySQL
# --------------------
engine = create_engine(
    "mysql+mysqlconnector://root:2024@127.0.0.1:3307/air_quality",
    echo=True,
    future=True
)

# ✅ Test connexion CORRECT
with engine.connect() as conn:
    res = conn.execute(text("SELECT 1")).scalar()
    print("✅ MySQL OK :", res)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "home.html",
        {}
    )

# --------------------
# Page Jauges
# --------------------
@app.get("/gauges", response_class=HTMLResponse)
def gauges(request: Request):

    df = pd.read_sql(
        """
        SELECT pm25, pm10, co2, nox, sox, nhx
        FROM mesures
        ORDER BY id DESC
        LIMIT 1
        """,
        engine
    )

    mesures = {k: 0.0 for k in SEUILS}
    depassements = {}
    critiques = []

    if not df.empty:
        mesures = {k: float(df.iloc[0][k]) for k in SEUILS}

        for pol, seuil in SEUILS.items():
            if mesures[pol] > seuil:
                depassements[pol] = mesures[pol]
            if mesures[pol] > seuil * 2:
                critiques.append(pol)

    # État global
    if len(critiques) >= 2:
        etat = "DANGEREUX"
        message = "⚠️ Niveau critique : évitez toute exposition prolongée."
    elif depassements:
        etat = "AIR POLLUÉ"
        message = "Qualité de l’air dégradée : port du masque recommandé."
    else:
        etat = "AIR SAIN"
        message = "Qualité de l’air normale."

    return templates.TemplateResponse(
        request,
        "gauges.html",
        {
            "mesures": mesures,
            "seuils": SEUILS,
            "etat": etat,
            "message": message
        }
    )


# --------------------
# Page Apriori
# --------------------
@app.get("/apriori", response_class=HTMLResponse)
def apriori_page(request: Request):
    # -------------------
    # Dernières mesures pour les gauges
    # -------------------
    df_latest = pd.read_sql(
        "SELECT pm25, pm10, co2, nox, sox, nhx FROM mesures ORDER BY id DESC LIMIT 1",
        engine
    )
    if not df_latest.empty:
        dernieres_mesures = df_latest.iloc[0].to_dict()
        # Convertir en float
        dernieres_mesures = {k: float(v) for k,v in dernieres_mesures.items()}
    else:
        dernieres_mesures = {k: 0.0 for k in SEUILS}

    # -------------------
    # Dataset pour Apriori
    # -------------------
    df = pd.read_sql("SELECT pm25, pm10, co2, nox, sox, nhx FROM mesures", engine)
    polluant = None
    table_dataset = []
    rules_html = []

    if not df.empty and len(df) >= 5:
        df_bin = df.gt(pd.Series(SEUILS)).astype(int)  # 1= dépassement, 0= sous seuil
        df_bin = df_bin.loc[:, df_bin.any()]

        table_dataset = df_bin.to_dict(orient='records')

        frequent_itemsets = apriori(df_bin, min_support=0.3, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            plot_apriori_network(rules)


            if not rules.empty:
                # Convertir frozenset en listes
                rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
                rules['consequents'] = rules['consequents'].apply(lambda x: list(x))

                # Garder uniquement les metrics souhaitées
                metrics_cols = ['antecedents','consequents','support','confidence','lift','leverage','conviction']
                rules_html = rules[metrics_cols].to_dict(orient='records')

                counts = rules['antecedents'].explode().value_counts()
                polluant = counts.idxmax()

    return templates.TemplateResponse(
        request,
        "apriori.html",
        {
            "dataset": table_dataset,
            "rules": rules_html,
            "polluant_influent": polluant,
            "dernieres_mesures": dernieres_mesures,  # ici corrigé
            "seuils": SEUILS
        }
    )

@app.get("/api/dernieres_mesures")
def api_dernieres_mesures():
    df = pd.read_sql("SELECT pm25, pm10, co2, nox, sox, nhx FROM mesures ORDER BY id DESC LIMIT 1", engine)
    if df.empty:
        return {k: 0.0 for k in SEUILS}
    return {k: float(df.iloc[0][k]) for k in SEUILS}

# --------------------
# Page Rapport
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from datetime import datetime
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
# --- Seuils d'alerte (exemple) ---
SEUILS = {
    "pm25": 25,
    "pm10": 50,
    "co2": 1000,
    "nox": 100,
    "sox": 100,
    "nhx": 50
}

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
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= value <= c_high:
            aqi = ((i_high - i_low)/(c_high - c_low))*(value - c_low) + i_low
            return round(aqi,2)
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
    df = pd.read_sql(
        "SELECT pm25, pm10, co2, nox, sox, nhx, lat, lon FROM mesures ORDER BY id DESC LIMIT 10",
        engine
    )

    if df.empty:
        return templates.TemplateResponse(
            "rapport.html",
            {
                "request": request,
                "mesures": {},
                "depassements": {},
                "status": "N/A",
                "position": None,
                "history": {},
                "aqi_global": None,
                "aqi_label": "N/A",
                "aqi_color": "#999999",
                "polluant_dominant": None,
                "aqi_parts": {}
            }
        )

    last_row = df.iloc[0]
    mesures = last_row[["pm25","pm10","co2","nox","sox","nhx"]].to_dict()
    history = {pol: df[pol].tolist()[::-1] for pol in ["pm25","pm10","co2","nox","sox","nhx"]}

    # --- Dépassements ---
    depassements = {k:v for k,v in mesures.items() if k in SEUILS and v>SEUILS[k]}
    status = "AIR POLUÉ" if depassements else "AIR SAIN"

    # --- Position ---
    position = None
    if pd.notnull(last_row["lat"]) and pd.notnull(last_row["lon"]):
        position = {"lat": float(last_row["lat"]), "lon": float(last_row["lon"])}

    # --- AQI multi-polluants ---
    aqi_parts = {}
    for pol, bpts in AQI_BREAKPOINTS.items():
        db_key = POLLUTANT_DB_MAP[pol]
        if db_key in mesures:
            val = mesures[db_key]
            aqi_val = compute_aqi_generic(val, bpts)
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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

@app.post("/capteur")
async def capteur_post(data: dict):
    try:
        pm25 = data.get("pm25", 0)
        pm10 = data.get("pm10", 0)
        co2 = data.get("co2", 0)
        nox = data.get("nox", 0)
        sox = data.get("sox", 0)
        nhx = data.get("nhx", 0)
        lat = data.get("lat", 0)
        lon = data.get("lon", 0)
        capteur_id = data.get("capteur_id", 0)

        query = text("""
            INSERT INTO mesures (capteur_id, pm25, pm10, co2, nox, sox, nhx, lat, lon)
            VALUES (:capteur_id, :pm25, :pm10, :co2, :nox, :sox, :nhx, :lat, :lon)
        """)

        with engine.connect() as conn:
            conn.execute(query, {
                "capteur_id": capteur_id,
                "pm25": pm25,
                "pm10": pm10,
                "co2": co2,
                "nox": nox,
                "sox": sox,
                "nhx": nhx,
                "lat": lat,
                "lon": lon
            })
            conn.commit()

        return JSONResponse({"status":"ok"})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/rapport/pdf")
def rapport_pdf():
    df = pd.read_sql(
        "SELECT pm25, pm10, co2, nox, sox, nhx, lat, lon FROM mesures ORDER BY id DESC LIMIT 10",
        engine
    )
    if df.empty:
        return {"error": "Aucune donnée disponible"}

    last = df.iloc[0]
    mesures = last[["pm25","pm10","co2","nox","sox","nhx"]].to_dict()

    # --- AQI multi-polluants ---
    aqi_parts = {}
    for pol, bpts in AQI_BREAKPOINTS.items():
        db_key = POLLUTANT_DB_MAP[pol]
        if db_key in mesures:
            val = mesures[db_key]
            aqi_val = compute_aqi_generic(val, bpts)
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

    # --- Création PDF ---
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = tmp_file.name
    tmp_file.close()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # --- TITRE ---
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "RAPPORT DE QUALITÉ DE L’AIR")
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, height-70, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

    # --- AQI GLOBAL ---
    c.setFillColor(HexColor(aqi_color))
    c.rect(50, height-150, width-100, 40, fill=1)
    c.setFillColorRGB(0,0,0)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height-135, f"AQI GLOBAL : {aqi_global} — {aqi_label}")

    # --- Polluant dominant ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height-190, f"Polluant dominant : {polluant_dominant}")

    # --- Tableau AQI ---
    y = height-230
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Polluant")
    c.drawString(200, y, "Concentration")
    c.drawString(360, y, "AQI")
    c.setFont("Helvetica", 11)
    y -= 20
    for pol, aqi in aqi_parts.items():
        db_key = POLLUTANT_DB_MAP[pol]
        val = mesures.get(db_key, 0)
        c.drawString(50, y, pol)
        c.drawString(200, y, str(round(val,2)))
        c.drawString(360, y, str(aqi))
        y -= 18

    # --- Position ---
    if pd.notnull(last["lat"]) and pd.notnull(last["lon"]):
        y -= 20
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Position du capteur")
        y -= 15
        c.setFont("Helvetica", 11)
        c.drawString(50, y, f"Latitude : {last['lat']} — Longitude : {last['lon']}")

    # --- Conclusion ---
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Conclusion sanitaire")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"La qualité de l’air est classée comme {aqi_label}, principalement influencée par {polluant_dominant}.")

    # --- Signature ---
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "Système de surveillance de la qualité de l’air — Donatien KADIMA")

    c.showPage()
    c.save()

    return FileResponse(pdf_path, media_type="application/pdf", filename="rapport_qualite_air.pdf")

import networkx as nx
import matplotlib.pyplot as plt
def plot_apriori_network(rules):
    if rules.empty:
        return

    G = nx.DiGraph()

    for _, row in rules.iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                G.add_edge(
                    a, c,
                    weight=row['confidence'],
                    lift=row['lift']
                )

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)

    edge_widths = [G[u][v]['weight'] * 4 for u, v in G.edges()]
    edge_colors = [G[u][v]['lift'] for u, v in G.edges()]

    nx.draw(
        G, pos,
        ax=ax,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=11,
        edge_color=edge_colors,
        width=edge_widths,
        edge_cmap=plt.cm.viridis,
        arrowsize=20
    )

    ax.set_title("Diagramme d’association des polluants (Apriori)")

    # --- Colorbar CORRECTEMENT attachée ---
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
    )
    sm.set_array([])

    fig.colorbar(sm, ax=ax, label="Lift")

    plt.tight_layout()
    plt.savefig("static/apriori_network.png")
    plt.close()

