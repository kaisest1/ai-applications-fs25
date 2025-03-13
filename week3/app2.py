# app.py
import gradio as gr
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Lade das Modell (stelle sicher, dass es vorher gespeichert wurde)
model_filename = "apartment_random_forest_regressor.pkl"
with open(model_filename, mode="rb") as f:
    model = pickle.load(f)

# Vorhersagefunktion
def predict(bfs_number, rooms, area, neubau):
    # Abrufen der Werte basierend auf der BFS-Nummer
    row = df[df['bfs_number'] == bfs_number]
    
    if row.empty:
        return "Fehler: BFS-Nummer nicht gefunden."
    
    # Werte aus der Zeile übernehmen
    pop = row['pop'].values[0]
    pop_dens = row['pop_dens'].values[0]
    frg_pct = row['frg_pct'].values[0]
    emp = row['emp'].values[0]
    tax_income = row['tax_income'].values[0]

    # Eingabewerte in ein DataFrame umwandeln
    input_data = pd.DataFrame([[rooms, area, pop, pop_dens, frg_pct, emp, tax_income, neubau]],
                              columns=['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 'neubau'])
    
    # Vorhersage mit dem trainierten Modell
    prediction = model.predict(input_data)[0]  # Gibt den Preis als einzelne Zahl zurück
    return f"{round(prediction, 2)} CHF"  # Vorhersage auf 2 Dezimalstellen gerundet zurückgeben

# Gradio-Interface erstellen
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="BFS Nummer"),  # BFS-Nummer als Zahl
        gr.Number(label="Zimmer"),  # Anzahl der Zimmer
        gr.Number(label="Fläche (m²)"),  # Fläche des Apartments
        gr.Checkbox(label="Neubau"),  # Checkbox für Neubau
    ],
    outputs="text",
    examples=[
        [51, 2, 50, False],  # Beispiel 1: Neubau = Nein
        [261, 3, 75, True],   # Beispiel 2: Neubau = Ja
        [118, 4, 100, False], # Beispiel 3: Neubau = Nein
    ],
    title="Apartment Preis Vorhersage",
    description="Geben Sie die BFS-Nummer, Anzahl der Zimmer, Fläche und Neubau-Status ein, um den geschätzten Preis des Apartments zu erhalten."
)

# Starte die Gradio-Anwendung
demo.launch()
