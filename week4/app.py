import gradio as gr
import torch
import numpy as np
import pandas as pd

# Das Modell
class RegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Lade das Modell
def load_model():
    model = RegressionModel(10)  # Da wir 10 Eingabewerte haben (7 aus den Daten + 3 manuell)
    model.load_state_dict(torch.load("regression_model.pth"))
    model.eval()
    return model

# Lade die CSV-Daten
df = pd.read_csv('bfs_municipality_and_tax_data.csv', sep=',', encoding='utf-8')
df['tax_income'] = df['tax_income'].str.replace("'", "").astype(float)

locations = {
    "Zürich": 261,
    "Kloten": 62,
    "Uster": 198,
    "Illnau-Effretikon": 296,
    "Feuerthalen": 27,
    "Pfäffikon": 177,
    "Ottenbach": 11,
    "Dübendorf": 191,
    "Richterswil": 138,
    "Maur": 195,
    "Embrach": 56,
    "Bülach": 53,
    "Winterthur": 230,
    "Oetwil am See": 157,
    "Russikon": 178,
    "Obfelden": 10,
    "Wald (ZH)": 120,
    "Niederweningen": 91,
    "Dällikon": 84,
    "Buchs (ZH)": 83,
    "Rüti (ZH)": 118,
    "Hittnau": 173,
    "Bassersdorf": 52,
    "Glattfelden": 58,
    "Opfikon": 66,
    "Hinwil": 117,
    "Regensberg": 95,
    "Langnau am Albis": 136,
    "Dietikon": 243,
    "Erlenbach (ZH)": 151,
    "Kappel am Albis": 6,
    "Stäfa": 158,
    "Zell (ZH)": 231,
    "Turbenthal": 228,
    "Oberglatt": 92,
    "Winkel": 72,
    "Volketswil": 199,
    "Kilchberg (ZH)": 135,
    "Wetzikon (ZH)": 121,
    "Zumikon": 160,
    "Weisslingen": 180,
    "Elsau": 219,
    "Hettlingen": 221,
    "Rüschlikon": 139,
    "Stallikon": 13,
    "Dielsdorf": 86,
    "Wallisellen": 69,
    "Dietlikon": 54,
    "Meilen": 156,
    "Wangen-Brüttisellen": 200,
    "Flaach": 28,
    "Regensdorf": 96,
    "Niederhasli": 90,
    "Bauma": 297,
    "Aesch (ZH)": 241,
    "Schlieren": 247,
    "Dürnten": 113,
    "Unterengstringen": 249,
    "Gossau (ZH)": 115,
    "Oberengstringen": 245,
    "Schleinikon": 98,
    "Aeugst am Albis": 1,
    "Rheinau": 38,
    "Höri": 60,
    "Rickenbach (ZH)": 225,
    "Rafz": 67,
    "Adliswil": 131,
    "Zollikon": 161,
    "Urdorf": 250,
    "Hombrechtikon": 153,
    "Birmensdorf (ZH)": 242,
    "Fehraltorf": 172,
    "Weiach": 102,
    "Männedorf": 155,
    "Küsnacht (ZH)": 154,
    "Hausen am Albis": 4,
    "Hochfelden": 59,
    "Fällanden": 193,
    "Greifensee": 194,
    "Mönchaltorf": 196,
    "Dägerlen": 214,
    "Thalheim an der Thur": 39,
    "Uetikon am See": 159,
    "Seuzach": 227,
    "Uitikon": 248,
    "Affoltern am Albis": 2,
    "Geroldswil": 244,
    "Niederglatt": 89,
    "Thalwil": 141,
    "Rorbas": 68,
    "Pfungen": 224,
    "Weiningen (ZH)": 251,
    "Bubikon": 112,
    "Neftenbach": 223,
    "Mettmenstetten": 9,
    "Otelfingen": 94,
    "Flurlingen": 29,
    "Stadel": 100,
    "Grüningen": 116,
    "Henggart": 31,
    "Dachsen": 25,
    "Bonstetten": 3,
    "Bachenbülach": 51,
    "Horgen": 295
}

# Die Vorhersagefunktion
def predict_price(rooms, area, luxurious, temporary, furnished, town):
    # Holen der bfs_number
    bfs_number = locations[town]
    
    # Filtern der Daten für die gewählte Stadt
    town_data = df[df['bfs_number'] == bfs_number].iloc[0]
    
    # Extrahiere die benötigten Werte aus der Stadt
    pop = town_data['pop']
    popdens = town_data['pop_dens']
    frc_pct = town_data['frg_pct']
    emp = town_data['emp']
    tex_income = town_data['tax_income']

    # Eingabewerte (manuell vom Benutzer und automatisch geladen)
    input_features = np.array([rooms, area, pop, popdens, frc_pct, emp, tex_income, luxurious, temporary, furnished])
    input_features = input_features.reshape(1, -1)

    # Umwandeln der Eingabewerte in Tensoren
    input_tensor = torch.tensor(input_features, dtype=torch.float32)

    # Lade das Modell
    model = load_model()

    # Berechne die Vorhersage
    with torch.no_grad():
        prediction = model(input_tensor)

    # Rückgabe der Vorhersage (aufgerundet)
    return np.round(prediction.item(), 0)

# Gradio-Interface erstellen
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Rooms"), 
        gr.Number(label="Area (in m²)"), 
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Temporary"),
        gr.Checkbox(label="Furnished"),
        gr.Dropdown(choices=list(locations.keys()), label="Town")
    ],
    outputs=gr.Number(label="Predicted Price")
)

iface.launch()
