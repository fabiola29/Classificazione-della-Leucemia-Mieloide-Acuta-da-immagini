import os
from prettytable import PrettyTable

# --- CONFIGURAZIONE PERCORSI ---
# La cartella principale deve chiamarsi 'data' ed essere nella stessa posizione dello script
PATH_DATA = './data'

# Le 5 classi specifiche del dataset AML-MLL
CLASSI = [
    'CBFB_MYH11',      # AML con inv(16)
    'NPM1',            # AML con mutazione NPM1
    'PML_RARA',        # Leucemia Promielocitica Acuta
    'RUNX1_RUNX1T1',   # AML con t(8;21)
    'CONTROL'          # Donatori Sani
]

def analizza_dataset():
    # Creazione tabella: Classe, Numero di Pazienti, Totale Immagini
    tabella = PrettyTable(['Classe', 'Pazienti (ID)', 'Totale Immagini'])
    tot_immagini = 0
    tot_pazienti = 0

    print(f"\n📊 ANALISI DATASET AML: {PATH_DATA}")
    
    if not os.path.exists(PATH_DATA):
        print(f"ERRORE: La cartella '{PATH_DATA}' non esiste. Assicurati che lo script sia accanto alla cartella data.")
        return

    for cls in CLASSI:
        class_path = os.path.join(PATH_DATA, cls)
        
        if not os.path.exists(class_path):
            tabella.add_row([cls, "NON TROVATA", 0])
            continue

        # Conta le sottocartelle dei pazienti (es. AQK, BJK)
        pazienti = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        num_pazienti = len(pazienti)
        
        # Conta tutte le immagini (.tif, .jpg, .png) ricorsivamente in ogni cartella paziente
        num_immagini = 0
        for root, dirs, files in os.walk(class_path):
            num_immagini += len([f for f in files if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])

        tabella.add_row([cls, num_pazienti, num_immagini])
        
        tot_pazienti += num_pazienti
        tot_immagini += num_immagini

    # Stampa i risultati
    print(tabella)
    print(f"Pazienti totali nel dataset: {tot_pazienti}")
    print(f"Immagini totali (uniche):     {tot_immagini}")

if __name__ == "__main__":
    analizza_dataset()
