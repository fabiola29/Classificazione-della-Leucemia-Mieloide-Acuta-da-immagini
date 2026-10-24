import os
from prettytable import PrettyTable

# --- CONFIGURAZIONE PERCORSO ---
# Percorso base dove si trovano 'Original' e 'Segmented'
BASE_PATH = "./dataset"

# Le 5 classi AML specifiche
CLASSI = [
    'CBFB_MYH11',      
    'NPM1',            
    'PML_RARA',        
    'RUNX1_RUNX1T1',   
    'CONTROL'          
]

# Estensioni delle immagini da contare
ESTENSIONI = ('.tif', '.png', '.jpg', '.jpeg')

def analizza_cartelle_dirette():
    print(f"📍 Analisi in corso nel percorso: {BASE_PATH}")
    
    if not os.path.exists(BASE_PATH):
        print(f"❌ ERRORE: La cartella '{BASE_PATH}' non esiste.")
        return

    # 1. TABELLA IMMAGINI
    tabella_immagini = PrettyTable(['Classe', 'Originali', 'Maschere', 'Mancanti'])
    tabella_immagini.align['Classe'] = 'l'
    
    # 2. TABELLA PAZIENTI
    tabella_pazienti = PrettyTable(['Classe', 'Pazienti (Orig)', 'Pazienti (Seg)'])
    tabella_pazienti.align['Classe'] = 'l'
    
    tot_paz_orig = 0
    tot_paz_seg = 0
    tot_orig = 0
    tot_seg = 0

    for cls in CLASSI:
        # Percorsi diretti: BASE_PATH/Original/Classe e BASE_PATH/Segmented/Classe
        path_orig = os.path.join(BASE_PATH, 'Original', cls)
        path_seg = os.path.join(BASE_PATH, 'Segmented', cls)

        # Funzione per contare i pazienti (sottocartelle)
        def conta_pazienti(percorso):
            if os.path.exists(percorso):
                return len([d for d in os.listdir(percorso) if os.path.isdir(os.path.join(percorso, d))])
            return 0

        # Funzione per contare i file ricorsivamente nelle cartelle pazienti
        def conta_file(percorso):
            count = 0
            if os.path.exists(percorso):
                for root, dirs, files in os.walk(percorso):
                    count += len([f for f in files if f.lower().endswith(ESTENSIONI)])
            return count

        # Conteggi Pazienti
        num_paz_orig = conta_pazienti(path_orig)
        num_paz_seg = conta_pazienti(path_seg)

        # Conteggi Immagini
        c_orig = conta_file(path_orig)
        c_seg = conta_file(path_seg)
        
        # Calcolo discrepanze (Originali senza Maschera)
        mancanti = max(0, c_orig - c_seg)
        
        # Popolamento delle due tabelle distinte
        tabella_immagini.add_row([cls, c_orig, c_seg, mancanti])
        tabella_pazienti.add_row([cls, num_paz_orig, num_paz_seg])
        
        tot_paz_orig += num_paz_orig
        tot_paz_seg += num_paz_seg
        tot_orig += c_orig
        tot_seg += c_seg

    # --- STAMPA DEI RISULTATI ---
    print("\n📷 TABELLA 1: DISTRIBUZIONE IMMAGINI")
    print(tabella_immagini)
    
    print("\n👥 TABELLA 2: NUMERO DI PAZIENTI PER CLASSE")
    print(tabella_pazienti)
    
    print(f"\n📊 RIEPILOGO GENERALE:")
    print(f"Total Patients (Original/Segmented): {tot_paz_orig} / {tot_paz_seg}")
    print(f"Total Originals:                     {tot_orig}")
    print(f"Total Masks:                         {tot_seg}")
    
    if tot_orig > tot_seg:
        print(f"⚠️ Warning: {tot_orig - tot_seg} images are missing a mask.")
    else:
        print(f"✅ Success: Every image has a corresponding mask.")

if __name__ == "__main__":
    analizza_cartelle_dirette()
