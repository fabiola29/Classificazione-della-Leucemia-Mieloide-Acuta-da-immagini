import os
from prettytable import PrettyTable

# --- CONFIGURAZIONE PERCORSO ---
# Punta alla cartella creata da prepare_data.py
BASE_PATH = "./dataset_split_con_test"

# Classi del dataset AML
CLASSI = ['CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1', 'CONTROL']

# Estensioni supportate
ESTENSIONI = ('.png', '.jpg', '.jpeg', '.tif')

def conta_file(percorso):
    """Conta le immagini con estensione supportata in un percorso."""
    count = 0
    if os.path.exists(percorso):
        for root, dirs, files in os.walk(percorso):
            count += len([f for f in files if f.lower().endswith(ESTENSIONI)])
    return count

def conta_pazienti(percorso):
    """Conta i pazienti (sottocartelle) presenti in un percorso."""
    if os.path.exists(percorso):
        return len([d for d in os.listdir(percorso) if os.path.isdir(os.path.join(percorso, d))])
    return 0

def analizza_sezione(path_sezione, nome_sezione):
    """Genera e stampa le tabelle per una specifica sezione (es. Test_Set o Fold_X/Training)."""
    if not os.path.exists(path_sezione):
        return

    print(f"\n" + "="*60)
    print(f"📊 ANALISI: {nome_sezione.upper()}")
    print("="*60)
    
    tabella_immagini = PrettyTable(['Classe', 'Originali', 'Maschere', 'Mancanti'])
    tabella_immagini.align['Classe'] = 'l'
    
    tabella_pazienti = PrettyTable(['Classe', 'Pazienti (Orig)', 'Pazienti (Seg)'])
    tabella_pazienti.align['Classe'] = 'l'
    
    tot_mancanti = 0

    for cls in CLASSI:
        path_orig = os.path.join(path_sezione, 'Original', cls)
        path_seg = os.path.join(path_sezione, 'Segmented', cls)

        # Conteggi Immagini
        n_orig = conta_file(path_orig)
        n_seg = conta_file(path_seg)
        mancanti = max(0, n_orig - n_seg)
        tot_mancanti += mancanti
        
        # Conteggi Pazienti
        paz_orig = conta_pazienti(path_orig)
        paz_seg = conta_pazienti(path_seg)

        # Popolamento Tabelle
        tabella_immagini.add_row([cls, n_orig, n_seg, mancanti])
        tabella_pazienti.add_row([cls, paz_orig, paz_seg])

    print(f"\n📷 Immagini - {nome_sezione}")
    print(tabella_immagini)
    
    print(f"\n👥 Pazienti - {nome_sezione}")
    print(tabella_pazienti)
    
    if tot_mancanti > 0:
        print(f"\n⚠️ Attenzione: Ci sono {tot_mancanti} immagini senza maschera in {nome_sezione}.")
    else:
        print(f"\n✅ Tutte le immagini in {nome_sezione} hanno una maschera corrispondente.")

def analizza_distribuzione_kfold():
    print(f"📍 Analisi della struttura K-Fold in corso: {BASE_PATH}")
    
    if not os.path.exists(BASE_PATH):
        print(f"❌ ERRORE: La cartella '{BASE_PATH}' non esiste.")
        print("Assicurati di aver eseguito prima lo script di preparazione dei dati (prepare_data.py).")
        return

    # 1. Analisi Test Set Esterno (Holdout)
    path_test = os.path.join(BASE_PATH, "Test_Set")
    if os.path.exists(path_test):
        analizza_sezione(path_test, "Test Set Esterno (Holdout)")

    # 2. Scansione dinamica e analisi di ciascun Fold
    elementi = sorted(os.listdir(BASE_PATH))
    cartelle_fold = [f for f in elementi if f.startswith("Fold_") and os.path.isdir(os.path.join(BASE_PATH, f))]

    if not cartelle_fold:
        print("\n⚠️ Nessuna cartella 'Fold_X' trovata nel percorso.")
        return

    for fold_dir in cartelle_fold:
        path_fold = os.path.join(BASE_PATH, fold_dir)
        
        # Analisi Training per il Fold corrente
        path_train = os.path.join(path_fold, "Training")
        analizza_sezione(path_train, f"{fold_dir} -> Training")

        # Analisi Validation per il Fold corrente
        path_val = os.path.join(path_fold, "Validation")
        analizza_sezione(path_val, f"{fold_dir} -> Validation")

if __name__ == "__main__":
    analizza_distribuzione_kfold()
