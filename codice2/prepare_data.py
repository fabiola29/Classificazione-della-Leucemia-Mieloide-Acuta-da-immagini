import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# --- CONFIGURAZIONE PERCORSI RELATIVI ---
DATASET_BASE = "../dataset"

# Sottocartelle sorgente (all'interno di ./dataset)
ORIG_BASE = os.path.join(DATASET_BASE, "Original")
SEG_BASE = os.path.join(DATASET_BASE, "Segmented")

# Cartella di destinazione finale
OUTPUT_DIR = "./dataset_split_con_test"

# --- PARAMETRI DI SPLIT ---
K_FOLDS = 5               # Numero di fold per la Cross-Validation
TEST_SIZE = 0.20          # 20% dei pazienti riservati al Test Set esterno
RANDOM_SEED = 42

def prepare_kfold_split():
    data_list = []
    
    if not os.path.exists(ORIG_BASE):
        print(f"❌ ERRORE: Non trovo la cartella 'dataset' nel percorso corrente.")
        print(f"Percorso cercato: {os.path.abspath(ORIG_BASE)}")
        return

    # 1. Scansione dinamica delle classi e dei pazienti
    classi = [c for c in os.listdir(ORIG_BASE) if not c.startswith('.')]
    
    for class_name in classi:
        class_path = os.path.join(ORIG_BASE, class_name)
        if not os.path.isdir(class_path): 
            continue
        
        pazienti = [p for p in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, p))]
        
        for p_id in pazienti:
            data_list.append({
                'patient_id': p_id,
                'class': class_name,
                'group_key': p_id  # Chiave di raggruppamento rigorosa per paziente
            })

    df_pazienti = pd.DataFrame(data_list)
    
    if df_pazienti.empty:
        print("❌ Nessun paziente trovato.")
        return

    print(f"📊 Totale pazienti unici trovati: {len(df_pazienti)}")
    print(df_pazienti['class'].value_counts())

    # 2. Pulizia della directory di destinazione
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. SPLIT ESTERNO: Isola il 20% come Test Set definitivo prima del K-Fold
    train_val_df, test_df = train_test_split(
        df_pazienti,
        test_size=TEST_SIZE,
        stratify=df_pazienti['class'],
        random_state=RANDOM_SEED
    )

    print(f"\n🔒 Test Set esterno isolato: {len(test_df)} pazienti")
    print(f"🔄 Pool di Training/Validation (per K-Fold): {len(train_val_df)} pazienti")

    # Copia fisica del Test Set nella sua cartella dedicata
    copia_pazienti(test_df, os.path.join(OUTPUT_DIR, "Test_Set"))

    # 4. Generazione dei K Fold sul restante 80% con StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train_val_df = train_val_df.reset_index(drop=True)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_df, train_val_df['class'], groups=train_val_df['group_key'])):
        fold_num = fold + 1
        print(f"\n🔄 Generazione Fold {fold_num}/{K_FOLDS}...")
        
        fold_train_df = train_val_df.iloc[train_idx]
        fold_val_df = train_val_df.iloc[val_idx]

        fold_dir = os.path.join(OUTPUT_DIR, f"Fold_{fold_num}")
        
        # Copia dei dati di Training e Validation per questo fold
        copia_pazienti(fold_train_df, os.path.join(fold_dir, "Training"))
        copia_pazienti(fold_val_df, os.path.join(fold_dir, "Validation"))
        
        print(f"   -> Train pazienti: {len(fold_train_df)} | Validation pazienti: {len(fold_val_df)}")

    print(f"\n✅ OPERAZIONE RIUSCITA!")
    print(f"La struttura K-Fold con Test Set è pronta in: {os.path.abspath(OUTPUT_DIR)}")

def copia_pazienti(df_pazienti, target_split_dir):
    """Funzione di supporto per copiare i file fisici nelle rispettive cartelle."""
    for _, row in df_pazienti.iterrows():
        p_id = row['patient_id']
        c_name = row['class']
        
        for folder_type in ['Original', 'Segmented']:
            src_dir = os.path.join(DATASET_BASE, folder_type, c_name, p_id)
            dest_dir = os.path.join(target_split_dir, folder_type, c_name, p_id)
            
            if os.path.exists(src_dir):
                os.makedirs(dest_dir, exist_ok=True)
                for file_name in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, file_name)
                    dest_file = os.path.join(dest_dir, file_name)
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)

if __name__ == "__main__":
    prepare_kfold_split()
