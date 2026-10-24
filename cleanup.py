import hashlib
import os

# --- CONFIGURAZIONE PERCORSI CORRETTI ---
PROJECT_DIR = './dataset_split/'
FOLDERS = ['Training', 'Testing', 'Validation']
SUB_TYPES = ['Original', 'Segmented']

def compute_hash(file_path):
    """Calcola l'hash MD5 di un file binario per verificarne l'identità."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def clean_dataset():
    hash_dict = {}
    duplicate_count = 0
    
    print("🔍 Avvio scansione del dataset splittato alla ricerca di duplicati fisici...")

    # Navighiamo attraverso la struttura reale presente sul server
    for data_type in FOLDERS:
        for sub_type in SUB_TYPES:
            base_search_path = os.path.join(PROJECT_DIR, data_type, sub_type)
            
            if not os.path.exists(base_search_path):
                continue
                
            print(f"📂 Analizzando cartella: {base_search_path}")
            
            # Scansione dinamica: os.walk legge da solo AML_*, CONTROL, i pazienti e le foto
            for root, dirs, files in os.walk(base_search_path):
                for file in files:
                    # Supporta i .png (generati dal processing) e i formati nativi
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        file_path = os.path.join(root, file)
                        file_hash = compute_hash(file_path)
                        
                        if file_hash in hash_dict:
                            hash_dict[file_hash].append(file_path)
                        else:
                            hash_dict[file_hash] = [file_path]

    print("\n🗑️ Scansione completata. Rimozione dei duplicati in corso...")

    # Rimozione dei file duplicati mantenendo integro il dataset
    for hash_value, file_paths in hash_dict.items():
        if len(file_paths) > 1:
            # Mantieni il primo file della lista (file_paths[0]), elimina gli altri
            for duplicate_path in file_paths[1:]:
                if os.path.exists(duplicate_path):
                    print(f"🗑️ Rimozione duplicato: {duplicate_path}")
                    os.remove(duplicate_path)
                    duplicate_count += 1
                    
                    # SICUREZZA ACCOPPIAMENTO (ORIGINAL <-> SEGMENTED)
                    # Se eliminiamo un file da Original, rimuoviamo la rispettiva maschera in Segmented
                    if 'Original' in duplicate_path:
                        paired_path = duplicate_path.replace('Original', 'Segmented')
                        if os.path.exists(paired_path):
                            os.remove(paired_path)
                            print(f"🔗 Rimossa anche la maschera segmentata accoppiata: {paired_path}")
                            
                    # Viceversa, se eliminiamo un duplicato in Segmented, rimuoviamo l'originale
                    elif 'Segmented' in duplicate_path:
                        paired_path = duplicate_path.replace('Segmented', 'Original')
                        if os.path.exists(paired_path):
                            os.remove(paired_path)
                            print(f"🔗 Rimossa anche l'immagine originale accoppiata: {paired_path}")

    print(f"\n✅ OPERAZIONE COMPLETATA!")
    print(f"Numero totale di file duplicati rimossi dal dataset: {duplicate_count}")

if __name__ == '__main__':
    clean_dataset()