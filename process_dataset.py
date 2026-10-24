import os
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURAZIONE ---
SOURCE_DATASET_PATH = "./data" 
OUTPUT_BASE = "./dataset"

def generate_mask(cv2_img):
    lab = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(lab)
    
    # 1. Soglia per ottenere la maschera
    _, mask_binary = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Rimuovi i piccoli rumori (filtro area)
    # Troviamo tutti i contorni
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Creiamo una maschera nera vuota
    final_mask = np.zeros_like(mask_binary)
    
    # 3. Mantieni solo il contorno più grande (la cellula)
    if contours:
        # Ordina per area e prendi il più grande
        largest_contour = max(contours, key=cv2.contourArea)
        # Se l'area è abbastanza grande, disegnala sulla maschera
        if cv2.contourArea(largest_contour) > 500: # <-- SOGLIA AREA: Aumenta se vedi ancora puntini
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
            
    # 4. Applica la maschera pulita
    mask_3d = cv2.merge([final_mask, final_mask, final_mask])
    color_mask = cv2.bitwise_and(cv2_img, mask_3d)
    
    return color_mask

def process_all():
    """Ciclo principale con protezione per evitare ricorsione sulla cartella data."""
    output_abs = os.path.abspath(OUTPUT_BASE)
    source_abs = os.path.abspath(SOURCE_DATASET_PATH)
    total_count = 0
    
    processed_classes = set()
    
    # Invece di walk su SOURCE_DATASET_PATH, iteriamo manualmente sui contenuti
    # per avere controllo totale sulla gerarchia
    if not os.path.exists(source_abs):
        print(f"Errore: {source_abs} non trovato.")
        return

    # Otteniamo le cartelle delle classi (es. NPM1, CBFB_MYH11)
    for class_name in os.listdir(source_abs):
        class_path = os.path.join(source_abs, class_name)
        
        # Saltiamo file o cartelle che non sono classi
        if not os.path.isdir(class_path) or class_name == os.path.basename(output_abs):
            continue
            
        print(f"📂 Elaborazione classe: {class_name}")
        processed_classes.add(class_name)
            
        for patient_folder in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_folder)
            
            if not os.path.isdir(patient_path):
                continue
                
            dest_orig = os.path.join(OUTPUT_BASE, "Original", class_name, patient_folder)
            dest_seg = os.path.join(OUTPUT_BASE, "Segmented", class_name, patient_folder)
            os.makedirs(dest_orig, exist_ok=True)
            os.makedirs(dest_seg, exist_ok=True)

            for file_name in os.listdir(patient_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    src_path = os.path.join(patient_path, file_name)
                    
                    try:
                        pil_img = Image.open(src_path).convert('RGB')
                        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2RGB) # Corretto per RGB
                        
                        out_name = os.path.splitext(file_name)[0] + ".png"
                        cv2.imwrite(os.path.join(dest_orig, out_name), cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR))
                        
                        mask_color = generate_mask(img_cv2)
                        
                        cv2.imwrite(os.path.join(dest_seg, out_name), mask_color, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        
                        total_count += 1
                    except Exception as e:
                        pass
    
    print(f"\n🚀 COMPLETATO! Processate {total_count} immagini.")
        
if __name__ == "__main__":
    process_all()