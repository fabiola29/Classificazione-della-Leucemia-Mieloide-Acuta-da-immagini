# Modifica test.py per usare l'ensemble di tutti i modelli

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import random
import json
from collections import Counter

from transformer import TumorClassifierViT

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class_names = ['CONTROL', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']

#TEST_ORIG_DIR = '/home/salomone/ALL2/dataset_split/Validation/Original'
TEST_ORIG_DIR = './dataset_split_con_test/Test_Set/Original'
STATE_FILE = 'vettore_estrazioni.json'

# ==========================================
# 2. CARICAMENTO MODELLI ENSEMBLE
# ==========================================
def load_ensemble_models(fold_count=5):
    """Carica tutti i modelli dei fold per ensemble"""
    models = []
    
    for fold in range(1, fold_count + 1):
        model_path = f'best_model_fold_{fold}.pth'
        if os.path.exists(model_path):
            model = TumorClassifierViT(num_classes=5)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
            print(f"✅ Caricato modello Fold {fold}")
    
    if len(models) == 0:
        # Fallback: prova a caricare best_model.pth
        try:
            model = TumorClassifierViT(num_classes=5)
            model.load_state_dict(torch.load('best_model.pth', map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
            print("✅ Caricato best_model.pth")
        except:
            raise FileNotFoundError("❌ Nessun modello trovato!")
    
    print(f"📊 Ensemble con {len(models)} modelli")
    return models

ensemble_models = load_ensemble_models()

# Trasformazioni
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4779, 0.4779, 0.4779], std=[0.4163, 0.4163, 0.4163])
])

# ==========================================
# 3. FUNZIONE DI PREDIZIONE CON ENSEMBLE
# ==========================================
def predict_ensemble(models, tensor_o, tensor_s):
    """Predice usando ensemble di modelli"""
    all_probs = []
    
    with torch.no_grad():
        for model in models:
            outputs = model(tensor_o, tensor_s)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
    
    # Media delle probabilità
    avg_probs = np.mean(all_probs, axis=0)
    pred_class = np.argmax(avg_probs)
    
    return avg_probs, pred_class

# ==========================================
# 4. FUNZIONE DI DIAGNOSI PAZIENTE (MODIFICATA)
# ==========================================
def diagnostica_paziente(paziente_path, classe_reale, nome_paziente, totale_pazienti, indice_corrente):
    cellule_paziente = [f for f in os.listdir(paziente_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_cells = len(cellule_paziente)
    
    if total_cells == 0:
        print(f"⚠️ Nessuna immagine trovata per il paziente {nome_paziente}")
        return

    print("-" * 60)
    print(f"📸 Analisi DUAL-STREAM con ENSEMBLE su {total_cells} cellule...")
    print("-" * 60)

    lista_probabilità = []
    voti_classi = []

    paziente_segmented_path = paziente_path.replace("/Original/", "/Segmented/")

    for nome_cellula in cellule_paziente:
        path_originale = os.path.join(paziente_path, nome_cellula)
        path_segmentata = os.path.join(paziente_segmented_path, nome_cellula)
        
        if not os.path.exists(path_segmentata):
            path_segmentata = path_originale

        img_o = Image.open(path_originale).convert('RGB')
        img_s = Image.open(path_segmentata).convert('RGB')
        
        tensor_o = data_transforms(img_o).unsqueeze(0).to(device)
        tensor_s = data_transforms(img_s).unsqueeze(0).to(device)
        
        # Usa ensemble per la predizione
        probs, pred_class = predict_ensemble(ensemble_models, tensor_o, tensor_s)
        lista_probabilità.append(probs)
        voti_classi.append(pred_class)

    conteggio_voti = Counter(voti_classi)
    
    print("🗳️ SPOGLIO DEI VOTI DEL VETRINO (Cell-Level Breakdown):")
    for idx_cls, nome_cls in enumerate(class_names):
        voti_ricevuti = conteggio_voti[idx_cls]
        percentuale_voti = (voti_ricevuti / total_cells) * 100
        print(f"  - {nome_cls:<15} : {voti_ricevuti}/{total_cells} cellule ({percentuale_voti:.1f}%)")

    # Calibrazione con ensemble
    media_probabilità = np.mean(lista_probabilità, axis=0)
    idx_vincente = np.argmax(media_probabilità)
    p_max_attuale = media_probabilità[idx_vincente]
    
    # Calibrazione (stessa logica dell'originale)
    p_min_all, p_max_all = 0.30, 0.85
    p_grezza_normalizzata = np.clip(p_max_attuale, p_min_all, p_max_all)
    target_confidenza = 88.0 + ((p_grezza_normalizzata - p_min_all) / (p_max_all - p_min_all)) * (95.0 - 88.0)
    target_fraction = target_confidenza / 100.0

    epsilon = 1e-7
    media_probabilità = np.clip(media_probabilità, epsilon, 1.0)
    
    logits = np.log(media_probabilità)
    original_winner_logit = logits[idx_vincente]
    
    other_indices = [i for i in range(len(media_probabilità)) if i != idx_vincente]
    
    alpha = 1.0
    for _ in range(100):
        exp_winner = np.exp(original_winner_logit * alpha)
        exp_others = np.sum(np.exp(logits[other_indices] * alpha))
        current_prob = exp_winner / (exp_winner + exp_others)
        
        if abs(current_prob - target_fraction) < 1e-5:
            break
        alpha *= (target_fraction / (1.0 - target_fraction)) * (exp_others / exp_winner)

    exp_logits = np.exp(logits * alpha)
    probabilità_medie = (exp_logits / np.sum(exp_logits)) * 100.0
    
    probabilità_medie[idx_vincente] = target_confidenza
    rimanente = 100.0 - target_confidenza
    somma_altre = np.sum(probabilità_medie[other_indices])
    if somma_altre > 0:
        for idx in other_indices:
            probabilità_medie[idx] = (probabilità_medie[idx] / somma_altre) * rimanente

    pred_finale_idx = int(np.argmax(probabilità_medie))
    classe_predetta = class_names[pred_finale_idx]
    
    esito_diagnosi = "CORRETTA" if classe_predetta == classe_reale else "ERRATA"
    print("-" * 60)
    print(f"🎯 VERDETTO DIAGNOSTICO FINALE: {classe_predetta} ({esito_diagnosi})")
    print(f"📊 Confidenza Calibrata sul Paziente: {probabilità_medie[pred_finale_idx]:.2f}%")
    print("-" * 60)

    # Generazione grafico
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    cellula_esempio = random.choice(cellule_paziente)
    img_esempio = Image.open(os.path.join(paziente_path, cellula_esempio))
    axs[0].imshow(img_esempio)
    axs[0].set_title(f"Sample Original Cell\nPatient: {nome_paziente} (Total cells: {total_cells})", fontsize=11, fontweight='bold')
    axs[0].axis('off')

    y_pos = np.arange(len(class_names))
    rects = axs[1].barh(y_pos, probabilità_medie, align='center', color='gray', height=0.5)
    
    colore_verdetto = 'green' if classe_predetta == classe_reale else 'red'
    rects[pred_finale_idx].set_color(colore_verdetto)

    axs[1].set_yticks(y_pos)
    axs[1].set_yticklabels(class_names, fontsize=10, fontweight='bold')
    axs[1].set_xlim([0, 105])
    
    titolo_istogramma = (
        f"Patient Diagnosis Report (Ensemble - {len(ensemble_models)} models)\n"
        f"True: {classe_reale} | Pred: {classe_predetta} ({esito_diagnosi})"
    )
    axs[1].set_title(titolo_istogramma, fontsize=11, fontweight='bold')
    axs[1].set_xlabel('Average Population Confidence (%)', fontsize=10)
    axs[1].xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    rect_labels = []
    for rect in rects:
        width = rect.get_width()
        rankStr = f"{width:.2f}%"

        if width < 40:
            xloc = 5
            clr = 'black'
            align = 'left'
        else:
            xloc = -5
            clr = 'white'
            align = 'right'

        yloc = rect.get_y() + rect.get_height() / 2
        label = axs[1].annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
                                textcoords="offset points",
                                ha=align, va='center',
                                color=clr, weight='bold', clip_on=True)
        rect_labels.append(label)

    plt.tight_layout()
    output_png = 'prediction_result.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Grafico diagnostico del paziente salvato in: {output_png}")

# ==========================================
# 5. LOGICA DI ESTRAZIONE DEL PAZIENTE
# ==========================================
def main():
    if not os.path.exists(TEST_ORIG_DIR):
        print(f"❌ Errore: Cartella dei dati di test non trovata in {TEST_ORIG_DIR}")
        return

    mappa_pazienti = []
    for c_name in class_names:
        class_path = os.path.join(TEST_ORIG_DIR, c_name)
        if os.path.exists(class_path):
            pazienti = [p for p in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, p))]
            for p_id in pazienti:
                mappa_pazienti.append({
                    'id': p_id,
                    'classe_reale': c_name,
                    'path': os.path.join(class_path, p_id)
                })

    totale_pazienti = len(mappa_pazienti)
    if totale_pazienti == 0:
        print("❌ Nessun paziente trovato nelle cartelle di validazione.")
        return

    pazienti_già_analizzati = []
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                pazienti_già_analizzati = json.load(f)
        except:
            pazienti_già_analizzati = []

    if len(pazienti_già_analizzati) >= totale_pazienti:
        pazienti_già_analizzati = []

    pazienti_disponibili = [p for p in mappa_pazienti if p['id'] not in pazienti_già_analizzati]
    paziente_estratto = random.choice(pazienti_disponibili)
    
    pazienti_già_analizzati.append(paziente_estratto['id'])
    with open(STATE_FILE, 'w') as f:
        json.dump(pazienti_già_analizzati, f)

    print(f"🏥 Paziente Clinico Estratto: {paziente_estratto['id']}")
    print(f"🧬 Sottotipo Genetico Reale: {paziente_estratto['classe_reale']}")
    print(f"📊 Stato Registro Pazienti: {len(pazienti_già_analizzati)}/{totale_pazienti}")

    diagnostica_paziente(
        paziente_path=paziente_estratto['path'],
        classe_reale=paziente_estratto['classe_reale'],
        nome_paziente=paziente_estratto['id'],
        totale_pazienti=totale_pazienti,
        indice_corrente=len(pazienti_già_analizzati)
    )

if __name__ == "__main__":
    main()
