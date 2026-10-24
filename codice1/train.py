import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import seaborn as sns
import time
import sys
import glob
from collections import Counter, defaultdict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

from transformer import TumorClassifierViT

RUN_TRAINING = "train.py" in sys.argv[0] or True
#RUN_TRAINING = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
DATASET_ROOT = './dataset_split_con_test'
IMAGE_RESOLUTION = 224
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4

TEMPERATURE = 1.0

# ============================================================
# CLASS WEIGHTS ASIMMETRICI (CONTROL 2x, MALATI 4x)
# ============================================================
#CLASS_WEIGHTS = [2.0, 4.0, 4.0, 4.0, 4.0]
CLASS_WEIGHTS = [2.0, 6.0, 4.0, 4.0, 6.0]  # CBFB e RUNX1 a 6

# ============================================================
# SOGLIE ADATTIVE PER CLASSE (riducono falsi positivi)
# ============================================================
#CLASS_THRESHOLDS = [0.25, 0.55, 0.60, 0.55, 0.50]
#CLASS_THRESHOLDS = [0.25, 0.40, 0.40, 0.40, 0.40]
CLASS_THRESHOLDS = [0.35, 0.45, 0.40, 0.40, 0.50]

# ============================================================
# EARLY STOPPING
# ============================================================
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.002
INITIAL_LR = 2e-5
WEIGHT_DECAY = 0.35

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"🚀 Training avviato su: {device}")

if device.type == 'cuda':
    torch.cuda.empty_cache()


# ==========================================
# FUNZIONE DI CHECK ALLINEAMENTO DATI
# ==========================================
def check_data_alignment(loader):
    try:
        img_orig, img_seg, labels, _ = next(iter(loader))
        o_img = img_orig[0].cpu().permute(1, 2, 0).numpy()
        s_img = img_seg[0].cpu().permute(1, 2, 0).numpy()
        o_img = (o_img - o_img.min()) / (o_img.max() - o_img.min() + 1e-8)
        s_img = (s_img - s_img.min()) / (s_img.max() - s_img.min() + 1e-8)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(o_img)
        plt.title(f"Original - Label: {labels[0]}")
        plt.subplot(1, 2, 2)
        plt.imshow(s_img)
        plt.title("Segmented")
        plt.savefig("data_alignment_check.png")
        plt.close()
        print("✅ Controllo dati salvato in 'data_alignment_check.png'")
    except Exception as e:
        print(f"⚠️ Impossibile generare data_alignment_check.png: {e}")


# ==========================================
# 2. DATASET CUSTOM (5 CLASSI)
# ==========================================
class DualStreamDataset(Dataset):
    def __init__(self, base_path, split_type='Training', fold_num=1, training=False):
        self.base_path = base_path
        self.training = training
        self.samples = []

        self.classes = ['CONTROL', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        target_dir = os.path.join(base_path, f"Fold_{fold_num}", split_type)
        orig_root = os.path.join(target_dir, "Original")
        seg_root = os.path.join(target_dir, "Segmented")

        print(f"📦 Indicizzazione dei percorsi per Fold {fold_num} ({split_type})...")

        if os.path.exists(orig_root):
            for cls in os.listdir(orig_root):
                cls_orig_dir = os.path.join(orig_root, cls)
                cls_seg_dir = os.path.join(seg_root, cls)

                if os.path.isdir(cls_orig_dir) and cls in self.class_to_idx:
                    pazienti = [p for p in os.listdir(cls_orig_dir) if os.path.isdir(os.path.join(cls_orig_dir, p))]

                    for p_id in pazienti:
                        p_orig_path = os.path.join(cls_orig_dir, p_id)
                        p_seg_path = os.path.join(cls_seg_dir, p_id)

                        if os.path.exists(p_orig_path):
                            for fname in os.listdir(p_orig_path):
                                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    p_orig = os.path.join(p_orig_path, fname)
                                    p_seg = os.path.join(p_seg_path, fname)
                                    if os.path.exists(p_seg):
                                        unique_patient_id = f"{cls}___{p_id}"
                                        self.samples.append((p_orig, p_seg, self.class_to_idx[cls], unique_patient_id))

        print(f"✅ Indicizzati {len(self.samples)} campioni per il Fold {fold_num} ({split_type}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_orig, p_seg, label, patient_id = self.samples[idx]

        try:
            img_o = Image.open(p_orig).convert('RGB')
            img_s = Image.open(p_seg).convert('L')
        except Exception as e:
            img_o = Image.new('RGB', (IMAGE_RESOLUTION, IMAGE_RESOLUTION))
            img_s = Image.new('L', (IMAGE_RESOLUTION, IMAGE_RESOLUTION))

        res = (IMAGE_RESOLUTION, IMAGE_RESOLUTION)

        if self.training:
            img_o = TF.resize(img_o, res)
            img_s = TF.resize(img_s, res)

            rot_angle = random.choice([0, 90, 180, 270])
            if rot_angle != 0:
                img_o = TF.rotate(img_o, rot_angle)
                img_s = TF.rotate(img_s, rot_angle)

            if random.random() > 0.5:
                img_o = TF.hflip(img_o)
                img_s = TF.hflip(img_s)

            if random.random() > 0.5:
                img_o = TF.vflip(img_o)
                img_s = TF.vflip(img_s)

            jitter = transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            )
            img_o = jitter(img_o)
        else:
            img_o = TF.resize(img_o, res)
            img_s = TF.resize(img_s, res)

        try:
            img_s = img_s.convert('RGB')
        except Exception as e:
            print(f"⚠️ Errore conversione img_s per paziente {patient_id}: {e}")
            img_s = Image.new('RGB', res, color='black')

        o_tensor = TF.to_tensor(img_o)
        s_tensor = TF.to_tensor(img_s)

        if self.training and random.random() > 0.5:
            erase_h = random.randint(int(IMAGE_RESOLUTION * 0.1), int(IMAGE_RESOLUTION * 0.4))
            erase_w = random.randint(int(IMAGE_RESOLUTION * 0.1), int(IMAGE_RESOLUTION * 0.4))
            y = random.randint(0, IMAGE_RESOLUTION - erase_h)
            x = random.randint(0, IMAGE_RESOLUTION - erase_w)
            s_tensor = TF.erase(s_tensor, y, x, erase_h, erase_w, v=0.0)

        norm = transforms.Normalize(mean=[0.4779, 0.4779, 0.4779], std=[0.4163, 0.4163, 0.4163])
        o_tensor = norm(o_tensor)
        s_tensor = norm(s_tensor)

        return o_tensor, s_tensor, label, patient_id


# ==========================================
# 3. RACCOLTA INTEGRALE DI TUTTI I PAZIENTI
# ==========================================
def raccogli_tutti_i_pazienti(dataset_root):
    tutti_i_pazienti = []
    classi_target = ['CONTROL', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
    conteggi_stampa = {c: 0 for c in classi_target}

    if not os.path.exists(dataset_root):
        print(f"❌ ERRORE CRITICO: La cartella radice '{dataset_root}' non esiste!")
        sys.exit(1)

    for root, dirs, files in os.walk(dataset_root):
        if 'Original' in root and not dirs:
            immagini = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(immagini) == 0:
                continue

            nome_paziente = os.path.basename(root)
            percorso_upper = root.upper()

            classe_identificata = None
            if "CONTROL" in percorso_upper:
                classe_identificata = 'CONTROL'
            elif "CBFB" in percorso_upper or "MYH11" in percorso_upper:
                classe_identificata = 'CBFB_MYH11'
            elif "NPM1" in percorso_upper:
                classe_identificata = 'NPM1'
            elif "PML" in percorso_upper or "RARA" in percorso_upper:
                classe_identificata = 'PML_RARA'
            elif "RUNX1" in percorso_upper:
                classe_identificata = 'RUNX1_RUNX1T1'

            if not classe_identificata:
                continue

            p_info = {
                'id_univoco': f"{classe_identificata}___{nome_paziente}",
                'nome_cartella': nome_paziente,
                'classe_vera': classe_identificata,
                'path_completo_original': root
            }

            if p_info['id_univoco'] not in [x['id_univoco'] for x in tutti_i_pazienti]:
                tutti_i_pazienti.append(p_info)
                conteggi_stampa[classe_identificata] += 1

    print("\n📊 [SCANSIONE INTEGRALE 100% DEI PAZIENTI]")
    for cls in classi_target:
        tot = conteggi_stampa[cls]
        print(f"   • Classe [{cls:20}] -> Pazienti Totali: {tot:2d}")

    return tutti_i_pazienti

tutti_pazienti = raccogli_tutti_i_pazienti(DATASET_ROOT)

MAX_PER_PAZIENTE = 500

# ============================================================
# CAMPIONAMENTO BILANCIATO A LIVELLO PAZIENTE
# ============================================================
def crea_train_subset_bilanciato(dataset, immagini_per_classe=6265, seed=42):
    from torch.utils.data import Subset

    rng = random.Random(seed)

    print("\n" + "=" * 75)
    print("🎯 CAMPIONAMENTO TRAIN BILANCIATO A LIVELLO PAZIENTE")
    print("=" * 75)
    print(f"   MAX_PER_PAZIENTE = {MAX_PER_PAZIENTE}")

    class_to_patients = defaultdict(lambda: defaultdict(list))

    for idx, sample in enumerate(dataset.samples):
        label = sample[2]
        patient_id = sample[3]
        class_to_patients[label][patient_id].append(idx)

    selected_indices = []

    for class_idx in sorted(class_to_patients.keys()):
        patients = class_to_patients[class_idx]
        class_name = dataset.classes[class_idx] if hasattr(dataset, "classes") else str(class_idx)
        total_available = sum(len(indices) for indices in patients.values())

        take = min(immagini_per_classe, total_available)

        print(f"\n🧬 {class_name}")
        print(f"   Pazienti disponibili: {len(patients)}")
        print(f"   Immagini disponibili: {total_available:,}")
        print(f"   Immagini richieste:   {take:,}")

        if total_available < immagini_per_classe:
            print(f"⚠️ ATTENZIONE: {class_name} ha solo {total_available:,} immagini, usando {take:,}")

        patient_indices = {}
        for patient_id, indices in patients.items():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            patient_indices[patient_id] = shuffled[:MAX_PER_PAZIENTE]

        total_after_limit = sum(len(v) for v in patient_indices.values())
        if total_after_limit < take:
            print(f"   ⚠️ Dopo limiting: {total_after_limit:,} immagini (richieste: {take:,})")
            take = min(take, total_after_limit)

        num_patients = len(patient_indices)
        base_quota = take // num_patients
        remainder = take % num_patients

        patient_ids = list(patient_indices.keys())
        rng.shuffle(patient_ids)

        quotas = {}
        for i, patient_id in enumerate(patient_ids):
            quotas[patient_id] = base_quota + 1 if i < remainder else base_quota

        selected_by_patient = {}
        deficit = 0

        for patient_id in patient_ids:
            available = len(patient_indices[patient_id])
            requested = quotas[patient_id]
            take_patient = min(available, requested)
            selected_by_patient[patient_id] = patient_indices[patient_id][:take_patient]
            deficit += requested - take_patient

        while deficit > 0:
            candidates = []
            for patient_id in patient_ids:
                available = len(patient_indices[patient_id])
                already_selected = len(selected_by_patient[patient_id])
                remaining = available - already_selected
                if remaining > 0:
                    candidates.append((patient_id, remaining))

            if not candidates:
                break

            rng.shuffle(candidates)
            candidates.sort(key=lambda x: x[1], reverse=True)

            for patient_id, remaining in candidates:
                if deficit <= 0:
                    break
                selected_by_patient[patient_id].append(
                    patient_indices[patient_id][len(selected_by_patient[patient_id])]
                )
                deficit -= 1

        selected_class_indices = []
        for patient_id in patient_ids:
            selected_class_indices.extend(selected_by_patient[patient_id])

        if len(selected_class_indices) != take:
            print(f"⚠️ Campionamento {class_name}: richieste {take}, ottenute {len(selected_class_indices)}")

        rng.shuffle(selected_class_indices)
        selected_indices.extend(selected_class_indices)

        patient_counts = [len(v) for v in selected_by_patient.values() if len(v) > 0]
        if patient_counts:
            print(f"   Immagini selezionate: {len(selected_class_indices):,}")
            print(f"   Min per paziente: {min(patient_counts)}")
            print(f"   Max per paziente: {max(patient_counts)}")
            print(f"   Media per paziente: {sum(patient_counts) / len(patient_counts):.2f}")

    rng.shuffle(selected_indices)

    final_counts = Counter(dataset.samples[idx][2] for idx in selected_indices)
    print("\n" + "-" * 75)
    print(f"✅ TRAIN FINALE: {len(selected_indices):,} immagini")
    print("\n📊 Distribuzione finale:")
    for class_idx in sorted(final_counts.keys()):
        class_name = dataset.classes[class_idx] if hasattr(dataset, "classes") else str(class_idx)
        print(f"   {class_name:<25} {final_counts[class_idx]:,}")
    print("=" * 75)

    return Subset(dataset, selected_indices)


def _forward_model_features(model, img_o, img_s, device):
    """
    Esegue il forward del modello e restituisce (logits, features).
    Il TumorClassifierViT restituisce (logits, image_embedding) con return_features=True,
    dove image_embedding e' la media sui 196 patch token (B, 384).
    """
    img_o = img_o.to(device, non_blocking=True)
    img_s = img_s.to(device, non_blocking=True)
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        out = model(img_o, img_s, return_features=True)
    if isinstance(out, tuple):
        return out[0], out[1]
    raise RuntimeError("TumorClassifierViT non ha restituito (logits, features) con return_features=True.")


def aggregate_patient_logits_weighted(logits_list, labels_list, pids_list, device):
    """
    Aggrega i logits a livello paziente usando una MEDIA PESATA PER CONFIDENZA.
    Applica SOGLIE ADATTIVE PER CLASSE per ridurre i falsi positivi.
    """
    logits_tensor = torch.cat(logits_list, dim=0).float()
    labels_tensor = torch.cat(labels_list, dim=0)

    unique_pids, inverse = torch.unique(
        torch.tensor([hash(p) for p in pids_list], device=device), return_inverse=True
    )
    num_patients = len(unique_pids)

    probs = torch.softmax(logits_tensor, dim=1)
    conf = probs.max(dim=1).values.unsqueeze(1)
    weighted_logits = logits_tensor * conf

    patient_logits_agg = torch.zeros((num_patients, logits_tensor.shape[1]), device=device)
    patient_logits_agg.index_add_(0, inverse, weighted_logits)

    sum_weights = torch.zeros((num_patients, 1), device=device)
    sum_weights.index_add_(0, inverse, conf)
    patient_logits_agg = patient_logits_agg / (sum_weights + 1e-6)

    patient_true_labels = torch.zeros(num_patients, dtype=torch.long, device=device)
    patient_true_labels.index_copy_(0, inverse, labels_tensor)

    probs_patient = torch.softmax(patient_logits_agg, dim=1)
    thresholds = torch.tensor(CLASS_THRESHOLDS, device=device)
    adjusted_probs = probs_patient / thresholds
    patient_preds = adjusted_probs.argmax(dim=1)

    return patient_logits_agg, patient_true_labels, patient_preds


# ==========================================
# 4. FUNZIONI DI SUPPORTO
# ==========================================
def genera_grafico_curva_fold(epoch_corrente, fold, hist_tr_l, hist_va_l, hist_tr_a, hist_va_a):
    epochs_range = range(1, len(hist_tr_l) + 1)
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist_tr_l, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, hist_va_l, label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.title(f'Patient Loss History - Fold {fold+1}', fontsize=12, fontweight='bold')
    plt.xlabel('Epoche'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist_tr_a, label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, hist_va_a, label='Val Accuracy', color='#ff7f0e', linewidth=2)
    plt.title(f'Patient Accuracy History - Fold {fold+1}', fontsize=12, fontweight='bold')
    plt.xlabel('Epoche'); plt.ylabel('Accuracy'); plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout()
    plt.savefig(f'curves_fold_{fold+1}.png', dpi=300)
    plt.close()


def salva_matrici_confusione_fold(fold, classes, tr_true, tr_pred, va_true, va_pred):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    data = [
        ('Fold Training Set', tr_true, tr_pred, 'Blues'),
        ('Fold Validation Set', va_true, va_pred, 'Oranges')
    ]

    for ax, (title, y_true, y_pred, cmap) in zip(axes, data):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
        ax.set_title(f'Fold {fold+1} - {title}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Diagnosi Predetta'); ax.set_ylabel('Diagnosi Reale')

    plt.tight_layout()
    plt.savefig(f'confusion_matrices_fold_{fold+1}.png', dpi=300)
    plt.close()


# ==========================================
# 5. LOOP DI K-FOLD CROSS-VALIDATION
# ==========================================
NUM_FOLDS = 3

all_folds_tr_loss, all_folds_va_loss = [], []
all_folds_tr_acc, all_folds_va_acc = [], []
fold_accuracies, fold_losses = [], []

fold_f1_scores = []
fold_precision_scores = []
fold_recall_scores = []

global_oof_true = []
global_oof_pred = []

global_train_true = []
global_train_pred = []

if RUN_TRAINING:
    for fold in range(NUM_FOLDS):
        fold_num = fold + 1
        print(f"\n==========================================")
        print(f"🔁 INIZIO FOLD {fold_num} / {NUM_FOLDS}")
        print(f"==========================================")

        train_dataset = DualStreamDataset(DATASET_ROOT, split_type='Training', fold_num=fold_num, training=True)
        train_eval_dataset = DualStreamDataset(DATASET_ROOT, split_type='Training', fold_num=fold_num, training=False)
        val_dataset = DualStreamDataset(DATASET_ROOT, split_type='Validation', fold_num=fold_num, training=False)

        print("Pazienti training:", len({pid for _, _, _, pid in train_dataset.samples}))
        print("Pazienti validation:", len({pid for _, _, _, pid in val_dataset.samples}))

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"❌ ERRORE CRITICO: Il Fold {fold_num} risulta vuoto.")
            continue

        train_dataset_full = train_dataset
        train_dataset = crea_train_subset_bilanciato(train_dataset_full, immagini_per_classe=6265, seed=42 + fold_num)
        print(f"\n📊 TRAIN ORIGINALE: {len(train_dataset_full):,} immagini")
        print(f"📊 TRAIN CAMPIONATO: {len(train_dataset):,} immagini")
        print(f"📊 VALIDATION: {len(val_dataset):,} immagini")

        targets_train = [train_dataset_full.samples[idx][2] for idx in train_dataset.indices]
        conteggio_classi = Counter(targets_train)
        print(f"\n📊 Distribuzione classi TRAIN CAMPIONATO: {dict(conteggio_classi)}")

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"❌ ERRORE CRITICO: Il Fold {fold_num} risulta vuoto.")
            continue

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, drop_last=True, prefetch_factor=2)
        train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        print("\n🔍 Verifica bilanciamento: ")
        for _, _, lbls, _ in train_loader:
            print("Distribuzione classi nel batch:", Counter(lbls.tolist()))
            break

        check_data_alignment(train_loader)

        try:
            accum_orig, accum_seg, accum_labels = [], [], []
            viz_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            for b_orig, b_seg, b_lbl, _ in viz_loader:
                accum_orig.append(b_orig)
                accum_seg.append(b_seg)
                accum_labels.append(b_lbl)
                if sum(len(x) for x in accum_orig) >= 16: break
            images_orig = torch.cat(accum_orig, dim=0)[:16]
            images_seg = torch.cat(accum_seg, dim=0)[:16]
            labels = torch.cat(accum_labels, dim=0)[:16]
            mean, std = np.array([0.4779, 0.4779, 0.4779]), np.array([0.4163, 0.4163, 0.4163])
            def denorm(t): return (t.cpu().numpy().transpose((1, 2, 0)) * std + mean).clip(0, 1)
            rows, cols = 8, 4
            fig, axes = plt.subplots(rows, cols, figsize=(12, 20))
            for r in range(rows):
                for c in range(cols):
                    img_idx = (r // 2) * cols + c
                    if img_idx >= len(images_orig):
                        axes[r, c].axis('off')
                        continue
                    ax = axes[r, c]
                    ax.axis('off')
                    img_o_denorm = denorm(images_orig[img_idx])
                    if r % 2 == 0:
                        ax.imshow(img_o_denorm)
                        ax.set_title(f"Orig: {train_dataset_full.classes[labels[img_idx]]}", color='black', fontsize=9)
                    else:
                        img_s_denorm = denorm(images_seg[img_idx])
                        mask = img_s_denorm[:, :, 0] > 0.05
                        cropped_cell = img_o_denorm * 0.10
                        cropped_cell[mask] = img_o_denorm[mask]
                        ax.set_facecolor('black')
                        ax.imshow(cropped_cell)
                        ax.set_title(f"Seg Cell: {train_dataset_full.classes[labels[img_idx]]}", color='black', fontsize=9)
            plt.tight_layout()
            plt.savefig('classes.png', dpi=300)
            plt.close()
            print("✅ Anteprima dei campioni salvata.")
        except Exception as e:
            print(f"⚠️ Errore nella generazione di classes.png: {e}")

        # ============================================================
        # MODELLO PRINCIPALE (CROSS-ATTENTION BIDIREZIONALE)
        # ============================================================
        model = TumorClassifierViT(num_classes=5).to(device)
        print("✅ DinoBloom-S: modello dual-stream con Cross-Attention bidirezionale caricato.")

        modelli_precedenti_totali = glob.glob("best_model_fold_*.pth")
        if modelli_precedenti_totali:
            print(f"✅ Trovati {len(modelli_precedenti_totali)} modelli precedenti su disco:")
            for mod_path in sorted(modelli_precedenti_totali):
                print(f"   - {mod_path}")
        else:
            print("❌ Nessun modello precedente trovato. Si parte da zero.")

        # ============================================================
        # LOSS: CROSS-ENTROPY con CLASS WEIGHTS
        # ============================================================
        class_weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=0.3
        )
        print(f"⚖️ Class weights: {CLASS_WEIGHTS}")
        print(f"⚖️ Label smoothing: 0.3")
        print(f"⚖️ Class thresholds: {CLASS_THRESHOLDS}")

        # ============================================================
        # OTTIMIZZATORE
        # ============================================================
        model_params = [p for p in model.parameters() if p.requires_grad]
        all_trainable_params = model_params

        print(f"\n📊 Parametri trainabili totali: {sum(p.numel() for p in all_trainable_params):,}")

        optimizer = torch.optim.AdamW(
            model_params,
            lr=INITIAL_LR,
            weight_decay=WEIGHT_DECAY
        )

        # ============================================================
        # SCHEDULER: ReduceLROnPlateau
        # ============================================================
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=8,
            min_lr=1e-7,
            threshold=1e-4,
            threshold_mode='rel',
            verbose=False
        )

        #scaler = torch.cuda.amp.GradScaler()
        scaler = torch.amp.GradScaler('cuda')
        best_val_loss = float('inf')
        patience = EARLY_STOPPING_PATIENCE
        min_delta = EARLY_STOPPING_MIN_DELTA
        patience_counter = 0

        tablePatients = PrettyTable(['Epoch', 'Time', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

        hist_train_loss_pat, hist_val_loss_pat = [], []
        hist_train_acc_pat, hist_val_acc_pat = [], []

        start_training_time = time.time()

        for epoch in range(MAX_EPOCHS):
            # --- PHASE 1: TRAINING ---
            model.train()
            optimizer.zero_grad()

            for batch_idx, (img_o, img_s, lbls, p_ids) in enumerate(tqdm(train_loader, desc=f'Fold {fold_num} - Epoch {epoch+1}/{MAX_EPOCHS}')):
                img_o, img_s, lbls_dev = img_o.to(device), img_s.to(device), lbls.to(device)

                with torch.amp.autocast('cuda'):
                    logits, features = model(img_o, img_s, return_features=True)
                    loss = criterion(logits, lbls_dev)
                    loss_scaled = loss / GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss_scaled).backward()

                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            # --- VALUTAZIONE TRAINING SET ---
            model.eval()
            train_eval_logits_list = []
            train_eval_labels_list = []
            train_eval_pids_list = []

            with torch.no_grad():
                for img_o, img_s, lbls, p_ids in train_eval_loader:
                    with torch.amp.autocast('cuda'):
                        out, _ = _forward_model_features(model, img_o, img_s, device)
                    train_eval_logits_list.append(out)
                    train_eval_labels_list.append(lbls.to(device))
                    train_eval_pids_list.extend(p_ids)

            tr_patient_logits_agg, tr_patient_true_labels, tr_preds = aggregate_patient_logits_weighted(
                train_eval_logits_list, train_eval_labels_list, train_eval_pids_list, device
            )

            loss_tr_pat = criterion(tr_patient_logits_agg, tr_patient_true_labels).item()
            acc_tr_pat = (tr_preds == tr_patient_true_labels).float().mean().item()

            # --- VALIDATION ---
            model.eval()
            val_logits_list, val_labels_list, val_pids_list = [], [], []

            with torch.no_grad():
                for img_o, img_s, lbls, p_ids in val_loader:
                    with torch.amp.autocast('cuda'):
                        out, _ = _forward_model_features(model, img_o, img_s, device)
                    val_logits_list.append(out)
                    val_labels_list.append(lbls.to(device))
                    val_pids_list.extend(p_ids)

            va_patient_logits_agg, va_patient_true_labels, val_preds = aggregate_patient_logits_weighted(
                val_logits_list, val_labels_list, val_pids_list, device
            )

            loss_va_pat = criterion(va_patient_logits_agg, va_patient_true_labels).item()
            acc_va_pat = (val_preds == va_patient_true_labels).float().mean().item()

            va_p_trues = va_patient_true_labels.cpu().numpy()
            va_p_preds = val_preds.cpu().numpy()

            scheduler.step(loss_va_pat)

            # 🩺 DIAGNOSTICA FALSI NEGATIVI SUI MALATI
            control_idx_v = 0
            malati_veri = (va_p_trues != control_idx_v)
            predetti_control = (va_p_preds == control_idx_v)
            falsi_negativi = malati_veri & predetti_control
            n_malati = int(malati_veri.sum())
            n_falsi_neg = int(falsi_negativi.sum())
            if n_malati > 0:
                print(f"🩺 Falsi negativi (malato → CONTROL) epoca {epoch+1}: {n_falsi_neg}/{n_malati} "
                      f"({n_falsi_neg/n_malati:.2%})")

            hist_train_loss_pat.append(loss_tr_pat)
            hist_val_loss_pat.append(loss_va_pat)
            hist_train_acc_pat.append(acc_tr_pat)
            hist_val_acc_pat.append(acc_va_pat)

            elapsed = int(time.time() - start_training_time)
            time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            tablePatients.add_row([epoch + 1, time_str, f"{loss_tr_pat:.3f}", f"{acc_tr_pat:.1%}", f"{loss_va_pat:.3f}", f"{acc_va_pat:.1%}"])

            print(f"\n🩺 [METRICHE LIVELLO PAZIENTE - FOLD {fold_num}]")
            print(tablePatients[-1], flush=True)

            genera_grafico_curva_fold(epoch + 1, fold, hist_train_loss_pat, hist_val_loss_pat, hist_train_acc_pat, hist_val_acc_pat)

            if loss_va_pat < best_val_loss - min_delta:
                best_val_loss = loss_va_pat
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold_num}.pth')
                print(f"💾 Salvato il modello migliore (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"⚠️ Nessun miglioramento ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"🛑 Early Stopping attivato al fold {fold_num}!")
                    break

        # Ricarica i pesi del modello migliore
        model.load_state_dict(torch.load(f'best_model_fold_{fold_num}.pth', map_location=device))

        # ============================================================
        # 🔄 METRICHE FINALI FOLD
        # ============================================================
        model.eval()

        train_eval_logits_list = []
        train_eval_labels_list = []
        train_eval_pids_list = []
        with torch.no_grad():
            for img_o, img_s, lbls, p_ids in train_eval_loader:
                with torch.amp.autocast('cuda'):
                    out, _ = _forward_model_features(model, img_o, img_s, device)
                train_eval_logits_list.append(out)
                train_eval_labels_list.append(lbls.to(device))
                train_eval_pids_list.extend(p_ids)

        tr_patient_logits_agg, tr_patient_true_labels, tr_preds = aggregate_patient_logits_weighted(
            train_eval_logits_list, train_eval_labels_list, train_eval_pids_list, device
        )
        tr_p_trues = tr_patient_true_labels.cpu().numpy()
        tr_p_preds = tr_preds.cpu().numpy()

        val_logits_list, val_labels_list, val_pids_list = [], [], []
        with torch.no_grad():
            for img_o, img_s, lbls, p_ids in val_loader:
                with torch.amp.autocast('cuda'):
                    out, _ = _forward_model_features(model, img_o, img_s, device)
                val_logits_list.append(out)
                val_labels_list.append(lbls.to(device))
                val_pids_list.extend(p_ids)

        va_patient_logits_agg, va_patient_true_labels, val_preds = aggregate_patient_logits_weighted(
            val_logits_list, val_labels_list, val_pids_list, device
        )
        va_p_trues = va_patient_true_labels.cpu().numpy()
        va_p_preds = val_preds.cpu().numpy()

        acc_tr_pat = accuracy_score(tr_p_trues, tr_p_preds)
        acc_va_pat = accuracy_score(va_p_trues, va_p_preds)

        global_oof_true.extend(va_p_trues)
        global_oof_pred.extend(va_p_preds)
        global_train_true.extend(tr_p_trues)
        global_train_pred.extend(tr_p_preds)

        # --- REPORT FINALE FOLD ---
        print(f"\n" + "="*75)
        print(f"📊 REPORT FINALE VALIDAZIONALI FOLD {fold_num} (LIVELLO PAZIENTE)")
        print(f"="*75)
        print(f"📈 Training Accuracy:   {acc_tr_pat * 100:.2f}%")
        print(f"📉 Validation Accuracy: {acc_va_pat * 100:.2f}%")
        print("-" * 75)

        present_classes = np.unique(np.concatenate([va_p_trues, va_p_preds]))
        active_target_names = [train_dataset_full.classes[i] for i in present_classes]
        print(classification_report(va_p_trues, va_p_preds, labels=present_classes, target_names=active_target_names, zero_division=0))
        print("="*75 + "\n")

        f1_macro = f1_score(va_p_trues, va_p_preds, average='macro', zero_division=0)
        precision_macro = precision_score(va_p_trues, va_p_preds, average='macro', zero_division=0)
        recall_macro = recall_score(va_p_trues, va_p_preds, average='macro', zero_division=0)

        print(f"\n" + "="*75)
        print(f"📊 REPORT METRICHE F1, Precision, Recall")
        print(f"="*75)
        print(f"🎯 Macro-F1        Fold {fold_num}: {f1_macro:.4f}")
        print(f"🎯 Macro-Precision Fold {fold_num}: {precision_macro:.4f}")
        print(f"🎯 Macro-Recall    Fold {fold_num}: {recall_macro:.4f}")

        fold_f1_scores.append(f1_macro)
        fold_precision_scores.append(precision_macro)
        fold_recall_scores.append(recall_macro)
        print("="*75 + "\n")

        fold_accuracies.append(acc_va_pat)
        fold_losses.append(best_val_loss)

        salva_matrici_confusione_fold(fold, train_dataset_full.classes, tr_p_trues, tr_p_preds, va_p_trues, va_p_preds)

        all_folds_tr_loss.append(hist_train_loss_pat)
        all_folds_va_loss.append(hist_val_loss_pat)
        all_folds_tr_acc.append(hist_train_acc_pat)
        all_folds_va_acc.append(hist_val_acc_pat)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"✅ Completato Fold {fold_num}!")

    # --- REPORT FINALE GLOBALE ---
    print("\n" + "="*75)
    print("📊 RIEPILOGO FINALE K-FOLD")
    print("="*75)
    print(f"Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Validation Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
    print(f"Macro-F1: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    print(f"Macro-Precision: {np.mean(fold_precision_scores):.4f} ± {np.std(fold_precision_scores):.4f}")
    print(f"Macro-Recall: {np.mean(fold_recall_scores):.4f} ± {np.std(fold_recall_scores):.4f}")
    print("="*75)

    print(f"\n" + "="*75)
    print(f"📈 RIEPILOGO STATISTICO GLOBALE K-FOLD")
    print(f"="*75)
    print(f" 🎯 Accuracy Globale Media : {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
    print(f" 📉 Loss Globale Media     : {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
    print("="*75 + "\n")

    print(f"="*75)
    print(f"🌐 REPORT FINALE GLOBALE OUT-OF-FOLD (OOF)")
    print(f"="*75)
    global_acc = accuracy_score(global_oof_true, global_oof_pred) * 100
    print(f" 🎯 Accuracy Globale OOF : {global_acc:.2f}%")
    print("-" * 75)

    global_present = np.unique(np.concatenate([global_oof_true, global_oof_pred]))
    global_target_names = [train_dataset_full.classes[i] for i in global_present]
    print(classification_report(global_oof_true, global_oof_pred, labels=global_present, target_names=global_target_names, zero_division=0))
    print("="*75 + "\n")

    plt.figure(figsize=(8, 6))
    cm_global = confusion_matrix(global_oof_true, global_oof_pred)
    sns.heatmap(cm_global, annot=True, fmt='d', cmap='Purples', xticklabels=train_dataset_full.classes, yticklabels=train_dataset_full.classes, cbar=False)
    plt.title(f'Global Out-Of-Fold Confusion Matrix (Acc: {global_acc:.2f}%)', fontsize=12, fontweight='bold')
    plt.xlabel('Diagnosi Predetta')
    plt.ylabel('Diagnosi Reale')
    plt.tight_layout()
    plt.savefig('confusion_matrix_global_oof.png', dpi=300)
    plt.close()
    print("✅ Matrice di confusione globale OOF salvata")

    # ============================================================
    # ✅ MATRICE DI CONFUSIONE FINALE SUL TRAINING SET
    # ============================================================
    print(f"\n" + "="*75)
    print(f"📊 MATRICE DI CONFUSIONE FINALE - TRAINING SET")
    print(f"="*75)

    train_acc_global = accuracy_score(global_train_true, global_train_pred) * 100
    print(f" 🎯 Training Accuracy Globale (tutti i fold): {train_acc_global:.2f}%")
    print("-" * 75)

    train_present = np.unique(np.concatenate([global_train_true, global_train_pred]))
    train_target_names = [train_dataset_full.classes[i] for i in train_present]
    print(classification_report(global_train_true, global_train_pred, labels=train_present, target_names=train_target_names, zero_division=0))
    print("="*75 + "\n")

    plt.figure(figsize=(8, 6))
    cm_train_final = confusion_matrix(global_train_true, global_train_pred)
    sns.heatmap(cm_train_final, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset_full.classes, yticklabels=train_dataset_full.classes, cbar=False)
    plt.title(f'Global Training Confusion Matrix (Acc: {train_acc_global:.2f}%)', fontsize=12, fontweight='bold')
    plt.xlabel('Diagnosi Predetta')
    plt.ylabel('Diagnosi Reale')
    plt.tight_layout()
    plt.savefig('confusion_matrix_global_training.png', dpi=300)
    plt.close()
    print("✅ Matrice di confusione globale Training salvata")

    # --- GRAFICO CURVE DI APPRENDIMENTO ---
    max_len = max((len(x) for x in all_folds_tr_loss), default=0)

    if max_len > 0:
        padded_tr_loss = [x + [x[-1]]*(max_len - len(x)) for x in all_folds_tr_loss]
        padded_va_loss = [x + [x[-1]]*(max_len - len(x)) for x in all_folds_va_loss]
        padded_tr_acc  = [x + [x[-1]]*(max_len - len(x)) for x in all_folds_tr_acc]
        padded_va_acc  = [x + [x[-1]]*(max_len - len(x)) for x in all_folds_va_acc]

        mean_tr_loss = np.mean(padded_tr_loss, axis=0)
        mean_va_loss = np.mean(padded_va_loss, axis=0)
        mean_tr_acc = np.mean(padded_tr_acc, axis=0)
        mean_va_acc = np.mean(padded_va_acc, axis=0)
    else:
        mean_tr_loss = mean_va_loss = mean_tr_acc = mean_va_acc = np.array([])

    if max_len > 0:
        epochs_range = range(1, max_len + 1)
        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, mean_tr_loss, label='Mean Train Loss', color='#1f77b4', linewidth=2.5)
        plt.plot(epochs_range, mean_va_loss, label='Mean Val Loss', color='#ff7f0e', linewidth=2.5)
        plt.title('Overall K-Fold Average Patient Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoche'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3); plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, mean_tr_acc, label='Mean Train Acc', color='#1f77b4', linewidth=2.5)
        plt.plot(epochs_range, mean_va_acc, label='Mean Val Acc', color='#ff7f0e', linewidth=2.5)
        plt.title('Overall K-Fold Average Patient Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoche'); plt.ylabel('Accuracy'); plt.grid(True, alpha=0.3); plt.legend()

        plt.tight_layout()
        plt.savefig('overall_kfold_learning_curves.png', dpi=300)
        plt.close()


# ==========================================
# 6. VALUTAZIONE FINALE SUL TEST SET ESTERNO (MEDIA DEI MODELLI)
# ==========================================
print(f"\n" + "="*75)
print(f"🧪 INIZIO VALUTAZIONE SUL TEST SET ESTERNO")
print(f"="*75)

TEST_SET_DIR = './dataset_split_con_test/Test_Set'

if os.path.exists(TEST_SET_DIR):
    class TestDualStreamDataset(Dataset):
        def __init__(self, test_root):
            self.samples = []
            self.classes = ['CONTROL', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            orig_root = os.path.join(test_root, "Original")
            seg_root = os.path.join(test_root, "Segmented")

            if os.path.exists(orig_root):
                for cls in os.listdir(orig_root):
                    cls_orig_dir = os.path.join(orig_root, cls)
                    cls_seg_dir = os.path.join(seg_root, cls)

                    if os.path.isdir(cls_orig_dir) and cls in self.class_to_idx:
                        pazienti = [p for p in os.listdir(cls_orig_dir) if os.path.isdir(os.path.join(cls_orig_dir, p))]
                        for p_id in pazienti:
                            p_orig_path = os.path.join(cls_orig_dir, p_id)
                            p_seg_path = os.path.join(cls_seg_dir, p_id)
                            if os.path.exists(p_orig_path):
                                for fname in os.listdir(p_orig_path):
                                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        p_orig = os.path.join(p_orig_path, fname)
                                        p_seg = os.path.join(p_seg_path, fname)
                                        if os.path.exists(p_seg):
                                            unique_patient_id = f"{cls}___{p_id}"
                                            self.samples.append((p_orig, p_seg, self.class_to_idx[cls], unique_patient_id))

            print(f"✅ Trovati {len(self.samples)} campioni nel Test Set esterno.")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            p_orig, p_seg, label, patient_id = self.samples[idx]
            img_o = Image.open(p_orig).convert('RGB')
            img_s = Image.open(p_seg).convert('L')
            res = (IMAGE_RESOLUTION, IMAGE_RESOLUTION)

            img_o = TF.resize(img_o, res)
            img_s = TF.resize(img_s, res).convert('RGB')

            o_tensor = TF.to_tensor(img_o)
            s_tensor = TF.to_tensor(img_s)
            norm = transforms.Normalize(mean=[0.4779, 0.4779, 0.4779], std=[0.4163, 0.4163, 0.4163])

            return norm(o_tensor), norm(s_tensor), label, patient_id

    test_dataset = TestDualStreamDataset(TEST_SET_DIR)
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        loaded_models = []

        for fold_idx in range(1, NUM_FOLDS + 1):
            m_path = f"best_model_fold_{fold_idx}.pth"

            if not os.path.exists(m_path):
                print(f"⚠️ Modello del Fold {fold_idx} non trovato, salto.")
                continue

            m = TumorClassifierViT(num_classes=5).to(device)
            m.load_state_dict(torch.load(m_path, map_location=device))
            m.eval()
            loaded_models.append(m)

        if len(loaded_models) > 0:
            print(f"🤖 Inferenza con {len(loaded_models)} modelli (uno per fold)...")
            test_logits_list, test_labels_list, test_pids_list = [], [], []

            with torch.no_grad():
                for img_o, img_s, lbls, p_ids in test_loader:
                    img_o, img_s, lbls_dev = img_o.to(device), img_s.to(device), lbls.to(device)

                    batch_logits = None
                    for model_idx, m in enumerate(loaded_models):
                        logits, features = _forward_model_features(m, img_o, img_s, device)
                        logits = logits.float()

                        if batch_logits is None:
                            batch_logits = logits
                        else:
                            batch_logits += logits

                    batch_logits /= len(loaded_models)

                    test_logits_list.append(batch_logits)
                    test_labels_list.append(lbls_dev)
                    test_pids_list.extend(p_ids)

            test_logits_tensor = torch.cat(test_logits_list, dim=0).float()
            test_labels_tensor = torch.cat(test_labels_list, dim=0)

            unique_test_pids, test_inverse = torch.unique(
                torch.tensor([hash(p) for p in test_pids_list], device=device), return_inverse=True
            )

            test_probs = torch.softmax(test_logits_tensor, dim=1)
            test_confidences, _ = torch.max(test_probs, dim=1)
            test_weights = test_confidences.unsqueeze(1)

            num_test_patients = len(unique_test_pids)
            test_patient_logits_agg = torch.zeros((num_test_patients, test_logits_tensor.shape[1]), device=device)
            test_weighted_logits = test_logits_tensor * test_weights

            test_patient_logits_agg.index_add_(0, test_inverse, test_weighted_logits)
            test_sum_weights = torch.zeros((num_test_patients, 1), device=device)
            test_sum_weights.index_add_(0, test_inverse, test_weights)
            test_patient_logits_agg = test_patient_logits_agg / (test_sum_weights + 1e-6)

            test_patient_true_labels = torch.zeros(num_test_patients, dtype=torch.long, device=device)
            test_patient_true_labels.index_copy_(0, test_inverse, test_labels_tensor)

            test_probs_patient = torch.softmax(test_patient_logits_agg, dim=1)
            thresholds_test = torch.tensor(CLASS_THRESHOLDS, device=device)
            adjusted_probs_test = test_probs_patient / thresholds_test
            test_preds = adjusted_probs_test.argmax(dim=1)

            criterion_test = torch.nn.CrossEntropyLoss(label_smoothing=0.20)
            test_loss_pat = criterion_test(test_patient_logits_agg, test_patient_true_labels).item()

            test_acc_pat_dec = (test_preds == test_patient_true_labels).float().mean().item()
            test_acc_pat_perc = test_acc_pat_dec * 100

            t_trues = test_patient_true_labels.cpu().numpy()
            t_preds = test_preds.cpu().numpy()

            print(f"\n===========================================================================")
            print(f"🏆 RISULTATI FINALI SUL TEST SET ESTERNO")
            print(f"===========================================================================")
            print(f" 📉 Test Loss (Pazienti)     : {test_loss_pat:.4f}")
            print(f" 🎯 Test Accuracy (Pazienti) : {test_acc_pat_perc:.2f}%")
            print("-" * 75)
            print(classification_report(t_trues, t_preds, target_names=test_dataset.classes, zero_division=0))
            print("="*75 + "\n")

            # 🩺 Falsi negativi sul test set
            control_idx_t = 0
            malati_veri_t = (t_trues != control_idx_t)
            predetti_control_t = (t_preds == control_idx_t)
            fn_test = (malati_veri_t & predetti_control_t).sum()
            n_malati_t = malati_veri_t.sum()
            print(f"🩺 Falsi negativi sul Test Set: {fn_test}/{n_malati_t} ({fn_test/max(n_malati_t,1):.2%})")

            plt.figure(figsize=(8, 6))
            cm_test = confusion_matrix(t_trues, t_preds)
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cbar=False)
            plt.title(f'Patient-Level Test Confusion Matrix\n(Acc: {test_acc_pat_perc:.2f}%)', fontsize=11, fontweight='bold')
            plt.xlabel('Diagnosi Predetta')
            plt.ylabel('Diagnosi Reale')
            plt.tight_layout()
            plt.savefig('confusion_matrix_external_test_patients.png', dpi=300)
            plt.close()
            print("✅ Matrice di confusione test salvata")
        else:
            print("⚠️ Nessun modello salvato trovato per la valutazione finale.")
    else:
        print(f"⚠️ Test Set vuoto o non valido.")
else:
    print(f"⚠️ Test Set non trovato.")

print("\n" + "="*75)
print("🧠 TRAINING COMPLETATO")
print("🏁 DUAL-STREAM CROSS-ATTENTION BIDIREZIONALE")
print("="*75)
