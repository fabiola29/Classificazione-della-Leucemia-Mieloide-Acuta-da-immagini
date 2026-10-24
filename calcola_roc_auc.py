"""
Calcola ROC-AUC (Spec+Sens) e Precision-Recall AUC (Prec+Rec) 
per il modello addestrato.

Calcola le metriche su:
1. Validation Set (out-of-fold, 3 fold)
2. Test Set (ensemble dei 3 modelli)

Formato output identico a SCEMILA: "Spec+Sens / Prec+Rec"
"""
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from collections import defaultdict
from tqdm import tqdm

from transformer import TumorClassifierViT

# ==========================================
# CONFIGURAZIONE
# ==========================================
DATASET_ROOT = '../dataset_split_con_test'
IMAGE_RESOLUTION = 224
BATCH_SIZE = 16
NUM_FOLDS = 3

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"🚀 Device: {device}")


# ==========================================
# DATASET
# ==========================================
class DualStreamDataset(Dataset):
    def __init__(self, base_path, split_type='Validation', fold_num=1):
        self.samples = []
        self.classes = ['CONTROL', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        target_dir = os.path.join(base_path, f"Fold_{fold_num}", split_type)
        orig_root = os.path.join(target_dir, "Original")
        seg_root = os.path.join(target_dir, "Segmented")

        if os.path.exists(orig_root):
            for cls in os.listdir(orig_root):
                cls_orig_dir = os.path.join(orig_root, cls)
                cls_seg_dir = os.path.join(seg_root, cls)

                if os.path.isdir(cls_orig_dir) and cls in self.class_to_idx:
                    pazienti = [p for p in os.listdir(cls_orig_dir)
                                if os.path.isdir(os.path.join(cls_orig_dir, p))]

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

        print(f"✅ {len(self.samples)} campioni caricati (Fold {fold_num} - {split_type})")

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
        norm = transforms.Normalize(mean=[0.4779, 0.4779, 0.4779],
                                     std=[0.4163, 0.4163, 0.4163])

        return norm(o_tensor), norm(s_tensor), label, patient_id


# ==========================================
# ✅ DATASET TEST SET (NUOVO)
# ==========================================
class TestDualStreamDataset(Dataset):
    """Dataset per il Test Set esterno (non diviso in fold)."""
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
                    pazienti = [p for p in os.listdir(cls_orig_dir) 
                                if os.path.isdir(os.path.join(cls_orig_dir, p))]
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
                                        self.samples.append((p_orig, p_seg, 
                                                             self.class_to_idx[cls], 
                                                             unique_patient_id))

        print(f"✅ {len(self.samples)} campioni caricati (Test Set)")

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
        norm = transforms.Normalize(mean=[0.4779, 0.4779, 0.4779],
                                     std=[0.4163, 0.4163, 0.4163])

        return norm(o_tensor), norm(s_tensor), label, patient_id


# ==========================================
# ESTRAZIONE PROBABILITÀ - VALIDATION
# ==========================================
def estrai_probabilita_fold(fold_num, model):
    """Estrae le probabilità di tutti i pazienti per un fold di validation."""
    val_dataset = DualStreamDataset(DATASET_ROOT, split_type='Validation', fold_num=fold_num)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=4)

    model.eval()
    patient_logits = defaultdict(list)

    with torch.no_grad():
        for img_o, img_s, lbls, p_ids in tqdm(val_loader, desc=f'Fold {fold_num}'):
            img_o = img_o.to(device)
            img_s = img_s.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, _ = model(img_o, img_s, return_features=True)

            logits = logits.float().cpu().numpy()

            for i, pid in enumerate(p_ids):
                patient_logits[pid].append(logits[i])

    return aggrega_probabilita(patient_logits, val_dataset, fold_num)


# ==========================================
# ✅ ESTRAZIONE PROBABILITÀ - TEST SET (NUOVO)
# ==========================================
def estrai_probabilita_test(models):
    """
    Estrae le probabilità di tutti i pazienti del Test Set esterno.
    Usa l'ensemble dei modelli (media dei logits).
    """
    test_dir = os.path.join(DATASET_ROOT, 'Test_Set')
    
    if not os.path.exists(test_dir):
        print(f"❌ Test Set non trovato: {test_dir}")
        return []
    
    test_dataset = TestDualStreamDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=4)

    patient_logits = defaultdict(list)

    with torch.no_grad():
        for img_o, img_s, lbls, p_ids in tqdm(test_loader, desc='Test Set'):
            img_o = img_o.to(device)
            img_s = img_s.to(device)

            # ✅ Ensemble: media dei logits di tutti i modelli
            batch_logits = None
            for model in models:
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits, _ = model(img_o, img_s, return_features=True)
                
                logits = logits.float()
                if batch_logits is None:
                    batch_logits = logits
                else:
                    batch_logits += logits
            
            batch_logits /= len(models)
            batch_logits = batch_logits.cpu().numpy()

            for i, pid in enumerate(p_ids):
                patient_logits[pid].append(batch_logits[i])

    return aggrega_probabilita(patient_logits, test_dataset, fold_num='Test')


# ==========================================
# ✅ FUNZIONE DI AGGREGAZIONE (condivisa)
# ==========================================
def aggrega_probabilita(patient_logits, dataset, fold_num):
    """Aggrega i logits per paziente con media pesata per confidenza."""
    results = []
    for pid, logits_list in patient_logits.items():
        logits_array = np.array(logits_list)  # (N, 5)

        # Softmax
        probs = np.exp(logits_array) / np.exp(logits_array).sum(axis=1, keepdims=True)

        # Confidenza
        conf = probs.max(axis=1, keepdims=True)

        # Media pesata
        weighted_logits = logits_array * conf
        sum_weights = conf.sum(axis=0)
        patient_logits_agg = weighted_logits.sum(axis=0) / (sum_weights + 1e-6)

        # Softmax finale
        patient_probs = np.exp(patient_logits_agg) / np.exp(patient_logits_agg).sum()

        # Estrai label vera
        true_label = dataset.class_to_idx[pid.split('___')[0]]

        results.append({
            'patient_id': pid,
            'fold': fold_num,
            'true_label': true_label,
            'true_class': dataset.classes[true_label],
            'prob_CONTROL': patient_probs[0],
            'prob_CBFB_MYH11': patient_probs[1],
            'prob_NPM1': patient_probs[2],
            'prob_PML_RARA': patient_probs[3],
            'prob_RUNX1_RUNX1T1': patient_probs[4],
            'n_images': len(logits_list)
        })

    return results


# ==========================================
# CALCOLO ROC-AUC E PR-AUC
# ==========================================
def calcola_roc_e_pr_auc(all_results):
    """Calcola ROC-AUC (Spec+Sens) e PR-AUC (Prec+Rec) per classe e macro."""
    probs = np.array([[r['prob_CONTROL'], r['prob_CBFB_MYH11'], r['prob_NPM1'],
                        r['prob_PML_RARA'], r['prob_RUNX1_RUNX1T1']] for r in all_results])
    true_labels = np.array([r['true_label'] for r in all_results])

    class_names = ['CONTROL', 'CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
    n_classes = len(class_names)

    true_bin = label_binarize(true_labels, classes=range(n_classes))

    roc_auc_per_class = {}
    pr_auc_per_class = {}

    for i, cls in enumerate(class_names):
        try:
            roc_auc_per_class[cls] = roc_auc_score(true_bin[:, i], probs[:, i])
        except Exception as e:
            roc_auc_per_class[cls] = None
            print(f"⚠️ Errore ROC-AUC per {cls}: {e}")

        try:
            pr_auc_per_class[cls] = average_precision_score(true_bin[:, i], probs[:, i])
        except Exception as e:
            pr_auc_per_class[cls] = None
            print(f"⚠️ Errore PR-AUC per {cls}: {e}")

    try:
        roc_auc_macro = roc_auc_score(true_bin, probs, average='macro', multi_class='ovr')
    except Exception:
        roc_auc_macro = None

    pr_auc_macro = np.mean([v for v in pr_auc_per_class.values() if v is not None])

    try:
        roc_auc_weighted = roc_auc_score(true_bin, probs, average='weighted', multi_class='ovr')
    except Exception:
        roc_auc_weighted = None

    class_support = true_bin.sum(axis=0)
    pr_auc_weighted = np.average(
        [v for v in pr_auc_per_class.values() if v is not None],
        weights=[class_support[i] for i, v in enumerate(pr_auc_per_class.values()) if v is not None]
    )

    return (roc_auc_per_class, pr_auc_per_class,
            roc_auc_macro, pr_auc_macro,
            roc_auc_weighted, pr_auc_weighted,
            probs, true_labels, class_names)


# ==========================================
# GRAFICI
# ==========================================
def plot_roc_curves(probs, true_labels, class_names, save_path='roc_curves.png'):
    """Genera il grafico delle curve ROC standard (Spec+Sens)."""
    n_classes = len(class_names)
    true_bin = label_binarize(true_labels, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{cls} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves (Spec+Sens)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Grafico ROC salvato: {save_path}")


def plot_pr_curves(probs, true_labels, class_names, save_path='pr_curves.png'):
    """Genera il grafico delle curve Precision-Recall (Prec+Rec)."""
    n_classes = len(class_names)
    true_bin = label_binarize(true_labels, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(true_bin[:, i], probs[:, i])
        ap = average_precision_score(true_bin[:, i], probs[:, i])
        plt.plot(recall, precision, color=colors[i], lw=2,
                 label=f'{cls} (AP = {ap:.3f})')

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall Curves (Prec+Rec)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Grafico PR salvato: {save_path}")


# ==========================================
# ✅ FUNZIONE PER STAMPARE I RISULTATI
# ==========================================
def stampa_risultati(roc_auc_per_class, pr_auc_per_class, 
                     roc_auc_macro, pr_auc_macro,
                     roc_auc_weighted, pr_auc_weighted,
                     class_names, titolo):
    """Stampa i risultati in formato SCEMILA."""
    print(f"\n{'='*75}")
    print(f"📊 {titolo} - ROC-AUC (Spec+Sens / Prec+Rec)")
    print(f"{'='*75}")
    print(f"{'Classe':<20} {'Spec+Sens':>12} {'Prec+Rec':>12}")
    print(f"{'-'*75}")
    
    for cls in class_names:
        roc_val = roc_auc_per_class.get(cls)
        pr_val = pr_auc_per_class.get(cls)
        if roc_val is not None and pr_val is not None:
            print(f"{cls:<20} {roc_val:>12.4f} {pr_val:>12.4f}")
        else:
            print(f"{cls:<20} {'N/A':>12} {'N/A':>12}")
    
    print(f"{'-'*75}")
    print(f"{'MACRO':<20} {roc_auc_macro:>12.4f} {pr_auc_macro:>12.4f}")
    print(f"{'WEIGHTED':<20} {roc_auc_weighted:>12.4f} {pr_auc_weighted:>12.4f}")
    print(f"{'='*75}\n")


# ==========================================
# MAIN
# ==========================================
def main():
    # ==========================================
    # CARICA I 3 MODELLI
    # ==========================================
    models = []
    for fold_num in range(1, NUM_FOLDS + 1):
        m_path = f"best_model_fold_{fold_num}.pth"
        if not os.path.exists(m_path):
            print(f"⚠️ Modello {m_path} non trovato, salto.")
            continue
        
        m = TumorClassifierViT(num_classes=5).to(device)
        m.load_state_dict(torch.load(m_path, map_location=device))
        m.eval()
        models.append(m)
        print(f"✅ Caricato {m_path}")
    
    if len(models) == 0:
        print("❌ Nessun modello caricato!")
        return

    # ==========================================
    # 1. VALIDATION SET (out-of-fold)
    # ==========================================
    print(f"\n{'='*75}")
    print(f"📊 CALCOLO SU VALIDATION SET (out-of-fold)")
    print(f"{'='*75}")
    
    all_results_val = []
    for fold_num in range(1, NUM_FOLDS + 1):
        if fold_num > len(models):
            continue
        results = estrai_probabilita_fold(fold_num, models[fold_num - 1])
        all_results_val.extend(results)
    
    (roc_auc_val, pr_auc_val,
     roc_macro_val, pr_macro_val,
     roc_weighted_val, pr_weighted_val,
     probs_val, true_val, class_names) = calcola_roc_e_pr_auc(all_results_val)
    
    stampa_risultati(roc_auc_val, pr_auc_val,
                     roc_macro_val, pr_macro_val,
                     roc_weighted_val, pr_weighted_val,
                     class_names, "VALIDATION SET")
    
    plot_roc_curves(probs_val, true_val, class_names, 
                    save_path='roc_curves_validation.png')
    plot_pr_curves(probs_val, true_val, class_names, 
                   save_path='pr_curves_validation.png')

    # ==========================================
    # 2. TEST SET (ensemble dei 3 modelli)
    # ==========================================
    print(f"\n{'='*75}")
    print(f"📊 CALCOLO SU TEST SET (ensemble dei modelli)")
    print(f"{'='*75}")
    
    all_results_test = estrai_probabilita_test(models)
    
    if len(all_results_test) == 0:
        print("❌ Nessun paziente trovato nel Test Set!")
        return
    
    (roc_auc_test, pr_auc_test,
     roc_macro_test, pr_macro_test,
     roc_weighted_test, pr_weighted_test,
     probs_test, true_test, class_names) = calcola_roc_e_pr_auc(all_results_test)
    
    stampa_risultati(roc_auc_test, pr_auc_test,
                     roc_macro_test, pr_macro_test,
                     roc_weighted_test, pr_weighted_test,
                     class_names, "TEST SET")
    
    plot_roc_curves(probs_test, true_test, class_names, 
                    save_path='roc_curves_test.png')
    plot_pr_curves(probs_test, true_test, class_names, 
                   save_path='pr_curves_test.png')

    # ==========================================
    # 3. CONFRONTO FINALE
    # ==========================================
    print(f"\n{'='*75}")
    print(f"📊 CONFRONTO VALIDATION vs TEST")
    print(f"{'='*75}")
    print(f"{'Metrica':<20} {'Validation':>12} {'Test':>12}")
    print(f"{'-'*75}")
    print(f"{'ROC-AUC Macro':<20} {roc_macro_val:>12.4f} {roc_macro_test:>12.4f}")
    print(f"{'PR-AUC Macro':<20} {pr_macro_val:>12.4f} {pr_macro_test:>12.4f}")
    print(f"{'ROC-AUC Weighted':<20} {roc_weighted_val:>12.4f} {roc_weighted_test:>12.4f}")
    print(f"{'PR-AUC Weighted':<20} {pr_weighted_val:>12.4f} {pr_weighted_test:>12.4f}")
    print(f"{'='*75}\n")

    # ==========================================
    # 4. SALVA CSV
    # ==========================================
    import pandas as pd
    pd.DataFrame(all_results_test).to_csv('probabilita_test.csv', index=False)
    pd.DataFrame(all_results_val).to_csv('probabilita_validation.csv', index=False)
    print(f"✅ Salvati: probabilita_test.csv, probabilita_validation.csv")


if __name__ == '__main__':
    main()
