# Classificazione della Leucemia Mieloide Acuta da immagini

Framework avanzato di deep learning progettato per la classificazione automatizzata dei sottotipi e l'analisi guidata da segmentazione di cellule di Leucemia Mieloide Acuta (LAM). Sfruttando un'architettura Dual-Stream Vision Transformer (ViT) con attenzione incrociata bidirezionale, il modello elabora simultaneamente immagini microscopiche cellulari grezze e maschere binarie automatizzate per effettuare diagnosi cliniche affidabili a livello di paziente su 5 distinte classi genetiche e di controllo.


---

## 📂 Struttura della Repository

La repository è organizzata in moduli dedicati alle diverse fasi di sperimentazione, documentazione e confronto metodologico:

```text
├── codice1/                            # Sperimentazione Primaria (Configurazione $K=3$ Fold)
│   ├── transformer.py                  # Architettura principale Dual-Stream Vision Transformer
│   ├── process_dataset.py              # Segmentazione cellulare e sogliatura automatica nello spazio colore LAB
│   ├── prepare_data.py                 # Split Stratified Group K-Fold basato sull'ID del paziente
│   ├── distribuzione_dataset_*.py      # Script per l'analisi statistica e della distribuzione delle classi
│   ├── cleanup.py                      # Rimozione duplicati e sincronizzazione tramite hash MD5
│   ├── train.py                        # Pipeline di addestramento con pesatura delle classi ed Early Stopping
│   ├── calcola_roc_auc.py              # Valutazione delle metriche ROC-AUC (Spec+Sens) e PR-AUC
│   └── test.py                         # Script di inferenza e diagnosi a livello di paziente (con Ensemble)
├── codice2/                            # Sperimentazione Secondaria (Configurazione $K=5$ Fold)
│   ├── transformer.py                  # Architettura Dual-Stream ViT adattata per 5 Folds
│   ├── process_dataset.py              # Pipeline di segmentazione cellulare LAB
│   ├── prepare_data.py                 # Split Stratified Group K-Fold ($K=5$) con Holdout
│   ├── distribuzione_dataset_*.py      # Script per l'analisi statistica delle classi
│   ├── cleanup.py                      # Pulizia e verifica integrità campioni (MD5)
│   ├── train.py                        # Pipeline di addestramento $K=5$ con pesatura e checkpointing
│   ├── calcola_roc_auc.py              # Calcolo metriche ROC-AUC e PR-AUC per 5 Folds
│   └── test.py                         # Inferenza Ensemble multi-fold a livello di paziente
├── documentazione/                     # Elaborato della Tesi Magistrale e Slide di Presentazione

```

---

## 🔬 Classi Supportate

Il modello classifica le immagini microscopiche cellulari in **5 distinte categorie biologiche**:
1. **CONTROL** (Donatori Sani)
2. **CBFB_MYH11** (LAM con $\text{inv}(16)$)
3. **NPM1** (LAM con mutazione $NPM1$)
4. **PML_RARA** (Leucemia Promielocitica Acuta)
5. **RUNX1_RUNX1T1** (LAM con $t(8;21)$)

---

## ⚙️ Caratteristiche Principali dell'Architettura

* **Elaborazione a Doppio Flusso (Dual-Stream):** Elabora in parallelo la morfologia cellulare originale e le maschere della regione di interesse (ROI) attraverso percorsi transformer indipendenti.
* **Aggregazione a Livello di Paziente:** Combina le predizioni delle singole cellule per ciascun paziente utilizzando un sistema di voto ponderato sulla confidenza per generare diagnosi cliniche stabili.
* **Suddivisione Disgiunta per Paziente:** Utilizza `StratifiedGroupKFold` per garantire la totale assenza di data leakage dei campioni dei pazienti tra i set di addestramento, validazione e test.
* **Inferenza Ensemble:** Supporta l'integrazione multi-modello attraverso i checkpoint dei K-fold (`best_model_fold_*.pth`) per migliorare la generalizzazione sui dati di test esterni.

---

## 🚀 Guida all'Uso

### Requisiti e Installazione

Clona la repository e installa le dipendenze necessarie:

```bash
git clone https://github.com/tuo-username/dinobloom-s.git
cd dinobloom-s
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy matplotlib scikit-learn prettytable seaborn tqdm pandas
```

### 1. Download del Dataset e Preprocessamento

Prima di eseguire gli script di elaborazione, è necessario scaricare il dataset biologico ufficiale:

1. Scarica il dataset pubblico AML-Cytomorphology_MLL_Helmholtz dal portale The Cancer Imaging Archive (TCIA):  https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/
2. Estrarre le immagini grezze e posizionarle nella cartella ./data all'interno della directory di lavoro (codice1/data/ e codice2/data/).
3. Eseguire la pipeline di segmentazione e suddivisione dei dati:

Prima di eseguire gli script di elaborazione, è necessario scaricare il dataset biologico ufficiale e rinominare la cartella con `./data` e control con CONTROL poi eseguire:


```bash
# Genera le maschere cellulari ad alto contrasto nello spazio colore LAB
python process_dataset.py

# Rimuovi eventuali duplicati fisici tramite hash MD5
python cleanup.py

# Crea gli split Stratified Group K-Fold con un Test Set isolato al 20%
python prepare_data.py
```

### 2. Addestramento del Modello
Avvia lo script di addestramento per iniziare la validazione incrociata con pesatura delle classi ed Early Stopping:

```bash
python train.py
```

### 3. Valutazione delle Metriche (ROC-AUC e PR-AUC)
Calcola le prestazioni Out-of-Fold sul set di validazione e sul set di test formattate come `Spec+Sens / Prec+Rec`:

```bash
python calcola_roc_auc.py
```

### 4. Inferenza a Livello di Paziente
Esegui l'inferenza diagnostica Ensemble su pazienti casuali o specifici dal set di test:

```bash
python test.py
```

---

## 📊 Output di Valutazione e Grafici

Il framework di valutazione esporta automaticamente report diagnostici completi, tra cui:
* **Matrici di Confusione** e **Istogrammi di Distribuzione della Confidenza** a livello di paziente
* **Curve ROC** (True Positive Rate vs. False Positive Rate)
* **Curve Precision-Recall** (Precision vs. Sensitivity)
* File CSV riassuntivi (`probabilita_test.csv`, `probabilita_validation.csv`)
