<div align="center">

# Hybrid AI Flood Prediction System

### Text + numeric hybrid model with a real-time monitoring deployment

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![BERT](https://img.shields.io/badge/BERT-Text_Features-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)

**A hybrid flood-risk model that fuses contextual text features (BERT + PCA) with scaled numerical weather data — plus a production script that polls live weather APIs and fires notifications on high risk.**

</div>

---

## Two Parts

```mermaid
flowchart TD
    subgraph DEV["Part 1 · Model Development (/notebooks)"]
        RAW["Weather + context data"] --> TXT["Generate Text_Description feature<br/>→ BERT embeddings → PCA"]
        RAW --> NUM["Scale numerical features"]
        TXT --> BAL["SMOTE class balancing"]
        NUM --> BAL
        BAL --> RF["RandomForestClassifier<br/>→ flood_model.pkl + scaler.pkl + pca.pkl"]
    end

    subgraph DEP["Part 2 · Real-Time Deployment (/deployment)"]
        RF --> MON["flood_monitor_production.py<br/>hourly schedule"]
        MON --> API["Fetch live weather per city"]
        API --> PRED["Predict flood severity"]
        PRED --> NOTIF{"High risk?"}
        NOTIF -- "yes" --> PUSH["Push notification via backend service"]
    end
```

## Part 1 — Model Development
- `1_Data_Exploration_and_Cleaning.ipynb` — analysis, the contextual `Text_Description` feature, BERT + PCA text processing, numeric scaling, SMOTE balancing
- `2_Model_Training_and_Evaluation.ipynb` — trains and evaluates the `RandomForestClassifier`
- Saves `flood_model.pkl`, `scaler.pkl`, `pca.pkl` to `/models`

## Part 2 — Real-Time Deployment
`deployment/flood_monitor_production.py` runs on a schedule, fetches live weather for a list of cities, predicts severity with the trained models, and sends a notification when high risk is detected.

> The deployment script is coupled to a project-specific weather/notification API and serves as a proof-of-concept for integrating the core model into a live system.

## Setup

```bash
git clone https://github.com/YazanAi-Dev3/flood-prediction-system.git
cd flood-prediction-system
pip install -r requirements.txt
jupyter lab      # explore /notebooks
```

## Tech Stack

`Python` · `scikit-learn` (RandomForest, SMOTE) · `BERT` + `PCA` (text features) · `Pandas`
