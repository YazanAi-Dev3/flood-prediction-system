#  Hybrid AI Flood Prediction System

This repository contains the complete lifecycle of a hybrid AI model for flood prediction, from data analysis and model training to a real-time deployment script.

---

##  Project Structure

This project is divided into two main parts:

1.  **Model Development (`/notebooks`)**: This part covers the core machine learning workflow, including data analysis, feature engineering, training, and evaluation. It is self-contained and fully reproducible.
2.  **Deployment (`/deployment`)**: This part contains a Python script that demonstrates how the trained model can be used in a real-world scenario by connecting to a live API for real-time monitoring and notifications.

---

##  Part 1: Model Development & Training

The entire process of building the AI model is documented in the Jupyter Notebooks inside the `/notebooks` directory.

* **`1_Data_Exploration_and_Cleaning.ipynb`**: Covers the initial data loading, analysis, visualization, and cleaning steps.
* **`2_Model_Training_and_Evaluation.ipynb`**: Details the full hybrid pipeline:
    * Generating the contextual `Text_Description` feature.
    * Processing text with BERT and PCA.
    * Scaling numerical features.
    * Balancing the dataset using SMOTE.
    * Training the final `RandomForestClassifier`.
    * Evaluating the model's performance on a test set.

The final trained model (`flood_model.pkl`) and the data processors (`scaler.pkl`, `pca.pkl`) are saved in the `/models` directory.

---

##  Part 2: Real-Time Deployment Script

The `/deployment` directory contains the script `flood_monitor_production.py`, which serves as a practical application of the trained model.

### How It Works
- The script runs on a configurable schedule (e.g., every hour).
- It fetches real-time weather data for a list of cities from a specific backend API.
- It uses the pre-trained models from the `/models` directory to make a flood severity prediction.
- If a high risk is detected, it sends a notification through the backend's notification service.

**Important Note:** This script is tightly coupled with a project-specific API (`https://std.scit.co/flood/...`) that may not be available long-term. It serves as a proof-of-concept for how the core model can be integrated into a live system.

### Running the Deployment Script
1.  Navigate to the deployment directory: `cd deployment`
2.  Run the script: `python flood_monitor_production.py`

---

##  Getting Started (For Model Reproduction)

To explore the model development process, start with the notebooks in the `/notebooks` directory. Make sure to install the dependencies first.

```bash
# Clone the repository
git clone [Your-Repository-URL]
cd [repository-folder-name]

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab to open the notebooks
jupyter lab