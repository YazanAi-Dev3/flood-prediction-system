import requests
import pandas as pd
import pickle
import schedule
import time
import logging
import os  # Required to check for file/folder existence
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

# --- 1. Global Configuration ---
SCHEDULE_INTERVAL_MINUTES = 60
BASE_URL = "https://std.scit.co/flood/public/api"
CITIES_ENDPOINT = f"{BASE_URL}/cities"
WEATHER_DATA_ENDPOINT = f"{BASE_URL}/weather-data/"
NOTIFICATION_ENDPOINT = f"{BASE_URL}/send-notification"
# The name of the folder where the local BERT model will be saved.
LOCAL_BERT_PATH = 'bert-base-uncased-local'

# --- 2. Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Function: %(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler("flood_monitor_production.log"),
        logging.StreamHandler()
    ]
)

# --- 3. Helper Function to Download BERT ---
def download_bert_model_if_not_exists(model_name_from_hub, save_path):
    """
    Checks if a BERT model exists locally. If not, it downloads and saves it.
    """
    if os.path.exists(save_path):
        logging.info(f"Local BERT model found at '{save_path}'. Loading from local files.")
        return
    
    logging.warning(f"Local BERT model not found at '{save_path}'.")
    logging.info(f"Downloading BERT model '{model_name_from_hub}' from Hugging Face Hub. This may take a moment...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name_from_hub)
        model = BertModel.from_pretrained(model_name_from_hub)
        
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        logging.info(f"BERT model successfully downloaded and saved to '{save_path}'.")
    except Exception as e:
        logging.critical(f"FATAL ERROR: Failed to download BERT model. Check your internet connection. Error: {e}", exc_info=True)
        exit()

# --- 4. Load Models and Processors (Once at Startup) ---
logging.info("--- SCRIPT INITIALIZATION ---")
try:
    logging.info("Loading StandardScaler (scaler.pkl)...")
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    logging.info("Loading PCA model (pca.pkl)...")
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    
    logging.info("Loading Flood Prediction Model (flood_model.pkl)...")
    with open('flood_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Download BERT if needed, then load it
    download_bert_model_if_not_exists('bert-base-uncased', LOCAL_BERT_PATH)
    
    logging.info(f"Loading BERT Tokenizer and Model from local path: {LOCAL_BERT_PATH}...")
    tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH)
    bert_model = BertModel.from_pretrained(LOCAL_BERT_PATH)
    
    logging.info("--- All models initialized successfully. ---")
except FileNotFoundError as e:
    logging.critical(f"FATAL ERROR: A model file (.pkl) was not found: {e}. Exiting.", exc_info=True)
    exit()
except Exception as e:
    logging.critical(f"FATAL ERROR: An unexpected error occurred during model loading: {e}", exc_info=True)
    exit()


# --- 5. Helper and Prediction Functions ---
def create_text_description(data_row):
    """Creates the unified text description from a DataFrame row."""
    return (
        f"Meteorological report for the month number {int(data_row.get('Month', 0))}: "
        f"Current rainfall is {data_row.get('Rainfall (mm)', 0)} mm, with {data_row.get('Previous 24-hour Rainfall (mm)', 0)} mm in the last 24 hours. "
        f"The location elevation is {data_row.get('Elevation (m)', 0)} meters. "
        f"Current conditions show a temperature of {data_row.get('Temperature (°C)', 0)}°C, "
        f"humidity at {data_row.get('Humidity (%)', 0)}%, and wind speed of {data_row.get('Wind Speed (km/h)', 0)} km/h. "
        f"Air pressure is {data_row.get('Air Pressure (hPa)', 0)} hPa, with {data_row.get('Cloud Cover (%)', 0)}% cloud cover "
        f"and a dew point of {data_row.get('Dew Point (°C)', 0)}°C."
    )

def process_and_predict(city_data_json):
    """Takes the raw JSON data, processes it fully, and returns a flood prediction."""
    # (The logic inside this function remains exactly the same as the final version we built)
    logging.info("Starting data processing for a new prediction...")
    df = pd.DataFrame([city_data_json])
    numerical_cols = scaler.feature_names_in_
    for col in numerical_cols:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    df['Text_Description'] = df.apply(create_text_description, axis=1)
    encoded_input = tokenizer(df['Text_Description'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1).numpy()
    pca_features = pca.transform(embeddings)
    pca_cols = [f'PCA_{i+1}' for i in range(pca_features.shape[1])]
    df_pca = pd.DataFrame(pca_features, columns=pca_cols)
    df_scaled = pd.DataFrame(scaler.transform(df[numerical_cols]), columns=numerical_cols)
    df_processed = pd.concat([df_scaled, df_pca], axis=1)
    prediction = model.predict(df_processed)[0]
    logging.info(f"Prediction successful. Result: '{prediction}'")
    return prediction

# --- 6. The Main Job Function ---
def check_floods_and_notify():
    """The main task that will be scheduled to run."""
    # (The logic inside this function remains exactly the same as the final version we built)
    logging.info("--- Starting new scheduled job cycle ---")
    try:
        cities_response = requests.get(CITIES_ENDPOINT, timeout=10)
        cities_response.raise_for_status()
        cities = cities_response.json()
        logging.info(f"Successfully fetched {len(cities)} cities.")
        for city in cities:
            city_id = city.get('id')
            city_name = city.get('name', 'Unknown')
            if not city_id:
                logging.warning(f"Skipping an entry because it has no ID. Data: {city}")
                continue
            logging.info(f"--- Processing City: {city_name} (ID: {city_id}) ---")
            try:
                weather_endpoint = f"{WEATHER_DATA_ENDPOINT}{city_id}"
                weather_response = requests.get(weather_endpoint, timeout=10)
                if weather_response.status_code == 200:
                    weather_data = weather_response.json()
                    prediction = process_and_predict(weather_data)
                    if prediction.lower() != 'low':
                        logging.warning(f"HIGH RISK DETECTED for {city_name}. Severity: {prediction}. Sending notification...")
                        notification_payload = {'title': f'⚠️ Flood Alert for {city_name}', 'body': f'Potential flood risk detected. Severity: {prediction}'}
                        post_response = requests.post(NOTIFICATION_ENDPOINT, json=notification_payload, timeout=10)
                        post_response.raise_for_status()
                        logging.info(f"Notification for {city_name} sent successfully.")
                    else:
                        logging.info(f"No significant flood risk detected for {city_name}.")
                else:
                    logging.warning(f"Could not retrieve weather data for {city_name}. Server responded with Status Code: {weather_response.status_code}.")
            except requests.exceptions.RequestException as city_req_err:
                logging.error(f"An API request error occurred for city {city_name}: {city_req_err}")
            except Exception as city_proc_err:
                logging.error(f"An unexpected error occurred while processing city {city_name}: {city_proc_err}", exc_info=True)
    except Exception as e:
        logging.error(f"A critical error occurred in the main job cycle: {e}", exc_info=True)
    logging.info("--- Scheduled job cycle finished ---")


# --- 7. Scheduling Logic ---
if __name__ == "__main__":
    schedule.every(SCHEDULE_INTERVAL_MINUTES).minutes.do(check_floods_and_notify)
    logging.info(f"Scheduler initialized. Will run every {SCHEDULE_INTERVAL_MINUTES} minutes.")
    logging.info("Running the first job immediately upon startup.")
    check_floods_and_notify()
    logging.info("Startup job complete. Entering main scheduling loop...")
    while True:
        schedule.run_pending()
        time.sleep(1)