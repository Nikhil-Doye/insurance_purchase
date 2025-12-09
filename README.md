# 🚗 Vehicle Insurance Purchase Prediction

## 📖 Overview
---
This repository contains code for predicting **vehicle insurance purchases** using advanced machine learning techniques. It includes multiple models, a preprocessing pipeline, and scripts for data preparation, training, and deployment. Additionally, a **Flask app** is included for demonstrating functionality of the model.

### 🌐 Demo
Visit the live demo: [Insurance Purchase Prediction](https://insurancepurchaseprediction.azurewebsites.net/predictdata)

## 🗂️ Repository Structure
---
- **`artifacts`**: Contains training and test datasets, preprocessor, and model pickle files.
- **`notebook`**: Includes two Jupyter notebooks:
  - **EDA Notebook**: Exploratory Data Analysis.
  - **Optimal Thresholding Notebook**: Optimizing model thresholds.
- **`src`**: Contains essential scripts:
  - `hyperparameters.py`: Defines model hyperparameters.
  - `exception.py`: Centralized exception handling.
  - `logger.py`: Logging utilities for tracking training and debugging.
  - `utils.py`: Helper functions for reuse across scripts.
- **`components`**: Core processing modules:
  - `ingest_data.py`: Handles data ingestion.
  - `data_transformation.py`: Transforms raw data into model-ready formats.
  - `model_trainer.py`: Trains, validates, and tests models, selecting the best one.
- **`pipeline`**:
  - `predict_pipeline.py`: Integrates the frontend with preprocessing and the trained model.
- **`templates`**: HTML templates for the web interface:
  - `home.html`: Input form page.
  - `index.html`: Landing page.
- **`app.py`**: Flask application for showcasing model predictions.
- **`requirements.txt`**: Lists all the necessary libraries for running the project.

## 🚀 Usage
---
1. **Training**: Run `ingest_data.py` to start the training process. Ensure dependencies from `requirements.txt` are installed.
2. **Demo**: Use `app.py` to launch the Flask application for interacting with the model.

## 🛠️ Getting Started
---
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/vehicle-insurance-prediction.git
   ```
2. Set up a Python virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the scripts in the `src` directory for tasks like model training and data preprocessing.
4. Run the pipeline:
   ```bash
   python pipeline/run_pipeline.py
   ```
5. Launch the Flask app:
   ```bash
   python app.py
   ```

## 🤝 Contributions
---
Contributions are always welcome! Here’s how you can contribute:
1. Fork this repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

Please ensure you adhere to the repository’s guidelines when making contributions.

---

Thank you for exploring this project! 🎉 Feel free to reach out with questions or feedback. Let’s innovate together! 🚀

