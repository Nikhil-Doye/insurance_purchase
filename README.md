<h2>Overview</h2>
<hr>
This repository contains code for predicting vehicle insurance purchase using machine learning techniques. It includes multiple models and preprocessing pipeline, along with scripts for data preprocessing, training, and a Flask app for model demonstration.

<h2>Repository Structure</h2>
<hr>
artifacts: Contains traina and test datasets, preprocessor and model pickle files. 
notebook: Consits of 2 jupyter notebooks, one for EDA and another for optimal thresholding.
src: Includes scripts for different purposes:
    `hyperparameters.py`: Defines hyperparameters used in the model.
    `exception.py`: Handles all the exceptions.
    `logger.py`: Logging functionalities for tracking the training process.
    `utils.py`: Consists of various functions which are used in other scripts.
    components:
        `ingest_data.py`: Data ingestion
        `data_transformation.py`: Transform data in such a manner which is suitable to be fed to the model.
        `model_trainer.py`: Contains training, validation and testing of the various models and picks the best one.
    pipeline:
        `predict_pipeline.py`: Connects front end with preprocessor and model.
templates:
    `home.html`: consists of the form page.
    `index.html`: landing page.
'app.py': Flask application for demonstrating the functionality of the trained model.
'requirements.txt': Cosnists of all the libraries required to run the model.

<h2>Usage</h2>
<hr>
Training: Use `data_ingestion.py` to execute the training process. Ensure necessary dependencies are installed by referring to the requirements.txt file.
Demo: Run `app.py` to launch the Flask app for demonstrating the trained model. Make sure to have the required libraries installed as mentioned in `requirements.txt`.

<h2>Getting Started</h2>
<hr>
Clone this repository.
Set up a Python environment and install the necessary dependencies listed in requirements.txt.
Utilize the provided scripts in the src directory for model training, data preprocessing, etc.
Execute the run_pipeline.py script to train the model.
Run the streamlit_app.py to experience the model via the Streamlit app.

<h2>Contributions</h2>
<hr>
Contributions are welcome! Feel free to fork this repository, make changes, and create a pull request. Please adhere to the repository's guidelines.