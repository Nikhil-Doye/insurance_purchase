<h2>Overview</h2>
<hr>
This repository contains code for predicting vehicle insurance purchases using machine learning techniques. It includes multiple models and a preprocessing pipeline, along with scripts for data preprocessing, and training, and a Flask app for model demonstration.

<h3>Demo - <a href="https://insurancepurchaseprediction.azurewebsites.net/predictdata">https://insurancepurchaseprediction.azurewebsites.net/predictdata</a></h3>

<h2>Repository Structure</h2>
<hr>
<ul>
<li>artifacts: Contains traina and test datasets, preprocessor and model pickle files.</li>
<li>notebook: Consists of 2 jupyter notebooks, one for EDA and another for optimal thresholding.</li>

<li>src: Includes scripts for different purposes:</li>
<ul>
    <li> `hyperparameters.py`: Defines hyperparameters used in the model.</li>
    <li>`exception.py`: Handles all the exceptions.</li>
    <li>`logger.py`: Logging functionalities for tracking the training process.</li>
    <li>`utils.py`: Consists of various functions which are used in other scripts.</li>
    </ul>
    <li>components:</li>
        <ul><li>`ingest_data.py`: Data ingestion</li>
        <li>`data_transformation.py`: Transform data in such a manner which is suitable to be fed to the model.</li>
        <li>`model_trainer.py`: Contains training, validation and testing of the various models and picks the best one.</li></ul>
    <li>pipeline:
        <ul><li>`predict_pipeline.py`: Connects front end with preprocessor and model.</li>
        </ul>
<li>templates:
    <ul><li>`home.html`: consists of the form page.</li>
    <li>`index.html`: landing page.</li></ul>
<li>`app.py`: Flask application for demonstrating the functionality of the trained model.</li>
<li>`requirements.txt``: Cosnists of all the libraries required to run the model.</li>
</ul>
</ul>
<h2>Usage</h2>
<hr>
<ul>
<li>Training: Use `data_ingestion.py` to execute the training process. Ensure necessary dependencies are installed by referring to the requirements.txt file.</li>
<li>Demo: Run `app.py` to launch the Flask app for demonstrating the trained model. Make sure to have the required libraries installed as mentioned in `requirements.txt`.</li>

<h2>Getting Started</h2>
<hr>
<ol>
<li>Clone this repository.</li>
<li>Set up a Python environment and install the necessary dependencies listed in requirements.txt.</li>
<li>Utilize the provided scripts in the src directory for model training, data preprocessing, etc.</li>
<li>Execute the run_pipeline.py script to train the model.</li>
<li>Run the app.py to experience the model via the Flask app.</li>
</ol>

<h2>Contributions</h2>
<hr>
Contributions are welcome! Feel free to fork this repository, make changes, and create a pull request. Please adhere to the repository's guidelines.
