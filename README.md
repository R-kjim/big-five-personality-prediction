<!-- # Personality Trait Classification using TF-IDF and BERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellowgreen)

A machine learning system that classifies text into Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) using BERT embeddings and a neural network classifier.

## Features

- End-to-end personality trait prediction pipeline
- Customizable classification thresholds
- Model interpretability with confidence scores

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

### File Structure

```
├── data/                           # Training and Raw datasets
├── models/                         # Created models
|   |-- BERT/                       # BERT Model
|   |-- TF-IDF/                     # TF-IDF model
├── notebooks/                      # Jupyter notebooks
├── src/
│   ├── tf_idf.py                    # TF-IDF Model prediction script
│   ├── bert_model.py                # BERT Model Prediction script
│   └── main.ipynb                   # Data cleaning script
│   └── model1.ipynb                 # TF-IDF Model training script
│   └── model2.ipynb                 # BERT Model training script
├── requirements.txt
└── README.md
```
### Methods
The models folder has been left blank intentionally. To populate it, one is required to train the models by running the notebooks. 
To start off, run ```src/main.ipnyb```. This notebook takes in the raw data  in ```data/updated_personality_data_train.csv``` and processes it for use in training the models, after which the resulting data is saved in a csv file ```data/clean_data.csv```

Proceed to individually train the models by runnings the ```src/model1.ipynb``` and ```src/model2.ipynb``` notebooks. This will result in populating the models folder with the relevant files for each model

To use the models, this functionalities are scripted in the ```.py``` files. To predict personality using the TF-IDF model, run the ```src/tf_idf.py``` file. To make use of the BERT model in predicting personality, run the ```src/bert_model.py``` file# big-five-personality-prediction -->
