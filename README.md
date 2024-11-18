# SeizureSen

**SeizureSen** is a project aimed at predicting seizures in dogs using intracranial EEG (iEEG) signals. The application utilizes machine learning to classify the preictal phase of brain activity, leveraging the dataset from the [American Epilepsy Society Seizure Prediction Challenge](https://www.kaggle.com/competitions/seizure-prediction/overview).

## Dataset

The project leverages the dataset provided by the **American Epilepsy Society Seizure Prediction Challenge** on Kaggle, which contains intracranial EEG recordings from both dogs and humans with epilepsy. While the dataset includes data from both species, this project specifically focuses on recordings from dogs, aligning with its goal of developing a seizure prediction application tailored for canine use.

## Problem Description

Epileptic brain activity is divided into four states:

1. **Interictal**: Normal baseline activity between seizures.
2. **Preictal**: Brain state leading up to a seizure.
3. **Ictal**: Active seizure phase.
4. **Postictal**: Recovery phase after a seizure.

The problem lies in accurately identifying the **preictal state**, which can help forecast seizures in dogs. This enables their owners to take timely interventions, mitigate risks, and manage the effects of epilepsy more effectively.

## Features

- **Machine Learning Models**: Classifier for preictal vs. interictal states in iEEG signals
- **Customizable Pipelines**: Easily adapt algorithms and preprocessing steps.
- **User Interface:** Includes an intuitive interface for testing the model and visualizing prediction results, making it easy to interpret and analyze outcomes.

---

## Getting Started

### Prerequisites

Ensure you have Python installed (>=3.8). You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/LucasVon0645/SeizureSen.git
cd SeizureSen
```

### Setting Up a Virtual Environment

1.Create a virtual environment (optional but recommended):

```bash
# Using venv (Python 3.X)
python -m venv env

# Activate the virtual environment
# On Windows
.\env\Scripts\Activate
# On macOS/Linux
source env/bin/activate
```

2.Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running the Application

1. Ensure the virtual environment is activated
2. Ensure a pre-trained model is available in /models
3. Run the the main script

```bash
python main.py
```
