# ann_classification

**ANN CLASSIFICATION** – A Streamlit-powered app for predicting customer churn using an Artificial Neural Network (ANN).

---

## Demo

Explore the live application here:  
[Live Demo](https://annclassification-ankur.streamlit.app/)

---

## Project Structure

| File/Folder                  | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `app.py`                     | Streamlit app for interactive churn prediction.                            |
| `annexperiment.ipynb`        | Jupyter notebook exploring model training, evaluation, and experimentation. |
| `prediction.ipynb`           | Notebook demonstrating how to make predictions using the trained model.     |
| `model.h5`                   | Trained ANN model in Keras HDF5 format.                                     |
| `Churn_Modelling.csv`        | Dataset used for training and evaluation purposes.                          |
| `label_encoder_gender.pkl`   | Fitted label encoder for the 'Gender' feature.                              |
| `onehot_encoder_geo.pkl`     | Fitted one-hot encoder for the 'Geography' feature.                         |
| `scaler.pkl`                 | Scaler (e.g., StandardScaler) for numeric feature normalization.            |
| `requirements.txt`           | Python dependencies needed to run the project.                             |

---

## Installation & Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/ankurpython/ann_classification.git
   cd ann_classification
