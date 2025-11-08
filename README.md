# 🩺 Multi Disease Prediction Web Application

A machine learning–powered **Flask web app** that predicts possible diseases based on a user’s symptoms.  
This project integrates data preprocessing, multiple classification models, and an ensemble voting system to provide accurate and explainable predictions.

---

## 📘 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Technologies Used](#technologies-used)
4. [Machine Learning Models](#machine-learning-models)
5. [Deep Learning Model](#deep-learning-model)
6. [Ensemble Approach](#ensemble-approach)
7. [Web Application (Flask)](#web-application-flask)
8. [Frontend Interface](#frontend-interface)
9. [Project Folder Structure](#project-folder-structure)
10. [How to Run the Project Locally](#how-to-run-the-project-locally)
11. [Results & Accuracy](#results--accuracy)
12. [Future Enhancements](#future-enhancements)
13. [Credits](#credits)

---

## 🧠 Project Overview
The **Multi Disease Prediction System** uses symptoms entered by the user to predict the most probable disease(s).  
It combines traditional machine learning algorithms and a deep learning model, further refined by an **ensemble voting classifier** for improved accuracy.

Key goals:
- Predict multiple diseases based on given symptoms
- Analyze and compare the performance of different models
- Deploy a user-friendly web app using Flask and HTML/CSS/JS frontend

---

## 📊 Dataset Information
- **Dataset name:** Diseases and Symptoms Dataset  
- **Source:** Kaggle ([dhivyeshrk/diseases-and-symptoms-dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset))  
- **Attributes:** Symptoms (binary features), Disease label  
- **Records:** ~400,000 (after augmentation and cleaning)

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3 |
| **Web Framework** | Flask |
| **Frontend** | HTML, CSS, JavaScript |
| **Machine Learning** | scikit-learn (Decision Tree, Naive Bayes, Logistic Regression) |
| **Deep Learning** | TensorFlow / Keras |
| **Data Processing** | pandas, numpy, LabelEncoder, MinMaxScaler |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | joblib |
| **Text Embedding** | SentenceTransformers (SBERT) |

---

## 🧩 Machine Learning Models
Three classical models were trained:
1. **Decision Tree Classifier**  
   - Easy to interpret but prone to overfitting.
   - Accuracy: **81.62%**
2. **Naive Bayes Classifier**  
   - Performs well on symptom-based data.  
   - Accuracy: **86.67%**
3. **Logistic Regression**  
   - Linear model suitable for multiclass classification.  
   - Accuracy: **86.71%**

Each model was evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

---

## 🧬 Deep Learning Model
A fully connected **Feedforward Neural Network (DNN)** was built using Keras:
- Input layer: Number of symptoms
- Hidden layers: 128 → 64 neurons (ReLU activation, Dropout 0.2)
- Output layer: Softmax (multi-class classification)
- Optimizer: Adam
- Loss: Categorical Crossentropy

**Validation Accuracy:** ~86.02%  
Although similar in performance to classical models, deep learning increased computational overhead — so it was excluded from deployment for efficiency.

---

## 🤝 Ensemble Approach
To achieve a more robust model, a **Soft Voting Ensemble** was implemented:
- Combines predictions from Decision Tree, Naive Bayes, and Logistic Regression
- Uses average probabilities for final prediction
- Final Ensemble Accuracy: **86.25%**

### Why Ensemble?
> “The ensemble balances out weaknesses of individual models — improving consistency and reliability of disease predictions.”

---

## 🌐 Web Application (Flask)
The Flask backend handles:
- User input (symptoms text)
- Text → symptom vector conversion
- Model loading and prediction
- Returning top probable diseases (JSON)

**Routes:**
| Route | Method | Description |
|--------|---------|-------------|
| `/` | GET | Homepage (HTML UI) |
| `/predict` | POST | Returns top predicted diseases (JSON) |

---

## 🎨 Frontend Interface
The frontend is built with **HTML, CSS, and JavaScript**, designed for simplicity and usability.

Features:
- Symptom input text box
- “Predict” button
- Display of top 3 probable diseases with prediction confidence
- Responsive design

<h3 align="center">🖥️ Example Layout</h3>

<pre align="center">
+-------------------------------------------+
| 🩺 Multi Disease Prediction System        |
|-------------------------------------------|
| [ I have fever and sore throat...      ]  |
| [ Predict ]                              |
|-------------------------------------------|
| Predicted Diseases:                      |
| 1. Typhoid — 87.65%                      |
| 2. Dengue — 7.41%                        |
| 3. Flu — 4.94%                           |
+-------------------------------------------+
</pre>


---

### 🖼️ Web App Screenshot

<p align="center">
  <img src="static/screenshot.png" alt="Web App Screenshot" width="800"/>
</p>

---

## 📁 Project Folder Structure


```
multi-disease-prediction/
│
├── app.py # Flask application
├── README.md # Project documentation
│
├── trained_models/ # Trained model files (not uploaded)
│ └── README.txt # Info about missing models
│
├── templates/
│ └── index.html # Frontend HTML
│
└── static/
├── style.css # Frontend styles
└── app.js # Frontend scripts
└── screenshot.png  
```

---

## 🖥️ How to Run the Project Locally

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/RakshithDN/multi-disease-prediction.git
cd multi-disease-prediction
```

### 2️⃣ Install the required packages (only the essentials):
```bash
pip install flask joblib numpy pandas scikit-learn
```

### 3️⃣ Place your trained models

Copy the following files into the `trained_models/` folder:

- `naive_bayes_model.pkl`
- `decision_tree_model.pkl`
- `logistic_regression_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `symptom_columns.pkl`


### 4️⃣ Run the Flask app:
python app.py

### 5️⃣ Open in browser:
http://127.0.0.1:5000

## 📈 Results & Accuracy
| Model | Accuracy | Notes |
|--------|-----------|--------|
| Decision Tree | 81.62% | Simple, interpretable |
| Naive Bayes | 86.67% | Best classical model |
| Logistic Regression | 86.71% | Stable and fast |
| Deep Learning (DNN) | 86.02% | High accuracy but heavy |
| Ensemble (Voting) | **86.25%** | Balanced & consistent |


## 🚀 Future Enhancements

- Expand the dataset to include more diseases and diverse symptom variations  
- Deploy the web app on a public cloud (Render / Hugging Face / Railway)  
- Add user authentication and patient history tracking  
- Improve UI/UX with better symptom suggestions and autocomplete  
- Build a REST API version for integration with hospital systems or mobile apps  
- Add multilingual support for wider accessibility  


## 🙌 Credits
**Developed by:** Rakshith D N  
**Dataset:** Diseases and Symptoms Dataset (Kaggle)  
**Frameworks:** Flask, scikit-learn, TensorFlow  


## 🧾 License

This project is open-source for educational purposes.  
You are free to modify and use it for learning or research.


---

### ✅ **After you paste this into your `README.md` file**, commit and push it:
```bash
git add README.md
git commit -m "Added detailed project README"
git push origin main
```
