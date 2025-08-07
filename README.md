# 🤖 Customer Churn Prediction Using Artificial Neural Networks (ANN)

## 📘 Project Overview
This project implements a deep learning-based **Artificial Neural Network (ANN)** to predict customer churn in a banking context. Using real-world structured data, the goal is to identify customers likely to leave the service, allowing businesses to proactively address churn risks.

The model was built using TensorFlow/Keras and evaluated based on accuracy, loss trends, and confusion matrix results. The project also includes proper feature engineering, preprocessing, model tuning, and performance visualization.

---

## 🎯 Objectives
- Load and preprocess the `Churn_Modelling.csv` dataset
- Engineer features using encoding and scaling
- Build an ANN using Keras Sequential API
- Train the model with **dropout** and **early stopping** for generalization
- Evaluate model performance using accuracy, loss plots, and a confusion matrix

---

## 🧠 Model Architecture
- **Input Layer**: 11 features (after encoding categorical data)
- **Hidden Layers**:
  - Layer 1: 11 units, `ReLU` activation
  - Layer 2: 7 units, `ReLU` activation + `Dropout(0.3)`
  - Layer 3: 6 units, `ReLU` activation + `Dropout(0.2)`
- **Output Layer**: 1 unit, `Sigmoid` activation
- **Optimizer**: Adam (`learning_rate = 0.01`)
- **Loss Function**: Binary Crossentropy
- **Callback**: EarlyStopping (monitors `val_loss`, patience=20)

---

## 📊 Dataset Description
- `Churn_Modelling.csv` contains 10,000 customer records.
- Features include: `CreditScore`, `Age`, `Tenure`, `Balance`, `Geography`, `Gender`, `IsActiveMember`, `EstimatedSalary`, etc.
- **Target variable**: `Exited` (1 = churned, 0 = retained)

---

## 📈 Model Performance
- **Final Accuracy**: ~85.35% on test set
- **Confusion Matrix**:  [[1477, 118],
                          [ 175, 230]]
- The model generalizes well and early stopping prevents overfitting.
- Training and validation accuracy/loss curves are plotted to visualize learning trends.

---

## 🛠 Tools & Libraries Used
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- TensorFlow / Keras
- scikit-learn (for metrics and data splitting)

---

## 🧮 Key Steps in Code
1. Load and clean the data
2. Encode categorical features (`Geography`, `Gender`)
3. Scale numerical features using `StandardScaler`
4. Build and compile the ANN using `Sequential()`
5. Train the model with validation split and early stopping
6. Evaluate using:
 - Accuracy score
 - Confusion matrix
 - Training vs Validation plots (accuracy/loss)

---

## 🧾 Project Files
- `customer_churn_ann.ipynb` – Full annotated notebook
- `Churn_Modelling.csv` – Dataset
- `README.md` – Project documentation

---

## 👨‍💻 Author
- **Ajay Babu Mahanti** – Data Science & Deep Learning Enthusiast  
*(Final-Year B.Tech Student @ IIT Roorkee)*

---

## 📌 Future Improvements
- Use SHAP for explainable AI
- Hyperparameter tuning using GridSearchCV + Keras wrappers
- Deploy model via Flask/Streamlit
- Compare with XGBoost and Random Forest for baseline benchmarking

---

## 📄 License
Open-source for educational and non-commercial use only.


