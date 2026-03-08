# FUTURE_ML_02
NLP-based support ticket classification system built using Python and Scikit-learn.
## 📖 Project Overview

Customer support teams receive thousands of support tickets every day.
Manually categorizing these tickets is time-consuming and inefficient.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify support tickets into predefined categories such as:

* Billing Issues
* Technical Problems
* Account Management
* General Queries

The model analyzes the **text content of support tickets** and predicts the most appropriate category.


## 🎯 Objectives

* Automate the classification of support tickets
* Reduce manual workload for support teams
* Improve response efficiency
* Apply NLP techniques for text processing
* Train and evaluate a machine learning classification model


## 🧠 Machine Learning Workflow

1. Data Collection
2. Text Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training
5. Model Evaluation
6. Ticket Category Prediction


## 🔍 Text Preprocessing Steps

* Convert text to lowercase
* Remove punctuation
* Remove stopwords
* Tokenization
* Text vectorization using **TF-IDF**


## 🤖 Machine Learning Model

The following algorithms can be used for classification:

* **Logistic Regression**
* **Naive Bayes**
* **Support Vector Machine (SVM)**

The trained model is saved and later used to classify new incoming support tickets.


## 🛠 Technologies & Libraries

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **Matplotlib**
* **Seaborn**
* **Streamlit**


## 📂 Project Structure

FUTURE_ML_01
│
├── tickets.csv              # Dataset containing support tickets
├── streamlit_app.py       # Streamlit application for ticket prediction
├── ticket_model.pkl        # Trained ML model file
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Ignored files configuration


## ⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/FUTURE_ML_01.git

Navigate to the project folder:

cd FUTURE_ML_01

Install required dependencies:

pip install -r requirements.txt


## ▶️ Running the Application

Run the Streamlit application:

streamlit run streamlit_app.py

After running, the application will open in your browser where you can **enter a support ticket message and get the predicted category**.


## 📊 Model Evaluation

The model performance is evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score

These metrics help measure how well the model classifies support tickets.


## 💡 Example Input

Support Ticket:

"My internet connection is very slow and keeps disconnecting."

Predicted Category:

Technical Issue


## 🔮 Future Improvements

* Implement deep learning models such as **LSTM or BERT**
* Improve text preprocessing techniques
* Deploy the model using cloud services
* Integrate with real customer support systems
* Add a dashboard for ticket analytics


## 👨‍💻 Author

**MOHAMMED SAIF**
AI & Machine Learning Engineering Student


## ⭐ Internship Information

This project was developed as part of the **Machine Learning Internship Program**.
