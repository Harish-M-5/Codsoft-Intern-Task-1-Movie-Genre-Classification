# codsoft-task-1-Movie-Genre-Classification
Movie-Genre-Classification-Using-NLP

---
## 📌 Introduction
Movie Genre Classification is a Natural Language Processing (NLP) based Machine Learning project that predicts the genre of a movie using its textual description.
This project is developed as Task 1 of the CodSoft Machine Learning Internship and demonstrates practical implementation of text classification using classical ML techniques.

---

## 📖 Overview
The system analyzes movie plot descriptions and classifies them into appropriate genres such as Action, Comedy, Drama, Thriller, Romance, etc.
The model uses TF-IDF vectorization for text feature extraction and Logistic Regression for multi-class classification.
A Streamlit web interface is provided for real-time prediction.

---

## 🛠️ Tech Used 
- Python
- Machine Learning
- Scikit-learn
- NLP Technique
- TF-IDF
- Logistic Regression
- Streamlit
- Dataset Format

---

## Dataset
https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

---

## 🧠 What is Movie Genre Classification Using NLP & Machine Learning?
Movie Genre Classification is a text classification problem where machine learning models are trained to identify patterns in movie descriptions.
Using NLP techniques, textual data is converted into numerical vectors, which are then used by ML algorithms to predict the most suitable genre for a given movie plot 

---

##  ⚙️ Project Setup

The project follows a modular structure separating the machine learning logic and the user interface.
TXT-based datasets are directly used without conversion to CSV.

---

## 📦 Install and Setup

1) Prerequisites:
- Python 3.8+
- pip package manager

2) Install Dependencies :

pip install -r requirements.txt

3) ▶️ Running the Application:

 streamlit run app.py

4) The application opens in the browser at:

http://localhost:8501

---

## 🏗️ Model Architecture

Text Input

   ↓
   
Text Cleaning & Preprocessing

   ↓
   
TF-IDF Vectorization

   ↓
   
Logistic Regression Classifier

   ↓
   
Predicted Movie Genre

---
## 🔐 Security
- No user data is stored
- No external API usage
- Runs completely on local machine
- Dataset used only for educational purposes

---

##  📂 Dataset Information
The dataset is provided in TXT format and contains movie metadata and descriptions.

Files Used:
- train_data.txt – Training dataset
- test_data.txt – Test dataset
- test_data_solution.txt – Test dataset with labels
- description.txt – Dataset explanation

Data Format:
  
ID ::: TITLE ::: GENRE ::: DESCRIPTION

---

## 🧩 Key Components
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Multi-class Logistic Regression model
- Streamlit UI for predictions
- Accuracy evaluation using test dataset

---

## 📊 Dataset Details
The dataset is sourced from publicly available movie databases and is used strictly for academic and internship learning purposes.

---

## 📷 output
<img width="1920" height="1080" alt="Screenshot 2025-12-25 201234" src="https://github.com/user-attachments/assets/928c2d98-8b41-48f5-873d-1c09049fecd6" />

---

##  📄 License
This project is licensed under the MIT License.

---

## 🙏 Acknowledgement
- CodSoft for providing the internship opportunity
- Open-source Python and ML community
- Scikit-learn and Streamlit developers

---

## 📈 Conclusion

This project successfully demonstrates how NLP and Machine Learning can be used to classify movie genres from textual descriptions.
It fulfills CodSoft Machine Learning Internship – Task 1 requirements and showcases practical ML implementation with a user-friendly interface.

---

## 👨‍💻 Author
Harish
Machine Learning | NLP | Streamlit


