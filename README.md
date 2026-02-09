# 🧠 Neural Heart Disease Predictor
The Neural Heart Disease Predictor is a web-based application that utilizes a trained transformer model to predict the likelihood of heart disease based on patient vitals. This project aims to provide a user-friendly interface for inputting patient data and obtaining accurate predictions, leveraging the power of machine learning and neural networks. The application is built using the Streamlit framework and integrates with a trained `CardioTransformer` model, which is trained on a dataset of patient vitals using the PyTorch library.

## 🚀 Features
- **User-Friendly Interface**: The application features a simple and intuitive interface for inputting patient data, including age, gender, height, weight, blood pressure, cholesterol level, glucose level, smoking status, alcohol intake, and physical activity level.
- **Trained Model**: The application uses a trained `CardioTransformer` model, which is a PyTorch neural network model that utilizes a transformer encoder to predict heart disease.
- **Data Preprocessing**: The application preprocesses the input data by scaling the features and splitting the data into training and test sets.
- **Prediction**: The application uses the trained model to make predictions on the input data, providing a likelihood of heart disease.
- **Visualization**: The application features visualization capabilities using Plotly, allowing users to visualize the predicted results.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend**: PyTorch
- **Database**: None
- **AI Tools**: PyTorch, Transformer
- **Build Tools**: None
- **Dependencies**: pandas, numpy, scikit-learn, torch, streamlit, joblib, plotly

## 📦 Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```
This will install all the necessary libraries and dependencies required to run the project.

## 💻 Usage
To run the application, navigate to the project directory and run the following command:
```bash
streamlit run app.py
```
This will start the Streamlit server and launch the application in your default web browser.

## 📂 Project Structure
```markdown
.
├── app.py
├── train_transformer.py
├── requirements.txt
├── cardio_train.csv
├── saved_model
│   ├── model.pth
│   └── scaler.pkl
└── README.md
```


## 📬 Contact
For any questions or concerns, please contact us at [rishithkoduri6@gmail.com](mailto:support@example.com).
