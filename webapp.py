## Importing necessary libraries for the web app
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report
# from sklearn import metrics
# from sklearn import tree
# from sklearn.metrics import accuracy_score
# import warnings
# warnings.filterwarnings('ignore')
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# df= pd.read_csv('Crop_recommendation.csv')

# #features = df[['temperature', 'humidity', 'ph', 'rainfall']]
# X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
# y = df['label']
# labels = df['label']

# # Split the data into training and testing sets
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
# RF = RandomForestClassifier(n_estimators=20, random_state=5)
# RF.fit(Xtrain,Ytrain)
# predicted_values = RF.predict(Xtest)
# x = metrics.accuracy_score(Ytest, predicted_values)

# import pickle
# # Dump the trained Naive Bayes classifier with Pickle
# RF_pkl_filename = 'RF.pkl'
# # Open the file to save as pkl file
# RF_Model_pkl = open(RF_pkl_filename, 'wb')
# pickle.dump(RF, RF_Model_pkl)
# # Close the pickle instances
# RF_Model_pkl.close()


#model = pickle.load(open('RandomForest.pkl', 'rb'))
RF_Model_pkl=pickle.load(open('RandomForest.pkl','rb'))

## Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # # Making predictions using the model
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

## Streamlit code for the web app interface
def main():  
    # # Setting the title of the web app
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS🌱", unsafe_allow_html=True)
    
    st.sidebar.title("CropPredict")
    # # Input fields for the user to enter the environmental factors
    st.sidebar.header("Find out the most suitable crop to grow in your farm 👨‍")
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0, max_value=140, value=0, step=1)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0, max_value=145, value=0, step=1)
    potassium = st.sidebar.number_input("Potassium", min_value=0, max_value=205, value=0, step=1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    inputs=[[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]                                               
   
    # # Validate inputs and make prediction
    inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    if st.sidebar.button("Predict"):
        if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop is: {prediction[0]}")


## Running the main function
if __name__ == '__main__':
    main()

