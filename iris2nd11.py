import streamlit as st
import numpy as np
import pickle
import sklearn 
from joblib import load

# Load the trained model
model = load('iris_knn_classifier.joblib')
# Main function for the Streamlit app
def main():
    st.title("Iris Species Prediction App - ML2 Assignment - Omar Altarakieh")
    st.write("IE MBD Apr 23 S1")

    # Input features
    sepal_length = st.slider('SepalLengthCm', 0.0, 8.0)
    sepal_width = st.slider('SepalWidthCm', 0.0, 8.0)
    petal_length = st.slider('PetalLengthCm', 0.0, 8.0)
    petal_width = st.slider('PetalWidthCm', 0.0, 8.0)

    # Organizing features into a numpy array
    user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediction
    if st.button("Predict"):
        species = model.predict(user_data)
        st.subheader(f"The predicted species is: {species[0]}")

if __name__ == '__main__':
    main()
