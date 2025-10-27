import streamlit as st
from joblib import load
import numpy as np

model = load('data/iris_model.joblib')
prediction = model.predict([[1,2,3,4]])

sepal_length = st.number_input("insert sepal length", value=None, placeholder="Insert sepal length")
sepal_width = st.number_input("insert sepal width", value=None, placeholder="Insert sepal width")
petal_length = st.number_input("insert petal length", value=None, placeholder="Insert petal length")
petal_width = st.number_input("insert petal length", value=None, placeholder="Insert petal width")

st.write("Given sepal length is ", sepal_length)
st.write("Given sepal width is ", sepal_width)
st.write("Given petal length is ", petal_length)
st.write("Given petal width is ", petal_width)

def make_pred(sepal_length, sepal_width, petal_length, petal_width):
    input_array = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return prediction

# if st.button('Predict'):
#     st.write('Predicting', prediction)

if st.button('Predict'):
    prediction = make_pred(sepal_length, sepal_width, petal_length, petal_width)
    if prediction == 0:
        st.success('Prediction: Iris Setosa')
        st.image("images/iris_setosa.png", caption="Iris Setosa")
    elif prediction == 1:
        st.success('Prediction: Iris Versicolor')
        st.image("images/iris_versicolor.png", caption="Iris Versicolor")
    elif prediction == 2:
        st.success('Prediction: Iris Virginica')
        st.image("images/iris_virginica.png", caption="Iris Virginica")
   
    
    
    
