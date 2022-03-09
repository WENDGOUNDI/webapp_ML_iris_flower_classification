from this import d
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("model.h5")
labels = np.load("labels.npy")

st.markdown("<h1 style='text-align: center; color: red;'>Machine Learning web-based Iris Flower Prediction</h1>", unsafe_allow_html=True)

a = float(st.number_input("Sepal length in cm"))
b = float(st.number_input("Sepal width in cm"))
c = float(st.number_input("Petal length in cm"))
d = float(st.number_input("Petal width in cm"))

btn = st.button("predict")

if btn:
    pred = model.predict(np.array([a,b,c,d]).reshape(1,-1))
    print(pred)
    print(np.argmax(pred))
    pred = labels[np.argmax(pred)]
    desc = "Your flower dimensions correspond to an {}".format(pred)

    st.subheader(desc)

    if pred=="Iris-Setosa":
        st.image("setosa.jpg")
    elif pred=="Iris Versicolour":
        st.image("versicolor.jpg")
    else:
        st.image("verginca.jpg")