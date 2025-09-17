#EJECUTAR ESTO PRIMERO EN CONSOLA
# Si da error, debes ir a PowerShell de Window y:
#      Get-ExecutionPolicy                           Si es Restricted; ejecuta
#      Set-ExecutionPolicy RemoteSigned              Colocar Sí
#   Luego de usar este script; ir a PowerShell:   Set-ExecutionPolicy Restricted

# Crea un ambiente virtual (puedes usar otro nombre en lugar de 'venv')
#   python -m venv venv
#   .\venv\Scripts\activate   # En Windows

# Instala la versión específica de scikit-learn
#   pip install scikit-learn==1.2.2
# Instala otras dependencias, incluyendo Streamlit
#  pip install streamlit pandas joblib
#-------------------------------------------------------------------------------------------------
# Desde la segunda vez: hacer:
# Si da error, debes ir a PowerShell de Window y:
#      Get-ExecutionPolicy                           Si es Restricted; ejecuta
#      Set-ExecutionPolicy RemoteSigned              Colocar Sí
# En consola de VSC:  .\venv\Scripts\activate

import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# -------------------------PROCESO DE DESPLIEGUE------------------------------
# En consola:
# pip install scikit-learn==1.3.2

# 01 --------------------------Load the model-------------------------------------------
clf = load('modelo_rfchurn_tunning.joblib')

# 02---------------- Variables globales para los campos del formulario-----------------------
Sexo_options=['M','F']
Sexo =''
E_Civil_options=['Casado','Divorciado','Soltero']
E_Civil=''
Educacion_options=['No.Sup','Sup.Incomp','Sup.Comp']
Educacion=''
Lic_Conducir_options=['Si','No']
Lic_Conducir =''
Edad=0.0000
Tarjetas=0
Deuda=0
Saldo=0
CrediScore=0
años_empleo=0.0000
Ingresos=0


# 03 Reseteo------------- Flag to track error---------------------------------------
error_flag = False

# Reset inputs function
def reset_inputs():
    global Sexo, E_Civil,Educacion,Lic_Conducir,Edad,Tarjetas,Deuda,Saldo,CrediScore,años_empleo,Ingresos, error_flag
    Sexo =''
    E_Civil=''
    Educacion=''
    Lic_Conducir =''
    Edad=0.0000
    Tarjetas=0
    Deuda=0
    Saldo=0
    CrediScore=0
    años_empleo=0.0000
    Ingresos=0
    error_flag = False

# Inicializar variables
reset_inputs()
# -----------------------------------------------------------------------------------------------

# ------------------------Título centrado-------------------------------------------------
st.title("Modelo Predictivo  con Random Forest Classifier")
st.markdown("Este modelo predice otorgar un crédito en base a diferentes características.")
st.markdown("---")

# ----------------------- Función para validar los campos del formulario----------------------------
def validate_inputs():
    global error_flag
    if any(val < 0 for val in [Edad,Tarjetas,Deuda,Saldo,CrediScore,años_empleo,Ingresos]):
        st.error("No se permiten valores negativos. Por favor, ingrese valores válidos en todos los campos.")
        error_flag = True
    else:
        error_flag = False

# ------------------------------------ Formulario en dos columnas------------------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    # Input fields en la primera columna
    with col1:
        Edad = st.number_input("**Edad**", min_value=0.0, value=float(Edad), step=1.0)
        Tarjetas = st.number_input("**Número de tarjetas**", min_value=0, value=Tarjetas, step=1)
        Deuda = st.number_input("**Deuda**", min_value=0, value=Deuda, step=1)
        Saldo = st.number_input("**Saldo**", min_value=0, value=Saldo, step=1)
        CrediScore = st.number_input("**CrediScore**", min_value=0, value=CrediScore, step=1)
        años_empleo = st.number_input("**Años de empleo**", min_value=0.0, value=float(años_empleo), step=1.0)
        Ingresos = st.number_input("**Ingresos**", min_value=0, value=Ingresos, step=1)

    # Input fields en la segunda columna
    with col2:
        Sexo = st.selectbox("**Sexo**", Sexo_options)
        E_Civil= st.selectbox("**Estado civil**", E_Civil_options)
        Educacion= st.selectbox("**Nivel de instrucción**", Educacion_options)
        Lic_Conducir = st.selectbox("**Licencia de conducir**", Lic_Conducir_options)
        
    # ----------------------------------------- Boton de Predecir-------------------------------------------------
    predict_button = st.form_submit_button("Predecir")

# Validar que no haya valores negativos en los campos cuando se presiona el botón
# Si hay error no permita seguir tipeando!!!!!!!!!!!!!!!!!!!
if predict_button and error_flag:
    st.stop()

if predict_button and not error_flag:
    # Crear DataFrame
    data = {
        'Sexo': [Sexo],
        'E_Civil': [E_Civil],
        'Educacion': [Educacion],
        'Lic_Conducir': [Lic_Conducir],
        'Edad': [Edad],
        'Tarjetas': [Tarjetas],
        'Deuda': [Deuda],
        'Saldo': [Saldo],
        'CrediScore': [CrediScore],
        'años_empleo': [años_empleo],
        'Ingresos': [Ingresos],
    }
    df = pd.DataFrame(data)

    # Realizar predicción
    probabilities_classes = clf.predict_proba(df)[0]

    # Obtener la clase con la mayor probabilidad
    class_predicted = np.argmax(probabilities_classes)

    # Asignar salida y probabilidad según la clase predicha
    # En el script original: #Exited: 0 Cliente retenido;  1 Cliente cerró cuenta
    if class_predicted == 0:
        outcome = "No se otorga crédito"
        probability_churn = probabilities_classes[0]
        style_result = 'background-color: lightgreen; font-size: larger;'
    else:
        outcome = "Se otorga crédito"
        probability_churn = probabilities_classes[1]
        style_result = 'background-color: lightcoral; font-size: larger;'

    # Mostrar resultado con estilo personalizado
    result_html = f"<div style='{style_result}'>La predicción fue de clase '{outcome}' con una probabilidad de {round(float(probability_churn), 4)}</div>"
    st.markdown(result_html, unsafe_allow_html=True)

# --------------------------- Boton de Resetear-------------------------------------
if st.button("Resetear"):
    # Resetear inputs
    reset_inputs()

# streamlit run app_streamlit.py       en la consola
#Coindice con 06_Random_Forest_pipelines.ipynb

#pip freeze > requirements.txt

#  LUEGO DE CREAR requirements: 
#    pip install -r requirements.txt