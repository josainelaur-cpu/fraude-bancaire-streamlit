import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(
    page_title="Transaction unique",
    page_icon="🔎"
)


st.title("🔎 Analyse d'une transaction")


st.write(
"""
Cette page permet d'analyser une transaction bancaire
et de prédire son niveau de risque.
"""
)


montant = st.number_input(
    "Montant",
    value=50000
)


type_transaction = st.selectbox(
    "Type de transaction",
    [
        "ATM",
        "Paiement en ligne",
        "Paiement électronique"
    ]
)


if st.button("🚨 Analyser"):

    st.success(
        "Analyse terminée"
    )