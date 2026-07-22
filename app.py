import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px


# ==========================
# Configuration
# ==========================

st.set_page_config(
    page_title="Détection Fraude Bancaire",
    page_icon="🏦",
    layout="wide"
)


# ==========================
# Chargement modèle
# ==========================

@st.cache_resource
def load_model():

    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")

    return model, scaler, encoders



model, scaler, encoders = load_model()



# ==========================
# Classes du modèle
# ==========================

classes = np.array(
    [
        "Normal",
        "Suspect",
        "Fraude"
    ]
)



# ==========================
# Interface
# ==========================

# ==========================
# PAGE D'ACCUEIL PROFESSIONNELLE
# ==========================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 45px;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }

    .card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        text-align: center;
        height: 150px;
    }

    .card h2 {
        color: #1f4e79;
    }

    .section {
        margin-top: 30px;
        font-size: 22px;
        font-weight: bold;
        color: #1f4e79;
    }

    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <div class="main-title">
    🏦 Système Intelligent de Détection de Fraude Bancaire
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <div class="subtitle">
    Application basée sur l'intelligence artificielle permettant
    d'analyser les transactions bancaires et d'identifier
    automatiquement les comportements suspects.
    </div>
    """,
    unsafe_allow_html=True
)



# ==========================
# Cartes statistiques
# ==========================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="card">
        <h2>🤖 IA</h2>
        <p>Modèle Machine Learning<br>
        pour la détection automatique</p>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="card">
        <h2>📊 Analyse</h2>
        <p>Transactions individuelles<br>
        ou fichiers CSV complets</p>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="card">
        <h2>🚨 Sécurité</h2>
        <p>Identification des fraudes<br>
        et transactions suspectes</p>
        </div>
        """,
        unsafe_allow_html=True
    )



# ==========================
# Présentation projet
# ==========================

st.markdown(
    """
    <div class="section">
    🎯 Objectif du projet
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
"""
Ce système permet aux institutions financières de réduire les risques
liés aux transactions frauduleuses grâce à un modèle prédictif capable
de classer une opération en trois catégories :

- ✅ Transaction normale
- ⚠️ Transaction suspecte
- 🚨 Fraude détectée

L'application propose deux modes d'utilisation :
"""
)



col1, col2 = st.columns(2)


with col1:

    st.info(
        """
        ### 🔎 Analyse individuelle
        
        Saisie manuelle d'une transaction :
        
        - Montant
        - Type de transaction
        - Statut opération
        - Localisation
        
        Résultat instantané avec probabilité.
        """
    )


with col2:

    st.info(
        """
        ### 📂 Analyse en masse
        
        Import d'un fichier CSV contenant
        plusieurs transactions.

        Génération automatique d'un rapport
        des prédictions.
        """
    )



# ==========================
# Technologies
# ==========================

st.markdown(
    """
    <div class="section">
    🛠️ Technologies utilisées
    </div>
    """,
    unsafe_allow_html=True
)


st.success(
"""
Python | Streamlit | Pandas | Scikit-Learn | Machine Learning | GitHub
"""
)


st.caption(
"Projet Intelligence Artificielle - Détection de Fraude Bancaire"
)



# ==========================
# Menu
# ==========================

mode = st.sidebar.radio(
    "Mode d'analyse",
    [
        "Accueil",
        "Transaction unique",
        "Analyse fichier CSV",
        "Dashboard"
    ]
)


# =====================================================
# MODE 1 : TRANSACTION UNIQUE
# =====================================================

if mode == "Transaction unique":


    st.subheader("🔎 Analyse d'une transaction")


    montant = st.number_input(
        "Montant",
        min_value=0,
        value=50000
    )


    date = st.date_input(
        "Date transaction"
    )


    type_transaction = st.selectbox(
        "Type de transaction",
        [
            "ATM",
            "Paiement en ligne",
            "Paiement électronique"
            
        ]
    )


    status = st.selectbox(
        "Status opération",
        [
            "En attente",
            "Validé",
            "Échoué"
        ]
    )


    localisation = st.selectbox(
    "Localisation",
    [
        "Bambey",
        "Bignona",
        "Bounkiling",
        "Cap Skirring",
        "Dagana",
        "Dahra",
        "Dakar",
        "Diamniadio",
        "Diourbel",
        "Fatick",
        "Foundiougne",
        "Gossas",
        "Goudiry",
        "Guinguinéo",
        "Guédiawaye",
        "Hann",
        "Kaffrine",
        "Kanel",
        "Kaolack",
        "Kolda",
        "Kédougou",
        "Linguère",
        "Louga",
        "Matam",
        "Mbacké",
        "Mbour",
        "Nioro",
        "Oussouye",
        "Pikine",
        "Podor",
        "Ranérou",
        "Richard Toll",
        "Rufisque",
        "Saint Louis",
        "Saint-Louis",
        "Saloum",
        "Sédhiou",
        "Tambacounda",
        "Thiès",
        "Tivaouane",
        "Touba",
        "Toubacouta",
        "Vélingara",
        "Ziguinchor"
    ]
)


    if st.button("🚨 Analyser"):


        data = pd.DataFrame({

            "ID Clients":[0],

            "Numero de compte":[0],

            "Identifiant operation":[0],

            "Type de transaction":[type_transaction],

            "Status operation":[status],

            "Localisation":[localisation],

            "Date":[date.toordinal()],

            "Montant":[montant]

        })



        # Encodage sécurisé

        for col, encoder in encoders.items():

            if col in data.columns:

                valeur = data[col].astype(str)


                valeur = valeur.apply(
                    lambda x:
                    x if x in encoder.classes_
                    else encoder.classes_[0]
                )


                data[col] = encoder.transform(valeur)



        # Ajout colonnes manquantes

        for col in scaler.feature_names_in_:

            if col not in data.columns:

                data[col] = 0



        # Respect ordre modèle

        data = data[
            scaler.feature_names_in_
        ]



        data = data.apply(
            pd.to_numeric,
            errors="coerce"
        )



        # Normalisation

        data_scaled = scaler.transform(data)



        # Prédiction

        prediction = model.predict(
            data_scaled
        )[0]
        

        probabilites = model.predict_proba(
            data_scaled
        )[0]



        # Tableau probabilités

        resultats = pd.DataFrame({

            "Classe":classes,

            "Probabilité":probabilites

        })



        resultats["Probabilité"] = (
            resultats["Probabilité"] * 100
        ).round(2).astype(str) + "%"



        st.subheader(
            "📊 Détail des probabilités"
        )


        st.dataframe(
            resultats
        )



        probabilite = probabilites.max()



        classe_prediction = classes[prediction]



        st.divider()



        if classe_prediction == "Normal":


            st.success(
                f"✅ Transaction normale\n\nProbabilité : {probabilite:.2%}"
            )



        elif classe_prediction == "Fraude":


            st.error(
                f"🚨 Fraude détectée\n\nProbabilité : {probabilite:.2%}"
            )



        else:


            st.warning(
                f"⚠️ Transaction suspecte\n\nProbabilité : {probabilite:.2%}"
            )





# =====================================================
# MODE 2 : ANALYSE CSV
# =====================================================

elif mode == "Analyse fichier CSV":


    st.subheader(
        "📂 Analyse de plusieurs transactions"
    )



    fichier = st.file_uploader(

        "Déposer un fichier CSV",

        type=["csv"]

    )



    if fichier is not None:


        df = pd.read_csv(fichier, sep=";")



        st.write(
            "Aperçu du fichier :"
        )


        st.dataframe(
            df.head()
        )



        if st.button(
            "🚨 Lancer l'analyse"
        ):



            resultats = []



            for index, ligne in df.iterrows():



                transaction = pd.DataFrame({

                    "ID Clients":[ligne["ID Clients"]],

                    "Numero de compte":[ligne["Numero de compte"]],

                    "Identifiant operation":[ligne["Identifiant operation"]],

                    "Type de transaction":[ligne["Type de transaction"]],

                    "Status operation":[ligne["Status operation"]],

                    "Localisation":[ligne["Localisation"]],

                    "Date":[
                        pd.to_datetime(
                            ligne["Date"]
                        ).toordinal()
                    ],

                    "Montant":[ligne["Montant"]]

                })




                # Encodage sécurisé

                for col, encoder in encoders.items():


                    if col in transaction.columns:


                        valeur = transaction[col].astype(str)



                        valeur = valeur.apply(

                            lambda x:
                            x if x in encoder.classes_
                            else encoder.classes_[0]

                        )



                        transaction[col] = encoder.transform(
                            valeur
                        )




                # Colonnes manquantes

                for col in scaler.feature_names_in_:


                    if col not in transaction.columns:


                        transaction[col] = 0




                transaction = transaction[
                    scaler.feature_names_in_
                ]



                transaction = transaction.apply(
                    pd.to_numeric,
                    errors="coerce"
                )




                # Scaling

                transaction_scaled = scaler.transform(
                    transaction
                )



                prediction = model.predict(
                    transaction_scaled
                )[0]



                probabilite = model.predict_proba(
                    transaction_scaled
                )[0]




                resultats.append({

                    "Montant":
                    ligne["Montant"],


                    "Localisation":
                    ligne["Localisation"],


                    "Classe":
                    classes[prediction],


                    "Probabilité fraude (%)":
                    round(
                        max(probabilite)*100,
                        2
                    )

                })




            resultat_df = pd.DataFrame(
                resultats
            )



            st.success(
                "✅ Analyse terminée"
            )



            st.dataframe(
                resultat_df
            )



            csv = resultat_df.to_csv(
                index=False
            ).encode("utf-8")



            st.download_button(

                "⬇️ Télécharger les résultats",

                csv,

                "resultats_fraude.csv",

                "text/csv"

            )