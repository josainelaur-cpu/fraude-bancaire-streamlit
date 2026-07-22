import streamlit as st
import pandas as pd
import joblib
import numpy as np


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
        "Fraude",
        "Normal",
        "Suspect"
    ]
)



# ==========================
# Interface
# ==========================

st.title("🏦 Système IA de Détection de Fraude Bancaire")

st.markdown(
    "Analysez une transaction bancaire et estimez le risque de fraude."
)



# ==========================
# Menu
# ==========================

mode = st.sidebar.radio(
    "Mode d'analyse",
    [
        "Transaction unique",
        "Analyse fichier CSV"
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
            "Validé"
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