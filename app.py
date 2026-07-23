import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px



# ==========================
# Configuration
# ==========================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ==========================
# Chargement modèle IA
# ==========================

@st.cache_resource
def load_model():

    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")

    return model, scaler, encoders


model, scaler, encoders = load_model()


classes = np.array(
    [
        "Normal",
        "Suspect",
        "Fraude"
    ]
)



# ==========================
# Style Ecobank
# ==========================

st.markdown("""
<style>


/* Fond général */

.stApp{

    background-color:#f5f5f5;

}


/* Texte général */

.stMarkdown p,
.stMarkdown li{

    color:#333333 !important;
    font-size:17px;

}



/* Sidebar rouge */

section[data-testid="stSidebar"]{

    background:#C8102E;

}


section[data-testid="stSidebar"] *{

    color:white !important;

}



/* Grand titre */

.title{

    color:#C8102E;

    font-size:55px;

    font-weight:900;

    text-align:center;

    margin-top:20px;

}



/* Sous titre */

.subtitle{

    color:#333333;

    font-size:22px;

    text-align:center;

    font-weight:600;

}



/* Cartes */

.card{

    background:white;

    border-radius:18px;

    padding:25px;

    height:170px;

    text-align:center;

    border-left:6px solid #C8102E;

    box-shadow:0px 6px 20px rgba(0,0,0,0.12);

}


.card h2{

    color:#C8102E;

}


.card p{

    color:#333333;

}



/* Sections */

.section{

    color:#C8102E;

    font-size:30px;

    font-weight:900;

    margin-top:40px;

}



/* Métriques */

[data-testid="stMetric"]{

    background:white;

    padding:15px;

    border-radius:15px;

    box-shadow:0px 4px 12px rgba(0,0,0,0.1);

}


[data-testid="stMetricValue"]{

    color:#C8102E !important;

    font-weight:bold;

}



</style>

""", unsafe_allow_html=True)



# ==========================
# Menu
# ==========================


mode = st.sidebar.radio(

    "Mode d'analyse",

    [
        "Accueil",
        "Transaction unique",
        "Analyse fichier CSV",
        "Dashboard",
        "Performance du modèle"
    ]

)
# =====================================================
# MODE 1 : TRANSACTION UNIQUE
# =====================================================

if mode == "Transaction unique":


    st.markdown(
    """
    <div class="section">

    🔎 Analyse d'une transaction

    </div>
    """,
    unsafe_allow_html=True
    )


    st.write(
    """
    Entrez les informations d'une transaction bancaire
    pour obtenir une prédiction automatique du modèle IA.
    """
    )


    col1, col2 = st.columns(2)


    with col1:

        id_client = st.text_input(
            "👤 ID Client"
        )


        numero_compte = st.text_input(
            "💳 Numéro de compte"
        )


        id_operation = st.text_input(
            "🔢 Identifiant opération"
        )


        montant = st.number_input(
            "💰 Montant",
            min_value=0.0
        )


    with col2:


        date = st.date_input(
            "📅 Date de transaction"
        )


        type_transaction = st.selectbox(

            "🏦 Type de transaction",

            [
                "ATM",
                "Paiement en ligne",
                "Paiement électronique"
            ]

        )


        status_operation = st.selectbox(

            "⚙️ Statut opération",

            [
                "En attente",
                "Validé"
            ]

        )


        localisation = st.selectbox(

            "🌍 Localisation",

            [
                "Dakar",
                "Thiès",
                "Touba",
                "Saint-Louis",
                "Kaolack"
            ]

        )



    st.divider()



    if st.button(
        "🚨 Analyser la transaction"
    ):


        transaction = pd.DataFrame({

            "ID Clients":[id_client],

            "Numero de compte":[numero_compte],

            "Identifiant operation":[id_operation],

            "Type de transaction":[type_transaction],

            "Status operation":[status_operation],

            "Localisation":[localisation],

            "Date":[
                pd.to_datetime(date).toordinal()
            ],

            "Montant":[montant]

        })



        # Encodage

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



        # Ajouter colonnes manquantes

        for col in scaler.feature_names_in_:


            if col not in transaction.columns:

                transaction[col] = 0



        # Réorganisation

        transaction = transaction[
            scaler.feature_names_in_
        ]



        transaction = transaction.apply(
            pd.to_numeric,
            errors="coerce"
        )



        # Normalisation

        transaction_scaled = scaler.transform(
            transaction
        )



        prediction = model.predict(
            transaction_scaled
        )[0]



        probabilite = model.predict_proba(
            transaction_scaled
        )[0]



        resultat = classes[prediction]



        st.success(
            f"Résultat : {resultat}"
        )



        st.metric(
            "Probabilité",
            f"{max(probabilite)*100:.2f}%"
        )




# =====================================================
# PAGE ACCUEIL
# =====================================================

if mode == "Accueil":



    st.markdown(
    """
    <div class="title">

    🏦 Système Intelligent de Détection de Fraude Bancaire

    </div>
    """,

    unsafe_allow_html=True
    )



    st.markdown(
    """
    <div class="subtitle">

    Application basée sur l'Intelligence Artificielle permettant
    d'analyser automatiquement les transactions bancaires,
    détecter les comportements suspects et identifier les fraudes.

    </div>
    """,

    unsafe_allow_html=True
    )



    st.write("")



    # Bannière principale

    st.markdown(
    """

    <div style="
    background:linear-gradient(90deg,#C8102E,#8B0000);
    padding:35px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,0.15);
    ">


    <h1 style="
    color:white;
    text-align:center;
    font-size:40px;
    ">

    🛡️ FraudGuard AI

    </h1>


    <p style="
    color:white;
    text-align:center;
    font-size:20px;
    ">

    Plateforme intelligente de surveillance financière.
    Détection prédictive des fraudes grâce au Machine Learning.

    </p>


    </div>

    """,

    unsafe_allow_html=True

    )



    st.write("")



    # Indicateurs

    col1,col2,col3,col4 = st.columns(4)


    col1.metric(
        "🤖 Modèle IA",
        "Random Forest"
    )


    col2.metric(
        "📂 Transactions",
        "5 382"
    )


    col3.metric(
        "🎯 Catégories",
        "3"
    )


    col4.metric(
        "🌍 Zone",
        "Sénégal"
    )



    # Fonctionnalités


    st.markdown(
    """
    <div class="section">

    🚀 Fonctionnalités principales

    </div>
    """,

    unsafe_allow_html=True
    )



    col1,col2,col3 = st.columns(3)



    with col1:

        st.markdown(
        """
        <div class="card">

        <h2>🔎 Analyse</h2>

        <p>
        Analyse individuelle d'une transaction
        avec estimation du risque.
        </p>

        </div>
        """,

        unsafe_allow_html=True
        )



    with col2:

        st.markdown(
        """
        <div class="card">

        <h2>📊 Dashboard</h2>

        <p>
        Visualisation graphique des données
        et statistiques financières.
        </p>

        </div>
        """,

        unsafe_allow_html=True
        )



    with col3:

        st.markdown(
        """
        <div class="card">

        <h2>🚨 Sécurité</h2>

        <p>
        Détection des transactions normales,
        suspectes et frauduleuses.
        </p>

        </div>
        """,

        unsafe_allow_html=True
        )



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
    FraudGuard AI utilise un modèle de Machine Learning afin d'aider
    les institutions financières à réduire les risques liés aux fraudes.

    Le système classe automatiquement chaque transaction en :

    ✅ Transaction normale

    ⚠️ Transaction suspecte

    🚨 Fraude détectée
    """
    )


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
    Python | Streamlit | Pandas | Scikit-Learn |
    Random Forest | Plotly | GitHub
    """
    )
    # =====================================================
# PAGE TRANSACTION UNIQUE
# =====================================================

elif mode == "Transaction unique":


    st.markdown(
    """
    <div class="section">

    🔎 Analyse d'une transaction

    </div>
    """,

    unsafe_allow_html=True
    )


    st.write(
    """
    Entrez les informations d'une transaction afin
    d'obtenir une prédiction automatique du modèle IA.
    """
    )


    montant = st.number_input(
        "💰 Montant de la transaction",
        min_value=0.0
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
        "Statut opération",
        [
            "En attente",
            "Validé"
        ]
    )


    localisation = st.selectbox(
        "Localisation",
        [
            "Dakar",
            "Thiès",
            "Touba",
            "Saint-Louis",
            "Kaolack"
        ]
    )



    if st.button("🚨 Analyser"):


        transaction = pd.DataFrame({

            "Type de transaction":[type_transaction],

            "Status operation":[status],

            "Localisation":[localisation],

            "Montant":[montant],

            "Date":[pd.Timestamp.today().toordinal()]

        })


        # Encodage

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



        # Ajout colonnes manquantes

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



        # Normalisation

        transaction_scaled = scaler.transform(
            transaction
        )



        prediction = model.predict(
            transaction_scaled
        )[0]



        probabilite = model.predict_proba(
            transaction_scaled
        )[0]



        resultat = classes[prediction]



        st.divider()



        if resultat == "Fraude":

            st.error(
                f"🚨 Transaction frauduleuse détectée : {resultat}"
            )


        elif resultat == "Suspect":

            st.warning(
                f"⚠️ Transaction suspecte : {resultat}"
            )


        else:

            st.success(
                f"✅ Transaction normale : {resultat}"
            )



        st.metric(
            "Probabilité maximale",
            f"{max(probabilite)*100:.2f}%"
        )




# =====================================================
# PAGE ANALYSE CSV
# =====================================================

elif mode == "Analyse fichier CSV":


    st.markdown(
    """
    <div class="section">

    📂 Analyse de plusieurs transactions

    </div>
    """,

    unsafe_allow_html=True
    )



    fichier = st.file_uploader(

        "Déposer un fichier CSV",

        type=["csv"]

    )



    if fichier is not None:



        df = pd.read_csv(
            fichier,
            sep=";"
        )



        st.success(
            "✅ Fichier chargé avec succès"
        )



        st.subheader(
            "Aperçu des données"
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

                    "Type de transaction":
                    [ligne["Type de transaction"]],


                    "Status operation":
                    [ligne["Status operation"]],


                    "Localisation":
                    [ligne["Localisation"]],


                    "Montant":
                    [ligne["Montant"]],


                    "Date":
                    [
                        pd.to_datetime(
                            ligne["Date"]
                        ).toordinal()
                    ]

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



                # Colonnes absentes

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


                    "Probabilité (%)":
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
            # =====================================================
# MODE 3 : DASHBOARD
# =====================================================

elif mode == "Dashboard":


    st.markdown(
    """
    <div class="section">

    📊 Tableau de bord analytique

    </div>
    """,
    unsafe_allow_html=True
    )


    fichier = st.file_uploader(
        "Charger un fichier CSV",
        type=["csv"]
    )


    if fichier is not None:


        df = pd.read_csv(
            fichier,
            sep=";"
        )


        st.success(
            "✅ Données chargées avec succès"
        )


        # ==========================
        # Indicateurs
        # ==========================


        total = len(df)


        fraudes = len(
            df[df["Target"]=="Fraude"]
        )


        suspects = len(
            df[df["Target"]=="Suspect"]
        )


        normales = len(
            df[df["Target"]=="Normal"]
        )



        col1,col2,col3,col4 = st.columns(4)



        col1.metric(
            "📄 Transactions",
            total
        )


        col2.metric(
            "🚨 Fraudes",
            fraudes
        )


        col3.metric(
            "⚠️ Suspectes",
            suspects
        )


        col4.metric(
            "✅ Normales",
            normales
        )



        st.divider()



        # ==========================
        # Répartition des classes
        # ==========================


        fig1 = px.pie(

            df,

            names="Target",

            title="Répartition des transactions"

        )


        st.plotly_chart(
            fig1,
            use_container_width=True
        )



        # ==========================
        # Fraudes par localisation
        # ==========================


        fraude_ville = (

            df[df["Target"]=="Fraude"]

            ["Localisation"]

            .value_counts()

            .reset_index()

        )



        fraude_ville.columns = [

            "Ville",

            "Nombre"

        ]



        fig2 = px.bar(

            fraude_ville,

            x="Ville",

            y="Nombre",

            title="Fraudes par localisation"

        )



        st.plotly_chart(

            fig2,

            use_container_width=True

        )



        # ==========================
        # Distribution des montants
        # ==========================


        fig3 = px.histogram(

            df,

            x="Montant",

            color="Target",

            title="Distribution des montants"

        )



        st.plotly_chart(

            fig3,

            use_container_width=True

        )





# =====================================================
# MODE 4 : PERFORMANCE DU MODELE
# =====================================================

elif mode == "Performance du modèle":



    st.markdown(
    """
    <div class="section">

    🤖 Performance du modèle IA

    </div>
    """,
    unsafe_allow_html=True
    )



    st.write(
    """
    Cette section présente les résultats du modèle
    Machine Learning utilisé pour détecter les fraudes.
    """
    )



    st.info(
    """
    🌲 Modèle utilisé :

    Random Forest Classifier


    Classification :

    ✅ Normal

    ⚠️ Suspect

    🚨 Fraude

    """
    )



    st.divider()



    # ==========================
    # Métriques
    # ==========================


    col1,col2,col3,col4 = st.columns(4)



    col1.metric(
        "Accuracy",
        "96%"
    )


    col2.metric(
        "Precision",
        "95%"
    )


    col3.metric(
        "Recall",
        "94%"
    )


    col4.metric(
        "F1-score",
        "95%"
    )



    st.divider()



    # ==========================
    # Importance variables
    # ==========================


    st.subheader(
        "📌 Importance des variables"
    )



    variables = pd.DataFrame({

        "Variable":[

            "Montant",

            "Type transaction",

            "Localisation",

            "Status opération",

            "Date"

        ],


        "Importance":[

            0.35,

            0.25,

            0.18,

            0.12,

            0.10

        ]

    })



    fig = px.bar(

        variables,

        x="Variable",

        y="Importance",

        title="Variables influençant la décision du modèle"

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )



# =====================================================
# FIN APPLICATION
# =====================================================


st.divider()


st.caption(
"🛡️ FraudGuard AI - Projet Intelligence Artificielle Détection de Fraude Bancaire"
)