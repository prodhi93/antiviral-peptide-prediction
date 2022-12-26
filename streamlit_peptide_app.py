import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from antiviral_analysis.peptide_class import Peptide
#import torch

st.title("Antiviral Peptide Predictor")

st.subheader("Enter the amino acid sequence of your peptide")
seq = st.text_input('Enter sequence')

#Picking which model user wants to deploy
st.subheader("Is your peptide ribosomally synthesised (biologically occurring), or synthetic?")
synth_type = st.selectbox('Synthesis Type',('Ribosomal', 'Synthetic')) 
#option is stored in this variable

st.subheader("Choose your threshold for positive antiviral activity")
threshold = st.slider("Percent threshold", min_value=0.01, max_value=100.00, step=0.01)
threshold = float(threshold)/100

st.subheader("Choose the number of amino acids in the shortest truncated segment")
min_length = st.slider("Minimum length", min_value=1, max_value=len(seq), step=1)
min_length = int(min_length)


@st.cache(suppress_st_warning=True)
def model_loader(model_name_or_path):
    model = load(str(model_name_or_path))
    return model

def extract_seq_data(sequence):
    pep = Peptide(str(sequence))
    data = np.array([pep.percent_helix(), pep.aromaticity(), pep.percent_sheet(), pep.aliphatic_index(),
              pep.hydrophobic_moment(), pep.percent_turn(), pep.boman_index(), pep.instability_index(),
              pep.isoelectric_point(), pep.charge_density(), pep.sequence_charge()]).reshape(1, -1) 
    return data
    
    
rfc_full = model_loader("smote_rfc_full_model.joblib")
if synth_type=="Ribosomal":
    rfc = model_loader("smote_rfc_ribo_model.joblib")
elif synth_type=="Synthetic":
    rfc = model_loader("smote_rfc_synth_model.joblib")


def get_truncated_data(seq, model, min_length):  
    seq_list = [seq]
    data = extract_seq_data(seq)
    data_list = [data]
    pred_list = [model.predict_proba(data)[0][1]]
    while len(seq) > min_length+1:
        seq = seq[1:-1]
        seq_list.append(seq)
        data = extract_seq_data(seq)
        data_list.append(data)
        pred_list.append(model.predict_proba(data)[0][1])
    return seq_list, data_list, pred_list

seq_preds = get_truncated_data(seq, rfc, min_length)
seq_preds_full = get_truncated_data(seq, rfc_full, min_length)


st.header("Prediction Tables")

df = pd.DataFrame(seq_preds[0], columns=["Sequence"])
df["Predicted Antiviral Probability"] = seq_preds[2]
df["Possible Antiviral?"] = df["Predicted Antiviral Probability"].apply(lambda x: "Yes" if x > threshold else "No" )

df_full = pd.DataFrame(seq_preds_full[0], columns=["Sequence"])
df_full["Predicted Antiviral Probability"] = seq_preds_full[2]
df_full["Possible Antiviral?"] = df_full["Predicted Antiviral Probability"].apply(lambda x: "Yes" if x > threshold else "No" )

st.subheader(f"Using model trained on {synth_type} Peptides")
st.dataframe(df)

st.subheader(f"Using model trained on both Ribosomal and Synthetic Peptides")
st.dataframe(df_full)



