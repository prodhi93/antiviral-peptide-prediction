import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import plotly.graph_objects as go
from antiviral_analysis.peptide_class import Peptide
#import torch

st.title("Antiviral Peptide Predictor")

with st.form(key="peptide_choices"):
    st.subheader("Enter the amino acid sequence of your peptide")
    seq = st.text_input('Enter sequence')

    #Picking which model user wants to deploy
    st.subheader("Is your peptide ribosomally synthesised (biologically occurring), or synthetic?")
    synth_type = st.radio('Synthesis Type',('Ribosomal', 'Synthetic')) 
    #option is stored in this variable

    st.subheader("Choose your threshold for positive antiviral activity")
    threshold = st.slider("Percent threshold", min_value=0.01, max_value=100.00, step=0.01, \
                          help="With an extreme needle-in-haystack problem such as this one, positive probabilities are low despite undersampling correction.\
                         Hint: 25% is a good starter value for positivity thresholds as the model returns probability values > 0.25 for confirmed antiviral peptides.")
    threshold = float(threshold)/100

    st.subheader("Choose the number of amino acids in the shortest truncated segment")
    min_length = st.slider("Minimum length", min_value=1, max_value=len(seq), step=1)
    min_length = int(min_length)
    
    submit_button = st.form_submit_button("Enter")

@st.cache(allow_output_mutation=True)
def model_loader(model_name_or_path):
    model = load(str(model_name_or_path))
    return model

def extract_seq_data(sequence):
    pep = Peptide(str(sequence))
    data = np.array([pep.percent_helix(), pep.aromaticity(), pep.percent_sheet(), pep.aliphatic_index(),
              pep.hydrophobic_moment(), pep.percent_turn(), pep.boman_index(), pep.instability_index(),
              pep.isoelectric_point(), pep.charge_density(), pep.sequence_charge(), pep.cysteine_count(),
              pep.polarity(), pep.h_bonding(), pep.bulky_properties(), pep.compositional_char_index(),
              pep.local_flexibility(), pep.electronic_props(), pep.helix_bend_pref(), pep.side_chain_size(),
              pep.ext_struct_pref(), pep.double_bend_pref(), pep.partial_specific_volume(), pep.flat_extended_pref(),
              pep.pK_C(), pep.ms_whim_scores()[0], pep.ms_whim_scores()[1], pep.ms_whim_scores()[2]]).reshape(1, -1) 
    return data

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

def customise_aggrid_df(data):
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        theme='streamlit', #Add theme color to the table
        enable_enterprise_modules=True,
        height=350, 
        width='200%',
        reload_data=True
    )

    data = grid_response['data']
    selected = grid_response['selected_rows'] 
    df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
    return df

if submit_button:
    rfc_full = model_loader("smote_rfc_full.joblib")
    if synth_type=="Synthetic":
        rfc_synth = model_loader("smote_rfc_synth.joblib")
        seq_preds_synth = get_truncated_data(seq, rfc_synth, min_length) 
    seq_preds_full = get_truncated_data(seq, rfc_full, min_length)


    st.header("Prediction Tables")

    if synth_type=="Synthetic":
        df = pd.DataFrame(seq_preds_synth[0], columns=["Sequence"])
        df["Predicted Antiviral Probability"] = seq_preds_synth[2]
        df["Possible Antiviral?"] = df["Predicted Antiviral Probability"].apply(lambda x: "Yes" if x > threshold else "No" )

        st.subheader(f"Using model trained on {synth_type} Peptides")
        customise_aggrid_df(df)

    df_full = pd.DataFrame(seq_preds_full[0], columns=["Sequence"])
    df_full["Predicted Antiviral Probability"] = seq_preds_full[2]
    df_full["Possible Antiviral?"] = df_full["Predicted Antiviral Probability"].apply(lambda x: "Yes" if x > threshold else "No" )

    st.subheader(f"Using model trained on both Ribosomal and Synthetic Peptides")
    customise_aggrid_df(df_full)
