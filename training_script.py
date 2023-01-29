from peptide_model_trainer import PeptideTrainer
import pandas as pd

data = pd.read_json("selected_feats_amp.json")
data_ribo = pd.read_json("selected_feats_ribosomal.json")
data_synth = pd.read_json("selected_feats_synthetic.json")

# barebones, only data that can be directly extracted from sequence
data = data[['HELIX', 'AROMATICITY','SHEET', 'ALIPHATIC INDEX', 'HYDROPHOBIC MOMENT', 'TURN', 'BOMAN INDEX',
       'INSTABILITY INDEX', 'ISOELECTRIC POINT', 'CHARGE DENSITY','SEQUENCE CHARGE', 'CYSTEINE COUNT',
        'POLARITY','H-BONDING', 'BULKY PROPERTIES', 'COMPOSITIONAL CHARACTERISTIC INDEX', 'LOCAL FLEXIBILITY',
        'ELECTRONIC PROPERTIES','HELIX BEND PREFERENCE','SIDE CHAIN SIZE','EXTENDED STRUCTURAL PREFERENCE',
        'DOUBLE BEND PREFERENCE', 'PARTIAL SPECIFIC VOLUME','FLAT EXTENDED PREFERENCE','pK-C','MS-WHIM-1 SCORE',
        'MS-WHIM-2 SCORE','MS-WHIM-3 SCORE', 'AV label']]

data_ribo = data_ribo[['HELIX', 'AROMATICITY','SHEET', 'ALIPHATIC INDEX', 'HYDROPHOBIC MOMENT', 'TURN', 'BOMAN INDEX',
       'INSTABILITY INDEX', 'ISOELECTRIC POINT', 'CHARGE DENSITY','SEQUENCE CHARGE', 'CYSTEINE COUNT',
        'POLARITY','H-BONDING', 'BULKY PROPERTIES', 'COMPOSITIONAL CHARACTERISTIC INDEX', 'LOCAL FLEXIBILITY',
        'ELECTRONIC PROPERTIES','HELIX BEND PREFERENCE','SIDE CHAIN SIZE','EXTENDED STRUCTURAL PREFERENCE',
        'DOUBLE BEND PREFERENCE', 'PARTIAL SPECIFIC VOLUME','FLAT EXTENDED PREFERENCE','pK-C','MS-WHIM-1 SCORE',
        'MS-WHIM-2 SCORE','MS-WHIM-3 SCORE', 'AV label']]


data_synth = data_synth[['HELIX', 'AROMATICITY','SHEET', 'ALIPHATIC INDEX', 'HYDROPHOBIC MOMENT', 'TURN', 'BOMAN INDEX',
       'INSTABILITY INDEX', 'ISOELECTRIC POINT', 'CHARGE DENSITY','SEQUENCE CHARGE', 'CYSTEINE COUNT',
        'POLARITY','H-BONDING', 'BULKY PROPERTIES', 'COMPOSITIONAL CHARACTERISTIC INDEX', 'LOCAL FLEXIBILITY',
        'ELECTRONIC PROPERTIES','HELIX BEND PREFERENCE','SIDE CHAIN SIZE','EXTENDED STRUCTURAL PREFERENCE',
        'DOUBLE BEND PREFERENCE', 'PARTIAL SPECIFIC VOLUME','FLAT EXTENDED PREFERENCE','pK-C','MS-WHIM-1 SCORE',
        'MS-WHIM-2 SCORE','MS-WHIM-3 SCORE', 'AV label']]

pt_rfc = PeptideTrainer(data)
pt_ribo_rfc = PeptideTrainer(data_ribo)
pt_synth_rfc = PeptideTrainer(data_synth)

pt_rfc.train_rfc_model()
pt_ribo_rfc.train_rfc_model()
pt_synth_rfc.train_rfc_model()

pt_rfc.evaluate_model()
pt_ribo_rfc.evaluate_model()
pt_synth_rfc.evaluate_model()
