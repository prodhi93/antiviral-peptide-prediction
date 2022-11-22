# Import libraries

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from Bio import SeqIO
from antiviral_analysis import Peptide_class, Taxonomy_extractor

# Read in CSV

amp = pd.read_csv("peptides-complete.csv")

amp = amp.drop_duplicates(subset=["SEQUENCE"]).reset_index()

amp["AV label"] = amp["TARGET GROUP"].apply(lambda x: 1 if "virus" in str(x).lower() else 0)


# Systematically go through the columns in the dataset and drop ones with information with little to 
# no relevance to the machine learning process as well as columns with a large number of null and missing values.

amp.drop(['TARGET ACTIVITY - REFERENCE','HEMOLITIC CYTOTOXIC ACTIVITY - NOTE',
       'HEMOLITIC CYTOTOXIC ACTIVITY - REFERENCE', 'SYNERGY - TARGET SPECIE',
       'SYNERGY - ACTIVITY MEASURE GROUP', 'SYNERGY - UNIT',
       'SYNERGY - PEPTIDE ID', 'SYNERGY - ANTIBIOTIC ID',
       'SYNERGY - ANTIBIOTIC NAME', 'SYNERGY - PEPTIDE ACTIVITY',
       'SYNERGY - PEPTIDE ANTIBIOTIC ACTIVITY',
       'SYNERGY - ANTIBIOTIC ACTIVITY',
       'SYNERGY - ANTIBIOTIC PEPTIDE ACTIVITY', 'SYNERGY - FICI',
       'SYNERGY - REFERENCE', 'UNIRPROT - ID', 'UNIRPROT - DESCRIPTION',
       'UNIRPROT - PRO PEPTIDE', 'UNIRPROT - URL', 'ARTICLES - JOURNAL',
       'ARTICLES - YEAR', 'ARTICLES - VOLUME', 'ARTICLES - PAGES',
       'ARTICLES - TITLE', 'ARTICLES - ADDITIONAL', 'ARTICLES - PUBMED',
       'ARTICLES - AUTHORS'],axis=1,inplace=True)


amp.drop(['PDB NAME',
       'PDB LINK', 'PDB FILE LINK'],axis=1,inplace=True)
amp.drop(['SOURCE GENE - NOTE', 'SOURCE GENE - DB LINK'],axis=1,inplace=True)
amp.drop(['INTERCHAIN BOND - NOTE','TARGET ACTIVITY - NOTE','UNUSUAL OR MODIFIED AMINO ACID - NOTE','UNUSUAL OR MODIFIED AMINO ACID - BEFORE MODIFICATION'],axis=1,inplace=True)

amp.drop(['SOURCE GENE - KINGDOM', 'SOURCE GENE - SOURCE',
       'SOURCE GENE - SUBKINGDOM', 'SOURCE GENE - GENE',
       'SOURCE GENE - GENE IN SEQUENCE'],axis=1,inplace=True)

amp.drop(['INTRACHAIN BOND - NOTE'],axis=1,inplace=True)

amp.drop("UNUSUAL OR MODIFIED AMINO ACID - POSITION",axis=1,inplace=True)

amp.drop(['HEMOLITIC CYTOTOXIC ACTIVITY - TARGET CELL',
       'HEMOLITIC CYTOTOXIC ACTIVITY - LYSIS GROUP',
       'HEMOLITIC CYTOTOXIC ACTIVITY - LYSIS VALUE',
       'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION',
       'HEMOLITIC CYTOTOXIC ACTIVITY - UNIT',
       'HEMOLITIC CYTOTOXIC ACTIVITY - PH',
       'HEMOLITIC CYTOTOXIC ACTIVITY - IONIC STRENGTH',
       'HEMOLITIC CYTOTOXIC ACTIVITY - SALT TYPE'],axis=1,inplace=True)

amp.drop(['TARGET ACTIVITY - ACTIVITY MEASURE GROUP',
       'TARGET ACTIVITY - ACTIVITY MEASURE VALUE',
       'TARGET ACTIVITY - CONCENTRATION', 'TARGET ACTIVITY - UNIT',
       'TARGET ACTIVITY - PH', 'TARGET ACTIVITY - IONIC STRENGTH',
       'TARGET ACTIVITY - SALT TYPE', 'TARGET ACTIVITY - MEDIUM',
       'TARGET ACTIVITY - CFU', 'TARGET ACTIVITY - CFU GROUP'],axis=1,inplace=True)

amp.drop("INTERCHAIN BOND - CHAIN 4", axis=1,inplace=True) # all NaN values


# Drop null value rows for columns where we crucially need the data for further machine learning processing.

amp = amp.dropna(subset=["TARGET ACTIVITY - TARGET SPECIES"])
amp = amp.dropna(subset=["SEQUENCE"])

# Reformat the sequence column to retain the first sequence where multiple sequences for a peptide are provided

amp["SEQUENCE"] = amp["SEQUENCE"].apply(lambda x: str(x.split(",")[0]))

# Drop rows where the peptide sequence contains "X"

amp.drop(index=amp[amp["SEQUENCE"].str.contains("X")].index,inplace=True)


# One-hot encoding column with N-terminus modifications

N_term = pd.get_dummies(amp["N TERMINUS"],prefix=["N-term_"])
amp = pd.concat([amp,N_term],axis=1)
amp.drop("N TERMINUS",axis=1,inplace=True)

# One-hot encoding column with peptide complexity data 

complexity = pd.get_dummies(amp["COMPLEXITY"])
amp = pd.concat([amp.drop("COMPLEXITY",axis=1),complexity],axis=1)

# One-hot encoding column with C-terminus modifications

C_term = pd.get_dummies(amp["C TERMINUS"],prefix=["C-term_"])
amp = pd.concat([amp.drop("C TERMINUS",axis=1),C_term],axis=1)

# Drop index and ID columns 

amp.drop(["index","ID"],axis=1,inplace=True)

# # One-hot encoding column with unusual/modified amino acid type data

unmodaa = pd.get_dummies(amp["UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE"],prefix="unusual/modified_aa_")
amp = pd.concat([amp.drop("UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE",axis=1),unmodaa],axis=1)


# Change values to 0/1 binary 

amp['INTERCHAIN BOND - CHAIN 1'] = amp['INTERCHAIN BOND - CHAIN 1'].apply(lambda x: 0 if math.isnan(x) else 1)
amp['INTERCHAIN BOND - CHAIN 2'] = amp['INTERCHAIN BOND - CHAIN 2'].apply(lambda x: 0 if math.isnan(x) else 1)
amp['INTERCHAIN BOND - CHAIN 3'] = amp['INTERCHAIN BOND - CHAIN 3'].apply(lambda x: 0 if math.isnan(x) else 1)

# One-hot encoding

amp = pd.concat([amp.drop('INTERCHAIN BOND - BOND',axis=1),pd.get_dummies(amp['INTERCHAIN BOND - BOND'])],axis=1)

amp["HEMOLITIC CYTOTOXIC ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)"] = amp["HEMOLITIC CYTOTOXIC ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)"].fillna(0)

amp = pd.concat([amp.drop(["INTRACHAIN BOND - BOND"],axis=1),pd.get_dummies(amp["INTRACHAIN BOND - BOND"])],axis=1)

amp.replace('Leishmania spp (promastigote)','Leishmania sp.',inplace=True)


# Assign broader taxonomy categories using data from NCBI for target species

species_col = amp["TARGET ACTIVITY - TARGET SPECIES"]
email = "prodhimanisha93@gmail.com"
sk_dict = Taxonomy_extractor.superkingdom_id(species_col, email)

amp["TARGET ACTIVITY - CELL TYPE"] = amp["TARGET ACTIVITY - TARGET SPECIES"].map(sk_dict)


# One-hot encoding and categorise target activity by broad superkingdom categories

cat_vars = pd.get_dummies(amp["TARGET ACTIVITY - CELL TYPE"],prefix="Target Activity")
cat_vars.values[cat_vars != 0] = amp["TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)"]

amp = pd.concat([amp.drop(["TARGET ACTIVITY - TARGET SPECIES","TARGET ACTIVITY - CELL TYPE",
                                  "TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)"],axis=1),cat_vars],axis=1)


# Derive engineered features based on peptide sequence data
# (particularly useful for making predictions off peptides where higher level
# data from complex experiments are difficult to obtain)

amp["HELIX"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).percent_helix())
amp["TURN"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).percent_turn())
amp["SHEET"]= amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).percent_sheet())
amp["HYDROPHOBIC MOMENT"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).hydrophobic_moment())
amp["SEQUENCE CHARGE"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).sequence_charge())
amp["CHARGE DENSITY"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).charge_density())
amp["ISOELECTRIC POINT"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).isoelectric_point())
amp["INSTABILITY INDEX"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).instability_index())
amp["AROMATICITY"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).aromaticity())
amp["ALIPHATIC INDEX"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).aliphatic_index())
amp["BOMAN INDEX"] = amp["SEQUENCE"].apply(lambda x: Peptide_class.Peptide(x).boman_index())

amp.to_csv('cleaned_amp.csv')

# Create two distinct datasets separated by ribosomal vs synthetic peptides

amp_ribosomal = amp[amp["SYNTHESIS TYPE"] == "Ribosomal"]
amp_ribosomal.to_csv("amp_ribosomal.csv")

amp_synthetic = amp[amp["SYNTHESIS TYPE"] == "Synthetic"]
amp_synthetic.to_csv("amp_synthetic.csv")
