
# Antiviral Peptide Prediction

This repository contains code to clean data, select and generate features, train a random forest classifier model, and predict the antiviral activity of a putative antimicrobial peptide.

You can run the application for prediction that uses my pre-trained model here:
https://prodhi93-antiviral-peptide-predict-streamlit-peptide-app-pm0byh.streamlit.app/


## File List

* peptides_complete.csv: Raw AMP data downloaded from https://dbaasp.org/
* cleaned_amp.json: Cleaned AMP dataset including all engineered features and one-hot encoding of existing features. Sparsely distributed features such as the microbe species against which the peptide demonstrates activity were redesigned to represent superkingdoms using taxonomy_extractor.py to provide more meaningful information for predictive model training.
* amp_ribosomal.json: Subset of cleaned_amp.json only containing peptides that are ribosomally synthesised (i.e. biologically occurring).
* amp_synthetic.json: Subset of cleaned_amp.json only containing peptides designed synthetically.
* selected_feats_amp.json: Dataset of all dataset AMPs with only selected features from the feature selection and engineering process.
* selected_feats_ribosomal.json: Subset of selected_feats_amp with only ribosomally synthesised peptides.
* selected_feats_synthetic.json: Subset of selected_feats_amp with only synthetic peptides.
## Data Cleaning

Peptide data for training was obtained from the Database of Antimicrobial Activity and Structure of Peptides (DBAASP), which contains structural and functional information on 14000+ antimicrobial compounds with a high therapeutic index. The raw dataset is included in this repository as peptides_complete.csv.

To run the data cleaning protocol from scratch on this dataset (or your own customised one, if you so wish), you can run the data_cleaning.py script. 
## Feature Selection and Engineering

The DBAASP dataset features ombined with property features engineered from the sequence itself results in a total of 350+ features. However, some of them are redundant and code for overlapping information while many others do not correlate in a significant way to the potential antiviral activity of the peptides. The feature_selection.ipynb notebook contains code for the visualisation of feature correlations and subsequent selection of the top 40 most relevant features for the prediction task.

## Model Training

Down the road, my hope is to expand this project and the app to take in all 350+ features--both default DBAASP and engineered ones--for prediction. However, recognising the limited availability of information from functional studies for new putative peptides, I chose to train predictive models which only take in input features that can be derived directly from a peptide's amino acid sequence. These are:
* Secondary structure features, namely: percent helix, percent beta sheet, and percent TURN
* Aromaticity 
* Alphatic index
* Hydrophobic moment
* Boman index
* Instability index
* Isolectric point
* Charge density
* Sequence charge
* Cysteine count (proxy for disulfide bond count)
* Polarity
* Hydrogen bonding'
* Bulky properties
* Compositional characteristic index
* Local flexibility 
* Electronic properties
* Helix bend preference
* Side chain size
* Extended structural preference
* Double bend preference
* Partial specific volume
* Flat extended preference
* pK-C
* MS-WHIM scores 1, 2 and 3

You can run training_script.py to train a random forest classifier model from scratch and obtain evaluation metrics on a test set. The peptide_model_trainer.py module in the antiviral_analysis package also includes functions to train a neural network model or a support vector classifier instead of an RFC. The RFC model consistently performed better than neural networks and SVM on all metrics on this particular dataset, which led me to use the trained RFC for the final prediction task.
## Prediction App

Finally, I built a Streamlit web app to predict potential antiviral activity from the sequence of a peptide using the trained RFC model. Since smaller subsections of a peptide can often have antiviral activity that the whole sequence may not (e.g. due to trailing amino acids which disrupt the required structure for antiviral activity), the app sequentially truncates the sequence at the end and outputs predicted antiviral activity for the whole sequence and the subsequent truncated chunks in series in a dataframe that can be sorted and filtered. The app allows you to enter a peptide sequence, a threshold value to qualify the sequence as positive for predicted antiviral activity, and  the minimum length of truncated sequence for which to output prediction. 
## References

Malak Pirtskhalava, Anthony A Amstrong, Maia Grigolava, Mindia Chubinidze, Evgenia Alimbarashvili, Boris Vishnepolsky, Andrei Gabrielian, Alex Rosenthal, Darrell E Hurt, Michael Tartakovsky, DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides as a resource for development of new therapeutics, Nucleic Acids Research, Volume 49, Issue D1, 8 January 2021, Pages D288–D297, https://doi.org/10.1093/nar/gkaa991
