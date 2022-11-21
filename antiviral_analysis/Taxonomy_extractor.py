from Bio import Entrez
from tqdm.auto import tqdm
import numpy as np

def get_tax_id(species):
    """Retrieves the taxid required to get data from ncbi taxomomy database, 
    by passing the the species name to Entrez.esearch()"""
    
    spec = species.replace('-', " ").split(" ")[:2]
    spec = " ".join(spec).replace(' ', "+").strip()
    search = Entrez.esearch(term = spec, db = "taxonomy", retmode = "xml")
    record = Entrez.read(search)
    
    if record["IdList"] == []:
        spec = species.split(" ")[0].strip()
        search = Entrez.esearch(term = spec, db = "taxonomy", retmax = 1, retmode = "xml")
        record = Entrez.read(search)
        try: 
            taxid = record['IdList'][0]
        except IndexError:
            taxid = record["IdList"]
    else:
        taxid = record["IdList"][0]        
    return taxid
        
def get_tax_data(taxid):
    """fetches record from NCBI using retrieved taxid"""
    
    search = Entrez.efetch(id = taxid, db = "taxonomy", retmode = "xml")
    return Entrez.read(search)


def get_sk_id(species, taxid):
    cancer_class = ["carcinoma","sarcoma","myeloma","melanoma","blastoma","leukemia","lymphoma","cytoma",
                "thelioma","neoplas","tumor","cancer"]
    if "virus" in species.lower() or "denv" in species.lower() or "hsv" in species.lower() or "fcov" in species.lower():
        data = np.nan
        skdom = "Viruses"
    elif any(cancer in species.lower() for cancer in cancer_class):
        data = np.nan
        skdom = "Tumour"
    else:
        if len(taxid) != 0:
            data = get_tax_data(taxid)
            skdom = [d['ScientificName'] for d in data[0]['LineageEx'] if d['Rank'] in ['superkingdom']][0]

        elif "-" in species:
            try:
                spec = species.split("-")[0].strip()
                taxid = get_tax_id(spec)
                data = get_tax_data(taxid)
                skdom = [d['ScientificName'] for d in data[0]['LineageEx'] if d['Rank'] in ['superkingdom']][0]
            except RuntimeError:
                taxid = get_tax_id(species)
                data = get_tax_data(taxid)
                skdom = [d['ScientificName'] for d in data[0]['LineageEx'] if d['Rank'] in ['superkingdom']][0]

        else:
            data = "NotFound"
            skdom = "NotFound"
    return data, skdom


def superkingdom_id(species_series, email ="", verbose=0, prog_bar=False):
    Entrez.email = email
    sk_dict = {}
    taxid_list = []

    species_list = list(species_series.unique())
    
    if prog_bar == True:
        for species in tqdm(species_list, position=0, leave=True):
            if verbose == 1:
                print ('\t'+species) 
            taxid = get_tax_id(species) 
            taxid_list.append(taxid)
            sk_dict[species] = get_sk_id(species, taxid)[1]
    else:
        for species in species_list:
            if verbose == 1:
                print ('\t'+species) 
            taxid = get_tax_id(species) 
            taxid_list.append(taxid)
            sk_dict[species] = get_sk_id(species, taxid)[1]

    return sk_dict
