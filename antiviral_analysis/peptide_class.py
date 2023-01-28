import numpy as np
import peptides
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from Bio.SeqUtils import ProtParam

class Peptide():
    def __init__(self, sequence):
        self.sequence = sequence
        self.desc = PeptideDescriptor(self.sequence.upper(),"eisenberg")
        self.glob = GlobalDescriptor(self.sequence.upper())
        self.pep_pkg_obj = peptides.Peptide(self.sequence)
        
    def cysteine_count(self):
        return self.sequence.upper().count("C")
    
    def hydrophobic_moment(self):
        try:
            self.desc.calculate_moment()
            return float(self.desc.descriptor)
        except KeyError:
            return np.nan

    def sequence_charge(self):
        try:
            self.glob.calculate_charge()
            return(float(self.glob.descriptor))
        except KeyError:
            return np.nan

    def charge_density(self):
        try:
            self.glob.charge_density()
            return(float(self.glob.descriptor))
        except KeyError:
            return np.nan

    def isoelectric_point(self):
        try:
            self.glob.isoelectric_point()
            return(float(self.glob.descriptor))
        except KeyError:
            return np.nan

    def instability_index(self):
        try:
            self.glob.instability_index()
            return(float(self.glob.descriptor))
        except KeyError:
            return np.nan

    def aromaticity(self):
        try:
            self.glob.aromaticity()
            return(float(self.glob.descriptor))
        except:
            return np.nan

    def aliphatic_index(self):
        try:
            self.glob.aliphatic_index()
            return(float(self.glob.descriptor))
        except KeyError:
            return np.nan

    def boman_index(self):
        try:
            self.glob.boman_index()
            return(float(self.glob.descriptor))
        except KeyError:
            return np.nan
    
    def polarity(self):
        try:
            return(float(self.pep_pkg_obj.cruciani_properties()[0]))
        except:
            np.nan
            
    def h_bonding(self):
        try:
            return(float(self.pep_pkg_obj.cruciani_properties()[2]))
        except:
            np.nan
            
    def bulky_properties(self):
        try:
            return(float(self.pep_pkg_obj.fasgai_vectors()[2]))
        except:
            np.nan
    
    def compositional_char_index(self):
        try:
            return(float(self.pep_pkg_obj.fasgai_vectors()[3]))
        except:
            np.nan
    
    def local_flexibility(self):
        try:
            return(float(self.pep_pkg_obj.fasgai_vectors()[4]))
        except:
            np.nan
    
    def electronic_props(self):
        try:
            return(float(self.pep_pkg_obj.fasgai_vectors()[5]))
        except:
            np.nan
    
    def helix_bend_pref(self):
        try:
            return(float(self.pep_pkg_obj.kidera_factors()[0]))
        except:
            np.nan
    
    def side_chain_size(self):
        try:
            return(float(self.pep_pkg_obj.kidera_factors()[1]))
        except:
            np.nan
    
    def ext_struct_pref(self):
        try:
            return(float(self.pep_pkg_obj.kidera_factors()[2]))
        except:
            np.nan
    
    def double_bend_pref(self):
        try:
            return(float(self.pep_pkg_obj.kidera_factors()[4]))
        except:
            np.nan
    
    def partial_specific_volume(self):
        try:
            return(float(self.pep_pkg_obj.kidera_factors()[5]))
        except:
            np.nan
    
    def flat_extended_pref(self):
        try:
            return(float(self.pep_pkg_obj.kidera_factors()[6]))
        except:
            np.nan
    
    def pK_C(self):
        try:
            return(float(self.pep_pkg_obj.kidera_factors()[8]))
        except:
            np.nan
    
    def ms_whim_scores(self):
        try:
            mswhim_scores = self.pep_pkg_obj.ms_whim_scores()
            return mswhim_scores[0], mswhim_scores[1], mswhim_scores[2]
        except:
            np.nan, np.nan, np.nan

    def percent_helix(self):
        return ProtParam.ProteinAnalysis(str(self.sequence).upper()).secondary_structure_fraction()[0]

    def percent_turn(self):
        return ProtParam.ProteinAnalysis(str(self.sequence).upper()).secondary_structure_fraction()[1]

    def percent_sheet(self):
        return ProtParam.ProteinAnalysis(str(self.sequence).upper()).secondary_structure_fraction()[2]
