import numpy as np
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from Bio.SeqUtils import ProtParam

class Peptide():
    def __init__(self, sequence):
        self.sequence = sequence
        self.desc = PeptideDescriptor(self.sequence.upper(),"eisenberg")
        self.glob = GlobalDescriptor(self.sequence.upper())

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

    def percent_helix(self):
        return ProtParam.ProteinAnalysis(str(self.sequence).upper()).secondary_structure_fraction()[0]

    def percent_turn(self):
        return ProtParam.ProteinAnalysis(str(self.sequence).upper()).secondary_structure_fraction()[1]

    def percent_sheet(self):
        return ProtParam.ProteinAnalysis(str(self.sequence).upper()).secondary_structure_fraction()[2]
