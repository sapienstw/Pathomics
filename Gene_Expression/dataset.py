import os
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset

from utils import get_target

# Gene Lists
pathway1=['BCAR1', 'CRK', 'CTRC', 'CTSG', 'DOCK1', 'EGFR', 'ELANE', 'FGA', 'FGB', 'FGG', 'FN1', 'FPR1', 'FPR2', 
          'FPR3', 'GPLD1', 'HGF', 'ITGA3', 'ITGA5', 'ITGAM', 'ITGAV', 'ITGB1', 'ITGB2', 'ITGB3', 'ITGB5', 'KLK4',
          'LRP1', 'MMP12', 'MMP13', 'MMP3', 'MMP9', 'NCL', 'PDGFD', 'PDGFRB', 'PLAU', 'PLAUR', 'PLG', 'RAC1', 
          'SERPINE1', 'SRC', 'TGFB1', 'VLDLR', 'VTN']

pathway2=['ARHGEF11', 'ARHGEF12', 'CDC42', 'ERBB2', 'LIMK1', 'LIMK2', 'MYH10', 'MYH11', 'MYH14', 'MYH9', 'MYL12B', 
          'MYL6', 'MYL9', 'PLXNB1', 'RAC1', 'RAC2', 'RHOA', 'RHOB', 'RHOC', 'RHOG', 'RND1', 'ROCK1', 'ROCK2', 'SEMA4D']

pathway3=['AKT1', 'B2M', 'BCL10', 'CARD11', 'CBL', 'CD247', 'CD28', 'CD3D', 'CD3E', 'CD3G', 'CD80', 'CD86', 'CD8A', 'CD8B',
          'CHUK', 'CSK', 'FYN', 'GRAP2', 'GRB2', 'HLA-A', 'HRAS', 'IKBKB', 'IKBKG', 'LAT', 'LCK', 'LCP2', 'MALT1', 'MAP3K14',
          'MAP3K8', 'NRAS', 'ORAI1', 'PAG1', 'PDPK1', 'PLCG1', 'PRF1', 'PRKCA', 'PRKCB', 'PRKCE', 'PRKCQ', 'PTPN6', 'PTPRC', 
          'RAP1A', 'RASGRP1', 'RASGRP2', 'RASSF5', 'SOS1', 'STIM1', 'TRAF6', 'TRPV6', 'VAV1', 'ZAP70']

pathway4=['BCL2', 'BCL2L1', 'CCNA2', 'CCND2', 'CCND3', 'CDK6', 'ELF1', 'FASLG', 'FOXP3', 'GAB2', 'GRB2', 'IL2', 'IL2RA',
          'IL2RB', 'IL2RG', 'IL4', 'JAK1', 'JAK3', 'LCK', 'LTA', 'MYC', 'PIK3CA', 'PIK3R1', 'PRF1', 'PTPN11', 'SHC1', 
          'SOS1', 'SP1', 'STAT5A', 'STAT5B']

pathway5=['AKT1', 'AKT1S1', 'AKT2', 'AKT3', 'BAD', 'BTC', 'CASP9', 'CD19', 'CD28', 'CD80',
          'CD86', 'CDKN1A', 'CDKN1B', 'CHUK', 'CREB1', 'EGF', 'EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'EREG', 'FGF1', 'FGF10', 
          'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF2', 'FGF20', 'FGF22', 'FGF23', 'FGF3','FGF4', 'FGF5', 'FGF6', 'FGF7', 
          'FGF8', 'FGF9', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'FOXO1', 'FOXO3', 'FOXO4', 'FRS2', 'FYN', 'GAB1', 'GRB2', 
          'GSK3A', 'GSK3B', 'HBEGF', 'IRS1', 'IRS2', 'KIT', 'KITLG', 'KL', 'KLB', 'LCK', 'MAPKAP1', 'MDM2', 
          'MLST8', 'MOV10', 'MTOR', 'NR4A1', 'NRG1', 'NRG2', 'NRG3', 'NRG4', 'PDGFA', 'PDGFB', 'PDGFRA',
          'PDGFRB', 'PDPK1', 'PHLPP1', 'PHLPP2', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'PIK3R2', 'PIK3R3', 'PTEN', 
          'PTPN11', 'RICTOR', 'RPS6KB2', 'THEM4', 'TNRC6A', 'TNRC6B', 'TNRC6C', 'TRAT1', 'TRIB3', 'TSC2', 'VAV1']

pathway6=['CISH', 'CSF2RB', 'GRB2', 'IL5', 'IL5RA', 'JAK2', 'LYN', 'PIK3CA', 'PIK3R1', 'PIM1', 'PTPN11', 'SDCBP', 
          'STAT5A', 'STAT5B']

pathway7=['AICDA', 'AKT1', 'ALOX15', 'ARG1', 'BCL6', 'CBL', 'CCL11', 'CCL17', 'CCL26', 'CD40LG', 'CEBPB',
          'COL1A1', 'COL1A2', 'DOK2', 'EGR2', 'ETS1', 'FCER2', 'FES', 'GRB2', 'GTF3A', 'HMGA1', 'IL10', 'IL13RA1',
          'IL13RA2', 'IL2RG', 'IL4', 'IL4R', 'IL5', 'IRF4', 'IRS1', 'IRS2', 'ITGB3', 'JAK1', 'JAK2', 'JAK3',
          'LTA', 'MAPK14', 'MTOR', 'MYB', 'MYBL1', 'OPRM1', 'PARP14', 'PIGR', 'PIK3CA', 'PIK3R1', 'PTPN6', 
          'RETNLB', 'RPS6KB1', 'SELP', 'SHC1', 'SOCS1', 'SOCS3', 'SOCS5', 'SP1', 'SPI1', 'STAT5A', 'STAT5B',
          'STAT6', 'TFF3', 'THY1']

pathway8=['BCL2', 'CDK2', 'CISH', 'DOK2', 'FOS', 'FYN', 'GAB2', 'GRB2', 'HRAS', 'IFNG', 'IKZF3', 'IL2', 'IL2RA', 
          'IL2RB', 'IL2RG', 'IRS1', 'IRS2', 'JAK1', 'JAK3', 'JUN', 'LCK', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK11', 
          'MAPK14', 'MAPK3', 'MAPK8', 'MAPK9', 'MAPKAPK2', 'MYC', 'NRAS', 'PIK3CA', 'PIK3R1', 'PPP2R5D', 'PRKCB', 'PRKCE',
          'PTK2B', 'PTPN11', 'RAF1', 'RASA1', 'RHOA', 'SHC1', 'SOCS1', 'SOCS2', 'SOCS3', 'SOS1', 'STAM', 'STAM2', 'STAT1', 
          'STAT3', 'STAT5A', 'STAT5B', 'SYK']

pathway9=['CD247', 'DOCK2', 'ELMO1', 'FYN', 'HCK', 'LCK', 'PAK2', 'RAC1']

pathway10=['AKT1', 'BAD', 'BCL2', 'BCL2L1', 'CBL', 'CCNA1', 'CCNB1', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CCNH', 'CRKL', 
          'E2F1', 'FAS', 'FASLG', 'FOS', 'GRB2', 'HRAS', 'IKZF3', 'IL2', 'IL2RA', 'IL2RB', 'IL2RG', 'IRS1', 'JAK1', 'JAK3', 
          'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'MYC', 'NMI', 'PCNA', 'PDPK1', 'PIK3CA', 'PIK3R1', 'PPIA', 'PTPN6', 'RAF1', 
          'RB1', 'RPS6KB1', 'SHC1', 'SOCS1', 'SOCS3', 'SOS1', 'STAT5A', 'STAT5B', 'TFDP1']

pathway11=['CNDP2', 'GCLC', 'GCLM', 'GGCT', 'GGT1', 'GGT3P', 'GGT5', 'GGT6', 'GGT7', 'GSS', 'OPLAH']

all_genes = pathway1 + pathway2 + pathway3 + pathway4 + pathway5 + pathway6 + pathway7 + pathway8 + pathway9 + pathway10 + pathway11
gene_counter = Counter(all_genes)

# Protein List
protein_counter = {
    'STAT5ALPHA': 5, 'LCK': 5, 'AKT': 4, 'AKTPS473': 4, 'AKTPT308': 4, 'IRS1': 4, 'MAPKPT202Y204': 4, 'BCL2': 3,
    'EGFR': 2, 'EGFRPY1068': 2, 'EGFRPY1173': 2, 'GAB2': 2, 'MTOR': 2, 'MTORPS2448': 2, 'CYCLINEB1': 1, 'CYCLINED1': 1,
    'CYCLINEE1': 1, 'FOXO3A': 1, 'GSK3ALPHABETA': 1, 'GSK3ALPHABETAPS21S9': 1, 'PCNA': 1, 'PTEN': 1, 'SRC': 1,
    'STAT3PY705': 1, 'SYK': 1
}

# For patch-level prediction model and feature extraction model
class WSI_Baseline_Dataset(Dataset):
    def __init__(self, image_folder, transforms, mode, txt_path=None):
        """
        image_folder: directory containing slides in that dataset
        transforms: image augmentation
        mode: 'train' or 'valid' or 'test'
        txt_path: PARADIGM data path
        """
        self.image_folder = image_folder
        self.slides = os.listdir(image_folder)   # record all slide names
        
        self.patches = []   # record all patch names from every slides
        for slide in self.slides:
            for patch in os.listdir(os.path.join(image_folder, slide)):
                self.patches.append(patch)
        
        print(f"{len(self.slides)} slides in {mode+'ing'} dataset.")
        print(f"{len(self.patches)} patches in {mode+'ing'} dataset.")

        self.transforms = transforms
        self.mode = mode
        self.PARADIGM = pd.read_table(txt_path, index_col='sample')
    
    def __len__(self):
        return len(self.patches)   # number of patches in this dataset

    def __getitem__(self, idx):
        patch_name = self.patches[idx]   # pick a patch from self.patches list
        slide_name = patch_name.split('_')[0]
        patch_path = os.path.join(self.image_folder, slide_name, patch_name)
        image = Image.open(patch_path)   # read the patch
        image = self.transforms(image)   # do augmentation
        if self.PARADIGM is not None:
            label = self.PARADIGM[slide_name[:15]].values   # get the label
            label = torch.FloatTensor(label)
            return slide_name, patch_name, image, label
        else:
            return slide_name, patch_name, image
        
class WSI_Dataset(Dataset):
    def __init__(self, k, image_folder, center_path, transforms, mode, txt_path=None):
        self.k = k
        self.image_folder = image_folder
        self.slides = os.listdir(image_folder)
        self.center_path = center_path
        self.transforms = transforms
        self.mode = mode

        slide_list = []
        if self.mode == 'train':
            slide_list_path = '/staging/biology/r10945019/Gene_Expression_Training_Slide.csv'
        elif self.mode == 'valid':
            slide_list_path = '/staging/biology/r10945019/Gene_Expression_Validation_Slide.csv'
        elif self.mode == 'test':
            slide_list_path = '/staging/biology/r10945019/Gene_Expression_Testing_Slide.csv'
        with open(slide_list_path, 'r', encoding='utf-8-sig') as f:
            csvreader = csv.reader(f, delimiter=' ', quotechar='|')
            for row in csvreader:
                slide_list.extend(row)

        if txt_path is not None:
            self.PARADIGM = pd.read_table(txt_path, index_col = 'sample')        

        print(f"{len(self.slides)} slides in {self.mode+'ing'} dataset.")

    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, idx):
        slide_name = self.slides[idx]
        center_record = pd.read_csv(os.path.join(self.center_path, slide_name+'.csv'))
        images = []
        for i in range(self.k):
            feature_name = center_record.loc[i][0]
            image_path = os.path.join(self.image_folder, slide_name, feature_name[:-4])
            image = Image.open(image_path)
            image = self.transforms(image)
            images.append(image)
        if self.PARADIGM is not None:
            label = self.PARADIGM[slide_name[:15]].values
            label = torch.FloatTensor(label)
            return slide_name, images, label
        else:
            return slide_name, images

# For all Feature2Pathway models (take feature as input)
# Each feature are in .npy format
class WSI_feature_Dataset(Dataset):
    def __init__(self, feature_folder, mode, predict_target, gene_label_path=None, protein_label_path=None):
        self.feature_folder = feature_folder
        self.mode = mode
        self.predict_target = predict_target
        #self.no_gene_label = ['GGT2', 'MIR26A1', 'AGO3', 'AGO2', 'MIR26A2', 'AGO4', 'AGO1']

        ID_list = []
        if self.mode == 'train':
            ID_list_path = '/staging/biology/r10945019/Gene_Expression_Training_Slide.csv'
        elif self.mode == 'valid':
            ID_list_path = '/staging/biology/r10945019/Gene_Expression_Validation_Slide.csv'
        elif self.mode == 'test':
            ID_list_path = '/staging/biology/r10945019/Gene_Expression_Testing_Slide.csv'
        with open(ID_list_path, 'r', encoding='utf-8-sig') as f:
            csvreader = csv.reader(f, delimiter=' ', quotechar='|')
            for row in csvreader:
                ID_list.extend(row)

        self.slides = []
        for slide_name in os.listdir(feature_folder):
            if slide_name[:12] in ID_list:
                self.slides.append(slide_name)
        print(f"{len(self.slides)} slides in {mode+'ing'} dataset.")

        if (gene_label_path is not None) & (protein_label_path is None):
            # For Gene Expression Prediction
            self.gene_expression = pd.read_table(gene_label_path, index_col='sample').transpose()
            self.protein_expression = None
        elif (gene_label_path is None) & (protein_label_path is not None):
            # For Protein Expression Prediction
            self.protein_expression = pd.read_table(protein_label_path, index_col='Sample_description').transpose()
            self.gene_expression = None


    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, idx):
        slide_name = self.slides[idx]
        slide_path = os.path.join(self.feature_folder, slide_name)
        feature_names = os.listdir(slide_path)   # k features
        features = np.array([np.load(os.path.join(slide_path, fname)) for fname in feature_names])   # a ndarray (k, 6144)
        features = torch.from_numpy(features)   # convert into a torch.tensor (k, 6144)

        if (self.gene_expression is not None) & (self.protein_expression is None):
            target_genes = get_target(self.predict_target, 'gene')

            gene_label = self.gene_expression.loc[slide_name[:15]][target_genes].tolist()
            gene_label = torch.FloatTensor(gene_label)

            return slide_name, features.float(), gene_label, target_genes
        
        elif (self.gene_expression is None) & (self.protein_expression is not None):
            target_proteins = get_target(self.predict_target, 'protein')

            protein_label = self.protein_expression.loc[slide_name[:15]][target_proteins].tolist()
            protein_label = torch.FloatTensor(protein_label)

            return slide_name, features.float(), protein_label, target_proteins
            
        elif (self.gene_expression is None) & (self.protein_expression is None):
            
            return slide_name, features.float()