import os
import numpy as np
from scipy import stats
from collections import Counter

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

def get_target(target_model, omics_type):
    if target_model == 1:
        # Model 1
        # All
        target_genes = list(set(all_genes))
        target_proteins = list(protein_counter.keys())

    elif target_model == 2:
        # Model 2
        # With occurence >= 2
        target_genes, target_proteins = [], []
        for name, fre in gene_counter.items():
            if fre >= 2:
                target_genes.append(name)
        for name, fre in protein_counter.items():
            if fre >= 2:
                target_proteins.append(name)

    elif target_model == 3:
        # Model 3
        # With occurence >= 3
        target_genes, target_proteins = [], []
        for name, fre in gene_counter.items():
            if fre >= 3:
                target_genes.append(name)
        for name, fre in protein_counter.items():
            if fre >= 3:
                target_proteins.append(name)

    elif target_model == 4:
        # Model 4
        # With occurence >= 4
        target_genes, target_proteins = [], []
        for name, fre in gene_counter.items():
            if fre >= 4:
                target_genes.append(name)
        for name, fre in protein_counter.items():
            if fre >= 4:
                target_proteins.append(name)

    elif target_model == 5:
        # Model 5
        # IL Pathway Family
        target_genes = list(set(pathway4 + pathway6 + pathway7 + pathway8 + pathway10))
        target_proteins = ['STAT5ALPHA', 'LCK', 'AKT', 'AKTPS473', 'AKTPT308', 'IRS1', 'MAPKPT202Y204',
                        'BCL2', 'GAB2', 'MTOR', 'MTORPS2448', 'CYCLINEB1', 'CYCLINED1', 'CYCLINEE1',
                        'PCNA', 'STAT3PY705', 'SYK']
    
    if omics_type == 'gene':
        return target_genes
    elif omics_type == 'protein':
        return target_proteins

def stat_corr(labels_record, preds_record):
    """
    Compute the Spearman correlation

    Input:
        labels_record
        preds_record
    """
    corr_record = []
    pathway_corr_00_25, pathway_corr_25_50, pathway_corr_50_75, pathway_corr_75_00 = 0, 0, 0, 0
    y_true = np.array(labels_record)
    y_pred = np.array(preds_record)
    for i in range(y_pred.shape[1]):
        corr, p = stats.spearmanr(y_true[:, i], y_pred[:, i])
        corr_record.append(corr)
        
        if (abs(corr) >= 0.00 and abs(corr) < 0.25): pathway_corr_00_25 += 1
        if (abs(corr) >= 0.25 and abs(corr) < 0.50): pathway_corr_25_50 += 1
        if (abs(corr) >= 0.50 and abs(corr) < 0.75): pathway_corr_50_75 += 1
        if (abs(corr) >= 0.75 and abs(corr) < 1.00): pathway_corr_75_00 += 1

    return corr_record, pathway_corr_00_25, pathway_corr_25_50, pathway_corr_50_75, pathway_corr_75_00

def find_inter_corr_pathway(corr_record_1, corr_record_2, target_names):
    hit = []
    for i in range(len(target_names)):
        if (abs(corr_record_1[i]) > 0.5) & (abs(corr_record_2[i]) > 0.5):
            hit.append(target_names[i])
    return hit
