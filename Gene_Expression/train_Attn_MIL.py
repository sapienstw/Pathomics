import os
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from dataset import WSI_feature_Dataset
from model import Feature2Pathway_baseline, Feature2Pathway_CNN, Feature2Pathway_SAPool_fc, Feature2Pathway_Attn
from utils import stat_corr, find_inter_corr_pathway

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
my_seed = 880310
same_seeds(my_seed)

ap = argparse.ArgumentParser()
ap.add_argument("-exp_name", "--exp_name", type=str, required=True, help='name of experiment, I will create a folder to store all data from this script (including models and figures).')
ap.add_argument("-feature_root", "--feature_root", type=str, required=True, help='feature folder including training, validation and testing')
ap.add_argument("-gene_path", "--gene_path", type=str, required=False, help='path of gene expression data')
ap.add_argument("-protein_path", "--protein_path", type=str, required=False, help='path of protein expression data')
ap.add_argument("-model_arch", "--model_arch", type=str, required=True, help='which model architecture you want to use in this experiment')
ap.add_argument("-predict_target", "--predict_target", type=str, required=True, help='dicide target genes and proteins')
ap.add_argument("-model_size", "--model_size", type=str, required=True, help='large or medium or small')
ap.add_argument("-k", "--k", type=int, required=True, help='number of clusters')
ap.add_argument("-gated", "--gated", type=bool, required=False, help='only for Feature2Pathway_Attn model architecture')
ap.add_argument("-lr", "--lr", type=float, required=True, help='initial learning rate')
ap.add_argument("-weight_decay", "--weight_decay", type=float, required=True, help='weight decay of the optimzer')
ap.add_argument("-n_epochs", "--n_epochs", type=int, required=True, help='how many epochs')
ap.add_argument("-batch_size_one_gpu", "--batch_size_one_gpu",  type=int, required=True, help='batch size of each GPU')
args = vars(ap.parse_args())

print('Experiment Name: %s' %(args['exp_name']))
print('Predict Target : %s' %(args['predict_target']))
print('feature_root: %s' %args['feature_root'])
print('Model Architecture: %s' %args['model_arch'])
print('KMeans k = %d' %args['k'])
print('Learning rate = %f' %(args['lr']))
print('Weight decay = %f' %(args['weight_decay']))

# Create a folder to store all results of this experiment
work_dir = '/staging/biology/r10945019/expression/'
if os.path.isdir(os.path.join(work_dir, args['exp_name'])) == False:
    os.mkdir(os.path.join(work_dir, args['exp_name']))
print('Data will be stored in %s folder!' %(os.path.join(work_dir, args['exp_name'])))
exp_folder = os.path.join(work_dir, args['exp_name'])

if torch.cuda.is_available():
    device = 'cuda'
else:
    raise BaseException("GPU is not available!")
num_gpu = torch.cuda.device_count()
print('Using', num_gpu, 'GPUs!')
print('cuda version:', torch.version.cuda)
print('Batch size of all GPUs = %d' %(num_gpu * args['batch_size_one_gpu']))

batch_size = args['batch_size_one_gpu'] * num_gpu

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

if args['predict_target'].split('_')[0] == 'gene':
    train_dataset = WSI_feature_Dataset(args['feature_root'], 'train', int(args['predict_target'].split('_')[-1]), args['gene_path'])
    valid_dataset = WSI_feature_Dataset(args['feature_root'], 'valid', int(args['predict_target'].split('_')[-1]), args['gene_path'])
elif args['predict_target'].split('_')[0] == 'protein':
    train_dataset = WSI_feature_Dataset(args['feature_root'], 'train', int(args['predict_target'].split('_')[-1]), args['protein_path'])
    valid_dataset = WSI_feature_Dataset(args['feature_root'], 'valid', int(args['predict_target'].split('_')[-1]), args['protein_path'])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6*num_gpu, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=6*num_gpu, shuffle=False)


# Get how many target
if args['predict_target'] == 'gene_1':
    class_num = 318
elif args['predict_target'] == 'gene_2':
    class_num = 66
elif args['predict_target'] == 'gene_3':
    class_num = 28
elif args['predict_target'] == 'gene_4':
    class_num = 15
elif args['predict_target'] == 'gene_5':
    class_num = 128
elif args['predict_target'] == 'protein_1':
    class_num = 25
elif args['predict_target'] == 'protein_2':
    class_num = 14
elif args['predict_target'] == 'protein_3':
    class_num = 8
elif args['predict_target'] == 'protein_4':
    class_num = 7
elif args['predict_target'] == 'protein_5':
    class_num = 17

# Define the model
model = Feature2Pathway_Attn(k=args['k'], size=args['model_size'], class_num=class_num, gated=args['gated'])
model = model.cuda()

# Define the loss function and optimizer
# Use MSE to optimize the model, and use MAE to be the metric
criterion_MSE = nn.MSELoss()
criterion_MAE = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

best_corr = 0
train_MAE_record, train_corr_record, train_good_corr_record = [], [], []   # record overall training MAE, overall correlation, and how many 'well-predicted' pathways of each epoch
valid_MAE_record, valid_corr_record, valid_good_corr_record = [], [], []   # record overall validation MAE, overall correlation, and how many 'well-predicted' patheays of each epoch
intersect_pathway_record = []   # record intersection of training and validation dataset of each epoch

for epoch in range(args['n_epochs']):

    model.train()
    MSE_record, MAE_record = [], []        # record MSE and MAE of each training step
    labels_record, preds_record = [], []   # record y_true and y_pred of each training step. This will be used to compute correlation later.
    
    for slide_name, features, label, target_names in tqdm(train_dataloader):

        if args['model_arch'] == 'Feature2Pathway_Attn':
            preds, alpha = model(features.to(device))
        else:
            preds = model(features.to(device))
        
        loss_MSE = criterion_MSE(preds, label.to(device))   # compute the MSE of this step
        loss_MAE = criterion_MAE(preds, label.to(device))   # compute the MAE of this step
        optimizer.zero_grad()   # reset the gradient
        loss_MSE.backward()     # back propagation
        optimizer.step()        # optimize the model
        MSE_record.append(loss_MSE.item())
        MAE_record.append(loss_MAE.item())
        labels_record.extend(label.detach().cpu().numpy().tolist())
        preds_record.extend(preds.detach().cpu().numpy().tolist())
    
    train_MSE = sum(MSE_record) / len(MSE_record)   # compute the training MSE of this epoch
    train_MAE = sum(MAE_record) / len(MAE_record)   # compute the training MAE of this epoch
    train_MAE_record.append(train_MAE)

    # Compute the correlation of this epoch (see more detail in util.py file)
    train_corr_list, train_pathway_corr_00_25, train_pathway_corr_25_50, train_pathway_corr_50_75, train_pathway_corr_75_00 = stat_corr(labels_record = labels_record, preds_record = preds_record)
    train_corr = np.nanmean(train_corr_list)                   # compute overall training correlation of this epoch
    train_good_pathway = train_pathway_corr_50_75 + train_pathway_corr_75_00   # how many genes/proteins get correlation >= 0.5
    train_corr_record.append(train_corr)
    train_good_corr_record.append(train_good_pathway)

    model.eval()
    MSE_record, MAE_record = [], []        # record MSE and MAE of each validation step
    labels_record, preds_record = [], []   # record y_true and y_pred of each validation step. This will be used to compute correlation later.

    for slide_name, features, label, target_names in tqdm(valid_dataloader):

        with torch.no_grad():   # Don't calculate gradient in validation phase
            if args['model_arch'] == 'Feature2Pathway_Attn':
                preds, alpha = model(features.to(device))
            else:
                preds = model(features.to(device))
        
        loss_MSE = criterion_MSE(preds, label.to(device))   # compute the MSE of this step
        loss_MAE = criterion_MAE(preds, label.to(device))   # compute the MAE of this step
        MSE_record.append(loss_MSE.item())
        MAE_record.append(loss_MAE.item())
        labels_record.extend(label.detach().cpu().numpy().tolist())
        preds_record.extend(preds.detach().cpu().numpy().tolist())

    valid_MSE = sum(MSE_record) / len(MSE_record)   # compute the validation MSE of this epoch
    valid_MAE = sum(MAE_record) / len(MAE_record)   # compute the validation MAE of this epoch
    valid_MAE_record.append(valid_MAE)

    # Compute the correlation of this epoch (see more detail in util.py file)
    valid_corr_list, valid_pathway_corr_00_25, valid_pathway_corr_25_50, valid_pathway_corr_50_75, valid_pathway_corr_75_00 = stat_corr(labels_record = labels_record, preds_record = preds_record)
    valid_corr = np.nanmean(valid_corr_list)                   # compute overall training correlation of this epoch
    valid_good_pathway = valid_pathway_corr_50_75 + valid_pathway_corr_75_00   # how many genes/proteins get correlation >= 0.5
    valid_corr_record.append(valid_corr)
    valid_good_corr_record.append(valid_good_pathway)

    # Find which genes/proteins occur in both training phase and validation phase
    intersect_omics = find_inter_corr_pathway(corr_record_1=train_corr_list, corr_record_2=valid_corr_list, target_names=target_names)
    intersect_pathway_record.append(len(intersect_omics))

    print('\n')
    print(f"[ Train | {epoch + 1:03d}/{args['n_epochs']:03d} ] MSE = {train_MSE:.5f}, MAE = {train_MAE:.5f}, Correlation = {train_corr:.5f}")
    print(f"[ Valid | {epoch + 1:03d}/{args['n_epochs']:03d} ] MSE = {valid_MSE:.5f}, MAE = {valid_MAE:.5f}, Correlation = {valid_corr:.5f}")
    print(f"[ Train | {epoch + 1:03d}/{args['n_epochs']:03d} | Omics-level Spearman Correlation ] {train_pathway_corr_50_75:03d} in (0.50 ~ 0.75), {train_pathway_corr_75_00:03d} in (0.75 ~ 1.00)")
    print(f"[ Valid | {epoch + 1:03d}/{args['n_epochs']:03d} | Omics-level Spearman Correlation ] {valid_pathway_corr_50_75:03d} in (0.50 ~ 0.75), {valid_pathway_corr_75_00:03d} in (0.75 ~ 1.00)")
    print(f"{len(intersect_omics):03d} genes/proteins have Spearman Correlation >= 0.5 in both training and validation datasets.")
    print(intersect_omics)

    # Set a checkpoint to save the model with most genes/proteins getting validation correlation >= 0.5
    if valid_good_pathway > best_corr:
        best_corr = valid_good_pathway
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(exp_folder, 'best_corr.ckpt'))
        print(f"Saving model with {best_corr} genes/proteins have Spearman correlation >= 0.5 in validation dataset.")

# Save the final-epoch model
#torch.save(model.state_dict(), os.path.join(exp_folder, 'last.ckpt'))
print('Best checkpoint at %d epoch!' %(best_epoch))

# Plot learning curve (MAE)
plt.style.use('ggplot')
plt.figure(figsize = (8, 8))
plt.plot(train_MAE_record, label = 'Training Dataset')
plt.plot(valid_MAE_record, label = 'Validation Dataset')
plt.title('Learning Curve (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(loc = 'best')
plt.savefig(os.path.join(exp_folder, 'MAE_learning_curve.svg'), format = 'SVG', dpi = 300)
plt.close()

# Plot learning curve (how many pathways get training correlation >= 0.5 in each epoch)
plt.clf()
plt.plot(train_good_corr_record, label = 'Training Dataset')
plt.title('Training Dataset Learning Curve (R >= 0.5)')
plt.xlabel('Epoch')
plt.ylabel('Number of Pathways with R >= 0.5')
plt.legend(loc = 'best')
plt.savefig(os.path.join(exp_folder, 'Corr_learning_curve_train.svg'), format = 'SVG', dpi = 300)
plt.close()

# Plot learning curve (how many pathways get validation correlation >= 0.5 in each epoch)
plt.clf()
plt.plot(valid_good_corr_record, label = 'Validation Dataset')
plt.title('Validation Dataset Learning Curve (R >= 0.5)')
plt.xlabel('Epoch')
plt.ylabel('Number of Pathways with R >= 0.5')
plt.legend(loc = 'best')
plt.savefig(os.path.join(exp_folder, 'Corr_learning_curve_valid.svg'), format = 'SVG', dpi = 300)
plt.close()

# Plot learning curve (Overall correlation)
plt.clf()
plt.plot(train_corr_record, label = 'Training Dataset')
plt.plot(valid_corr_record, label = 'Validation Dataset')
plt.xlabel('Epoch')
plt.ylabel('Pearson Correlation')
plt.legend(loc = 'best')
plt.savefig(os.path.join(exp_folder, 'Overall_Corr_learning_curve.svg'), format = 'SVG', dpi = 300)
plt.close()
