import os
import csv
import random
import argparse
import numpy as np
from collections import Counter
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import WSI_feature_Dataset
from model import Feature2Pathway_baseline, Feature2Pathway_CNN, Feature2Pathway_SAPool_fc, Feature2Pathway_Attn
from utils import stat_corr, get_target

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
ap.add_argument("-batch_size_one_gpu", "--batch_size_one_gpu",  type=int, required=True, help='batch size of each GPU')
args = vars(ap.parse_args())

print('Experiment Name: %s' %(args['exp_name']))
print('This is for transfer learning!')
print('Predict Target : %s' %(args['predict_target']))
print('feature_root: %s' %args['feature_root'])
print('Model Architecture: %s' %args['model_arch'])
print('KMeans k = %d' %args['k'])

# Create a folder to store all results of this experiment
work_dir = '/staging/biology/r10945019/expression/'
if os.path.isdir(os.path.join(work_dir, args['exp_name'])) == False:
    os.mkdir(os.path.join(work_dir, args['exp_name']))
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
    test_dataset = WSI_feature_Dataset(args['feature_root'], 'test', int(args['predict_target'].split('_')[-1]), args['gene_path'])
elif args['predict_target'].split('_')[0] == 'protein':
    train_dataset = WSI_feature_Dataset(args['feature_root'], 'train', int(args['predict_target'].split('_')[-1]), args['protein_path'])
    valid_dataset = WSI_feature_Dataset(args['feature_root'], 'valid', int(args['predict_target'].split('_')[-1]), args['protein_path'])
    test_dataset = WSI_feature_Dataset(args['feature_root'], 'test', int(args['predict_target'].split('_')[-1]), args['protein_path'])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6*num_gpu, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=6*num_gpu, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=6*num_gpu, shuffle=False)

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

if args['predict_target'].split('_')[0] == 'gene':
    target_names = get_target(int(args['predict_target'].split('_')[-1]), 'gene')
elif args['predict_target'].split('_')[0] == 'protein':
    target_names = get_target(int(args['predict_target'].split('_')[-1]), 'protein')

# Define the model
load_model_path = os.path.join(exp_folder, 'best_corr.ckpt')
print(f"Load model: {load_model_path}")
model = Feature2Pathway_Attn(k=args['k'], size=args['model_size'], class_num=class_num, gated=args['gated'])
ckpt = torch.load(load_model_path,  map_location = 'cpu')
model = nn.DataParallel(model)
model.load_state_dict(ckpt)
model = model.cuda()
model.eval()

# Define the loss function and optimizer
# Use MSE to optimize the model, and use MAE to be the metric
criterion_MSE = nn.MSELoss()
criterion_MAE = nn.L1Loss()

def eval_omics(loader, model, target_names, csv_name):
    model.eval()
    MSE_record, MAE_record = [], []
    labels_record, preds_record = [], []
    for slide_name, features, label, target_names in tqdm(loader):

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

    overall_MSE = sum(MSE_record) / len(MSE_record)   # compute the validation MSE of this epoch
    overall_MAE = sum(MAE_record) / len(MAE_record)   # compute the validation MAE of this epoch

    # Compute the correlation of this epoch (see more detail in util.py file)
    corr_list, omics_corr_00_25, omics_corr_25_50, omics_corr_50_75, omics_corr_75_00 = stat_corr(labels_record=labels_record, preds_record=preds_record)
    overall_corr = np.nanmean(corr_list)              # compute overall training correlation of this epoch

    print(f"MSE = {overall_MSE:.3f}, MAE = {overall_MAE:.3f}")
    print(f"Spearman Correlation = {overall_corr:.3f}, 0.00 ~ 0.25: {omics_corr_00_25}, 0.25 ~ 0.50: {omics_corr_25_50}, 0.50 ~ 0.75: {omics_corr_50_75}, 0.75 ~ 1.00: {omics_corr_75_00}")

"""
def eval_omics(loader, model, target_names, csv_name):
    MSE_record, MAE_record = [], []
    labels_record, preds_record = [], []
    model.eval()
    with torch.no_grad():
        for slide_name, features, label, target_names in tqdm(loader):
            if args['model_arch'] == 'Feature2Pathway_Attn':
                preds, alpha = model(features.cuda())
            else:
                preds = model(features.cuda())
            loss_MSE = criterion_MSE(preds, label.cuda())
            loss_MAE = criterion_MAE(preds, label.cuda())
            MSE_record.append(loss_MSE.item())
            MAE_record.append(loss_MAE.item())
            labels_record.extend(label.detach().cpu().numpy().tolist())
            preds_record.extend(preds.detach().cpu().numpy().tolist())
        overall_MSE = sum(MSE_record) / len(MSE_record)
        overall_MAE = sum(MAE_record) / len(MAE_record)
        corr_list, omics_corr_00_25, omics_corr_25_50, omics_corr_50_75, omics_corr_75_00 = stat_corr(labels_record=labels_record, preds_record=preds_record)
        overall_corr = np.nanmean(corr_list)

    with open(os.path.join(exp_folder, csv_name), 'w', newline='') as f:
        assert len(corr_list) == len(target_names)
        writer = csv.writer(f)
        writer.writerow(['Omics', 'Spearman Corr'])
        for i in range(len(corr_list)):
            writer.writerow([target_names[i], corr_list[i]])

    print(f"MSE = {overall_MSE:.3f}, MAE = {overall_MAE:.3f}")
    print(f"Spearman Correlation = {overall_corr:.3f}, 0.00 ~ 0.25: {omics_corr_00_25}, 0.25 ~ 0.50: {omics_corr_25_50}, 0.50 ~ 0.75: {omics_corr_50_75}, 0.75 ~ 1.00: {omics_corr_75_00}")
"""

print("[ Training Dataset ]")
eval_omics(train_dataloader, model, target_names, 'training_data_results.csv')

print("[ Validation Dataset ]")
eval_omics(valid_dataloader, model, target_names, 'validation_data_results.csv')

print("[ Testing Dataset ]")
eval_omics(test_dataloader, model, target_names, 'testing_data_results.csv')

