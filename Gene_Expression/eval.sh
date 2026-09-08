#!/usr/bin/sh
#SBATCH -A MST109178
#SBATCH -J eval_1
#SBATCH -p ngs1gpu
#SBATCH -c 6
#SBATCH --mem=90g
#SBATCH --gres=gpu:1
#SBATCH -o eval_1_out.log
#SBATCH -e eval_1_err.log
#SBATCH --mail-user=chen16237@gmail.com
#SBATCH --mail-type=END

#source ~/anaconda3/etc/profile.d/conda.sh

module load pkg/Anaconda3
source activate /home/r10945019/anaconda3/envs/Pytorch

python eval.py \
-exp_name 'GeneExpression_Attn_1_' \
-feature_root '/staging/biology/r10945019/data/Lung/LUAD_DX_40x_Inception_ResNet_v2_flatten_feature_KMeans_location_depend_newBlurRemoval/400/' \
-gene_path '/staging/biology/r10945019/info/Lung/TCGA.LUAD.sampleMap_HiSeqV2_PANCAN' \
-model_arch 'Feature2Pathway_Attn' \
-predict_target 'gene_1' \
-model_size 'medium' \
-k 400 \
-batch_size_one_gpu 1 \
