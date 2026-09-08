#!/usr/bin/sh
#SBATCH -A MST109178
#SBATCH -J KMeans_cluster_k50
#SBATCH -p ngs48G           
#SBATCH -c 14               # 使用的數 請參考Queue資源設定 
#SBATCH --mem=46g           # 使用的記憶體量 請參考Queue資源設定
#SBATCH -o KMeans_cluster_400_out.log
#SBATCH -e KMeans_cluster_400_err.log
#SBATCH --mail-user=chen16237@gmail.com
#SBATCH --mail-type=END

#source ~/anaconda3/etc/profile.d/conda.sh

module load pkg/Anaconda3
source activate /home/r10945019/anaconda3/envs/Pytorch

python kmeans_cluster_gen.py \
-raw_feature_path '/staging/biology/r10945019/data/Lung/LUAD_DX_40x_Inception_ResNet_v2_flatten_feature' \
-cluster_feature_path '/staging/biology/r10945019/data/Lung/LUAD_DX_40x_Inception_ResNet_v2_flatten_feature_KMeans_location_depend_newBlurRemoval/' \
-patch_info_path '/staging/biology/r10945019/data/Lung/TCGA_Lung_raw_patch/' \
-kmeans_depend 'location' \
-k 400
