#!/usr/bin/sh
#SBATCH -A MST109178
#SBATCH -J fft_blur_detection
#SBATCH -p ngs48G           
#SBATCH -c 14               # 使用的數 請參考Queue資源設定 
#SBATCH --mem=46g           # 使用的記憶體量 請參考Queue資源設定
#SBATCH -o fft_out.log
#SBATCH -e fft_err.log
#SBATCH --mail-user=chen16237@gmail.com
#SBATCH --mail-type=END

#source ~/anaconda3/etc/profile.d/conda.sh

module load pkg/Anaconda3
source activate /home/r10945019/anaconda3/envs/Pytorch

python fft_blur_detection.py 