import os
import cv2
import shutil
import numpy as np

#image_root = '/staging/biology/r10945019/data/Lung/TCGA_LUAD_PARADIGM_40x_DX/'
#feature_root = '/staging/biology/r10945019/data/Lung/LUAD_DX_40x_Inception_ResNet_v2_flatten_feature/'

image_root = '/staging/biology/r10945019/data/Lung/TCGA_LUAD_40x/DX(no_matched_pathway)/'
feature_root = '/staging/biology/r10945019/data/Lung/new_LUAD_Inception_Resnet_v2_feature/'

image_mv_root = '/staging/biology/r10945019/data/Lung/TCGA_LUAD_PARADIGM_40x_DX_FFT_BlurImage/'
feature_mv_root = '/staging/biology/r10945019/data/Lung/TCGA_LUAD_PARADIGM_40x_DX_FFT_BlurFeature/'

def variance_of_laplacian(image, ksize=13):
    var = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize).var()
    return var

def fft_blur_detection(image, size, thresh):
    h, w = image.shape
    cX, cY = int(w/2.), int(h/2.)
    
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    
    return mean, (mean<=thresh)

'''
for mode in ['train', 'valid', 'test']:
    for slide in os.listdir(os.path.join(image_root, mode)):
        for image in os.listdir(os.path.join(image_root, mode, slide)):
            img = cv2.imread(os.path.join(image_root, mode, slide, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean, blur = fft_blur_detection(img, 70, 14)
            fm = variance_of_laplacian(img, 13)
            if blur and (fm < 4e14 or fm > 6e14):
                shutil.copy(os.path.join(image_root, mode, slide, image), os.path.join(image_mv_root, mode, slide, image))
                feature = image+'.npy'
                shutil.copy(os.path.join(feature_root, mode, slide, feature), os.path.join(feature_mv_root, mode, slide, feature))
'''
for slide in os.listdir(image_root):
    for image in os.listdir(os.path.join(image_root, slide)):
        img = cv2.imread(os.path.join(image_root, slide, image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean, blur = fft_blur_detection(img, 70, 14)
        fm = variance_of_laplacian(img, 13)
        if blur and (fm < 4e14 or fm > 6e14):
            shutil.copy(os.path.join(image_root, slide, image), os.path.join(image_mv_root, image))
            feature = image+'.npy'
            shutil.copy(os.path.join(feature_root, slide, feature), os.path.join(feature_mv_root, feature))