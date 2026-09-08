from sklearn.cluster import KMeans
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('-raw_feature_path', '--raw_feature_path', type = str, required = True, help = 'root of raw feature')
ap.add_argument('-cluster_feature_path', '--cluster_feature_path', type = str, required = True, help = 'root of clustered feature')
ap.add_argument('-patch_info_path', '--patch_info_path', type = str, required = False, help = 'raw patch containing location info')
ap.add_argument('-kmeans_depend', '--kmeans_depend', type = str, required = True, help = 'feature or location')
ap.add_argument('-k', '--k',  type = int, required = True, help = 'k')
args = vars(ap.parse_args())

print('K =', args['k'])

my_seed = 880310

def feature_k_means(k, slide_folder):
    '''
    KMeans depending on feature
    '''
    patch_record = {}
    for patch in os.listdir(slide_folder):
        fname = os.path.basename(os.path.join(slide_folder, patch))
        patch_name = os.path.splitext(fname)[0][:-8]
        patch_feature = np.load(os.path.join(slide_folder, patch))
        patch_feature = patch_feature.flatten().tolist()
        patch_record[patch_name] = patch_feature

    X = []
    for value in patch_record.values():
        X.append(value)
    #X = np.array(X)
    kmeans = KMeans(n_clusters = k, max_iter = 1000, random_state = my_seed).fit(X)

    final_kmeans_results = {}
    for patch in patch_record.keys():
        feature = patch_record[patch]
        feature = np.array(feature)[np.newaxis]
        cluster = kmeans.predict(feature)
        final_kmeans_results[patch] = int(cluster)

    return final_kmeans_results


def location_k_means(k, slide_folder, patch_info_path):
    '''
    KMean depending on location

    Input:
    k: number of clusters
    slide_folder: folder containing patch-level features
    patch_info_path: slide-level tile_selection.tsv file path

    patch_record: a dictionary. Keys are the patch names, and values are their locations.
                  (ex.{'TCGA-86-8668-01Z-00-DX1.d720d486-02c7-4f98-8feb-e0e50a12c158_12959' : [77, 100]})

    Output:
    final_kmeans_results: A dictionary. Keys are patch names, and values are their clusters.
                         (ex. {'TCGA-86-8668-01Z-00-DX1.d720d486-02c7-4f98-8feb-e0e50a12c158_12959': 8})
    '''

    patch_info = pd.read_table(patch_info_path, index_col = 'Tile')

    patch_record = {}
    for patch in os.listdir(slide_folder):
        fname = os.path.basename(os.path.join(slide_folder, patch))
        patch_name = os.path.splitext(fname)[0][:-8]
        patch_location = patch_info.loc[patch_name][3:5].tolist()
        patch_record[patch_name] = patch_location
    
    #X = np.array(patch_record.values())
    X = []
    for i in patch_record.values():
        X.append(i)
    kmeans = KMeans(n_clusters = k, max_iter = 1000, random_state = my_seed).fit(X)

    final_kmeans_results = {}
    for patch in patch_record.keys():
        location = patch_record[patch]
        location = np.array(location)[np.newaxis]
        cluster = kmeans.predict(location)
        final_kmeans_results[patch] = int(cluster)

    return final_kmeans_results

def mean_aggregate(slide_folder, feature_names, final_kmeans_results, k):
    '''
    Input:
    slide_folder: folder containing patch-level features
    feture_names: a list containg every patch-level feature names
    final_kmeans_results: output of Function k_means
    k: number of clusters

    Output:
    mean_agg_results: A dictionary. Keys are cluster index, and values are mean aggregation results
                     (ex. {8: [0.1, 0.2, 0.3, ..., -1.5]})   
                     btw, len(mean_agg_results) == k
    '''
    mean_agg_results = {}
    for i in range(k):
        k_list = []
        for patch_feature in feature_names:

            fname = os.path.basename(os.path.join(slide_folder, patch_feature)) # fname containing file extension .npy
            patch_name = os.path.splitext(fname)[0][:-8]   # patch_name: without file extention

            if final_kmeans_results[patch_name] == int(i):
                feature = np.load(os.path.join(slide_folder, patch_feature))
                if feature.ndim != 1:
                    feature = feature.flatten()
                feature = feature.tolist()
                k_list.append(feature)
        k_array = np.array(k_list)
        k_mean_agg = k_array.mean(axis = 0)
        mean_agg_results[int(i)] = k_mean_agg
    
    return mean_agg_results

k = str(args['k'])

for _slide in tqdm(os.listdir(args['raw_feature_path'])):
    slide_path = os.path.join(args['raw_feature_path'], _slide)
    feature_names = os.listdir(slide_path)
    final_kmeans_results = location_k_means(
        k = args['k'],
        slide_folder=slide_path,
        patch_info_path=os.path.join(args['patch_info_path'], _slide, 'tile_selection.tsv')
    )
    mean_agg_results = mean_aggregate(
        slide_folder=slide_path,
        feature_names=feature_names,
        final_kmeans_results=final_kmeans_results,
        k=args['k']
    )

    if os.path.isdir(os.path.join(args['cluster_feature_path'], k, _slide)) == False:
        os.mkdir(os.path.join(args['cluster_feature_path'], k, _slide))
    for i, cluster in enumerate(mean_agg_results.values()):
        cluster_array = np.array(cluster)
        np.save(os.path.join(args['cluster_feature_path'], k, _slide, _slide+str(i)), cluster_array)

