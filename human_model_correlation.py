import pandas as pd
import human_categories as hm
import tensorflow as tf
import numpy as np
import torch as torch
import scipy as sp
import matplotlib.pyplot as plt

sixteen_categories =  hm.get_human_object_recognition_categories()
sixteen_categories_ind = {}
for ind, cat in enumerate(sixteen_categories):
    sixteen_categories_ind[cat] = ind

# layer_names = ['input_1', 'block1_conv1', 'block1_conv2', 'block1_pool', 
# 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 
# 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 
# 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 
# 'flatten', 'fc1', 'fc2', 'predictions']
    
def get_tensors(session_no):
    human_data_df= pd.read_csv(
        "/Users/stephanie/Desktop/Serre Lab/model_human_comparison/model-vs-human/raw-data/colour/colour_subject-0" + str(session_no)+ "_session_1.csv")
    human_data = human_data_df["object_response"].values.tolist()
    human_hot_tensor = torch.empty(0, 16)
    for response in human_data:
        if response == "na": 
            hot_tensor = torch.full((1, 16), np.nan)
            human_hot_tensor = np.row_stack((human_hot_tensor,hot_tensor))
        for cat_ind, cat in enumerate(sixteen_categories): 
            if response == cat: 
                hot_tensor = tf.one_hot(cat_ind, 16)
                human_hot_tensor = np.row_stack((human_hot_tensor,hot_tensor))

    return human_hot_tensor

human_1 = get_tensors(1)
human_2 = get_tensors(2)
human_3 = get_tensors(3)
human_4 = get_tensors(4)

mean_human_decision = np.stack((human_1, human_2, human_3, human_4), -1).mean(-1)

# harmonized_dist_path =  "/Users/stephanie/Desktop/Serre Lab/controversial_result_1/sixteen_way_class2/harmonized_classifier_margins_2500.csv"
# baseline_dist_path = "/Users/stephanie/Desktop/Serre Lab/controversial_result_1/sixteen_way_class2/baseline_classifier_margins_2500.csv"

harmonized_dist_path_fc1 =  "/Users/stephanie/Desktop/Serre Lab/controversial_result_1/layers_sixteen_way_class/harmonized_classifier_margins_fc1.csv"
baseline_dist_path_fc1 = "/Users/stephanie/Desktop/Serre Lab/controversial_result_1/layers_sixteen_way_class/baseline_classifier_margins_fc1.csv"
harmonized_dist_path_fc2 =  "/Users/stephanie/Desktop/Serre Lab/controversial_result_1/layers_sixteen_way_class/harmonized_classifier_margins_fc2.csv"
baseline_dist_path_fc2 = "/Users/stephanie/Desktop/Serre Lab/controversial_result_1//layers_sixteen_way_class/baseline_classifier_margins_fc2.csv"


# this uses the cdist correlation distances
# harmonized_correlation = sp.spatial.distance.cdist(mean_human_decision.T, harmonized_distances.T, "correlation")
# harmonized_correlation = harmonized_correlation.diagonal()
# baseline_correlation = sp.spatial.distance.cdist(mean_human_decision.T, baseline_distances.T, "correlation")
# baseline_correlation = baseline_correlation.diagonal()

from scipy import stats


def get_correlation(res): 
    human_correlation = []
    model_ind = 16
    for correlation in res[:16]:
        print(correlation.shape)
        human_correlation.append(correlation[model_ind])
        model_ind +=1
    return human_correlation


def generate_correlation_plot(harmonized_logits_path, baseline_logits_path, mean_human_decision, save_dir):
    harmonized_distances_df = pd.read_csv(harmonized_logits_path)
    baseline_distances_df = pd.read_csv(baseline_logits_path)

    harmonized_distances = harmonized_distances_df.values[~np.isnan(mean_human_decision).any(axis=1)]
    baseline_distances = baseline_distances_df.values[~np.isnan(mean_human_decision).any(axis=1)]
    mean_human_decision = mean_human_decision[~np.isnan(mean_human_decision).any(axis=1)]
    print(harmonized_distances.shape)
    print(baseline_distances.shape)
    print(mean_human_decision.shape)

    res_harmonized = stats.spearmanr(mean_human_decision, harmonized_distances)
    res_baseline = stats.spearmanr(mean_human_decision, baseline_distances)

    harmonized_correlation = get_correlation(res_harmonized.correlation)
    baseline_correlation = get_correlation(res_baseline.correlation)

    plt.plot(np.arange(0, 16), harmonized_correlation, c="r", label='Harmonized distance distance')
    plt.plot(np.arange(0, 16), baseline_correlation, c="b", label="Baseline correlation distance")
    ax = plt.gca()
    ax.legend()
    ax.set_xticks(np.arange(0, 16), sixteen_categories)
    ax.set_xticklabels(sixteen_categories, rotation = 45)
    plt.tight_layout()
    plt.savefig("/Users/stephanie/Desktop/Serre Lab/controversial_result_1/layers_sixteen_way_class/spearmann_plt_"+save_dir+".png")
    plt.show()



generate_correlation_plot(harmonized_dist_path_fc1, baseline_dist_path_fc1, mean_human_decision, 'fc1')
generate_correlation_plot(harmonized_dist_path_fc2, baseline_dist_path_fc2, mean_human_decision, 'fc2')
