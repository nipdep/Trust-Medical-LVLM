from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Optional, Union
import numpy as np
from scipy.stats import f
from scipy.stats import ttest_ind
import pandas as pd
import scipy
import json

"""
Input Requirement
y_true: 1d array-like
y_pred: 1d array-like
"""

def pred_no_op(y_true, y_pred):
    return y_pred

def pred_sum(y_true, y_pred):
    return np.array(y_pred).sum()

def pred_mean(y_true, y_pred):
    return np.array(y_pred).mean()

def pearson_corr(y_true, y_pred, nan_to_num: Optional[Union[float, int]] = None):
    x = np.array(y_pred, dtype=np.float32)
    if nan_to_num is not None:
        x = np.nan_to_num(x, nan=float(nan_to_num))
    y = np.array(y_true, dtype=np.float32)
    non_nan_indices = np.where(~np.isnan(x))[0]
    if non_nan_indices.size >= 2:
        corr = scipy.stats.pearsonr(x[non_nan_indices], y[non_nan_indices])[0]
    else:
        corr = np.nan
    return corr


def failure(y_true, y_pred, fails_num: Optional[Union[float, int]] = np.nan):
    # Calculate the proportion of occurrences of fails_num in the y_pred sequence.
    x = np.array(y_pred, dtype=np.float32)
    if np.isnan(fails_num):
        failure = np.isnan(x).sum() / x.size
    else:
        failure = (x == fails_num).sum() / x.size
    return failure

def parse_box_string(box_str):
    # Remove triple quotes and any additional newline characters
    box_str = box_str.replace("'''", "").replace("\n", "").strip("[]")
    parts = box_str.split(",")
    parsed_parts = []
    for part in parts:
        # Clean up any stray spaces
        clean_part = part.strip()
        if '/' in clean_part:
            numerator, denominator = clean_part.split('/')
            parsed_parts.append(float(numerator) / float(denominator))
        else:
            parsed_parts.append(float(clean_part))
    return parsed_parts

def iou_judge(box1_list, box2_list):
    print('box1_list: {}'.format(box1_list))
    print('box2_list: {}'.format(box2_list))
    cnt = 0
    box_len = len(box1_list)
    for i in range(box_len):
        box_1 = json.loads(box1_list[i])
        box_2 = parse_box_string(box2_list[i])
        x1_min, y1_min, x1_max, y1_max = box_1
        x2_min, y2_min, x2_max, y2_max = box_2

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)
        inter_area = inter_width * inter_height 
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if inter_area == 0 or union_area == 0:
            iou = 0
        else:
            iou = inter_area / union_area
        print("iou",iou)
        if iou > 0.5:
            print("success grounding!")
            cnt += 1
    
    grounding_rate = (cnt * 1.0 / box_len) * 100.0       
    return grounding_rate

def statistical_parity(y_pred, group):
    """Calculate Statistical Parity / Demographic Parity."""
    # Get indices for both groups
    g_m = (group == 1)  
    
    # Proportion of positive predictions for each group
    P_m = np.mean(y_pred[g_m])
    
    return P_m

def equal_opportunity(y_true, y_pred, group):
    """Calculate Equal Opportunity (TPR equality across groups)."""
    g_f = (group == 0)
    g_m = (group == 1)
    
    # Calculate TPR for both groups (True Positive Rate)
    # TPR = TP / (TP + FN) or TP / Total positives in the group
    positives_f = np.sum(y_true[g_f] == 1)
    positives_m = np.sum(y_true[g_m] == 1)
    
    TPR_f = (np.sum((y_pred[g_f] == 1) & (y_true[g_f] == 1)) / positives_f
             if positives_f > 0 else np.nan)
    TPR_m = (np.sum((y_pred[g_m] == 1) & (y_true[g_m] == 1)) / positives_m
             if positives_m > 0 else np.nan)
    
    return TPR_f, TPR_m

def equalized_odds(y_true, y_pred, group):
    """Calculate Equalized Odds (TPR and FPR equality across groups)."""
    g_f = (group == 0)
    g_m = (group == 1)
    
    # True Positive Rate (TPR)
    positives_f = np.sum(y_true[g_f] == 1)
    positives_m = np.sum(y_true[g_m] == 1)
    
    TPR_f = (np.sum((y_pred[g_f] == 1) & (y_true[g_f] == 1)) / positives_f
             if positives_f > 0 else np.nan)
    TPR_m = (np.sum((y_pred[g_m] == 1) & (y_true[g_m] == 1)) / positives_m
             if positives_m > 0 else np.nan)
    
    # False Positive Rate (FPR)
    negatives_f = np.sum(y_true[g_f] == 0)
    negatives_m = np.sum(y_true[g_m] == 0)
    
    FPR_f = (np.sum((y_pred[g_f] == 1) & (y_true[g_f] == 0)) / negatives_f
             if negatives_f > 0 else np.nan)
    FPR_m = (np.sum((y_pred[g_m] == 1) & (y_true[g_m] == 0)) / negatives_m
             if negatives_m > 0 else np.nan)
    
    return (TPR_f, FPR_f), (TPR_m, FPR_m)

def overall_accuracy_equality(y_true, y_pred, group):
    """Calculate Overall Accuracy Equality across groups."""
    g_f = (group == 0)
    g_m = (group == 1)
    
    # Accuracy for both groups
    accuracy_f = np.mean(y_pred[g_f] == y_true[g_f])
    accuracy_m = np.mean(y_pred[g_m] == y_true[g_m])
    
    return accuracy_f, accuracy_m

def treatment_equality(y_true, y_pred, group):
    """Calculate Treatment Equality (ratio of errors across groups)."""
    g_f = (group == 0)
    g_m = (group == 1)
    
    # Confusion matrix components: TP, FP, TN, FN for both groups
    def get_confusion_components(y_true, y_pred):
        # Compute confusion matrix and handle cases where not all classes are present
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle cases with only one class in y_true
            tn = fp = fn = tp = 0
            if y_true[0] == 0:
                tn = cm[0, 0]
                fp = cm[0, 1] if cm.shape[1] > 1 else 0
            else:
                fn = cm[1, 0] if cm.shape[0] > 1 else 0
                tp = cm[1, 1] if cm.shape[1] > 1 else 0
        return tn, fp, fn, tp
    
    tn_f, fp_f, fn_f, tp_f = get_confusion_components(y_true[g_f], y_pred[g_f])
    tn_m, fp_m, fn_m, tp_m = get_confusion_components(y_true[g_m], y_pred[g_m])
    
    # Error ratios
    error_ratio_f = (fp_f + fn_f) / len(y_true[g_f]) if len(y_true[g_f]) > 0 else 0
    error_ratio_m = (fp_m + fn_m) / len(y_true[g_m]) if len(y_true[g_m]) > 0 else 0
    
    return error_ratio_f, error_ratio_m

def manual_welch_anova(binary_array, group_names):
    """
    Conduct Welch's ANOVA test manually across all groups and return F-statistic and p-value.

    Parameters:
    - binary_array (list or np.array): Binary array (0s and 1s).
    - group_names (list): List of group names corresponding to each element in binary_array.

    Returns:
    - dict: Dictionary with F-statistic and p-value.
    """
    # Organize data by group
    data = pd.DataFrame({'values': binary_array, 'group': group_names})
    groups = data.groupby('group')['values']
    
    # Calculate per-group statistics
    means = groups.mean()
    counts = groups.size()
    variances = groups.var(ddof=1)

    # Calculate Welch's ANOVA components
    weighted_mean = (means * counts / variances).sum() / (counts / variances).sum()
    df_num = len(groups) - 1
    df_denom = ((1 + (variances / counts) / ((counts / variances).sum() ** 2)) ** 2).sum() / (len(groups) - 1)
    F_stat = (weighted_mean - means.mean()) ** 2 / (variances / counts).sum()
    
    # Calculate p-value from F-distribution
    p_value = 1 - f.cdf(F_stat, df_num, df_denom)
    
    return {'F': F_stat, 'p-value': p_value}

def generic_t_test(binary_results, group_mask):
    """
    Conducts a two-tailed t-test for two binary groups within binary results.
    
    Parameters:
    - binary_results (list or np.array): Binary array (0s and 1s).
    - group_mask (list or np.array): Binary group mask (0s and 1s) indicating group membership.
    
    Returns:
    - t_score (float): T-statistic.
    - p_value (float): P-value of the t-test.
    """
    # Separate the two groups based on the mask
    group1_values = [val for val, mask in zip(binary_results, group_mask) if mask == 1]
    group2_values = [val for val, mask in zip(binary_results, group_mask) if mask == 0]
    
    # Perform two-tailed t-test (Welch's t-test with equal_var=False)
    t_score, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
    
    return t_score, p_value

_supported_metrics = {
    # aggregation op
    "pred_no_op": pred_no_op,
    "pred_sum": pred_sum,
    "pred_mean": pred_mean,

    # general metrics
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score, 
    "f1_score": f1_score,
    "pearson_corr": pearson_corr,
    "failure": failure,
    "iou_judge": iou_judge,

    # fairness metrics
    "statistical_parity": statistical_parity,
    "equal_opportunity": equal_opportunity,
    "equalized_odds": equalized_odds,
    "overall_accuracy_equality": overall_accuracy_equality,
    "treatment_equality": treatment_equality,
}
