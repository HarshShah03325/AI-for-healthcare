import numpy as np
import pandas as pd
import tensorflow as tf

def compute_class_sens_spec(pred, label, class_num):
    """
    Compute sensitivity and specificity for a particular example
    for a given class.

    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (num classes, height, width, depth).
        label (np.array): binary array of labels, shape is
                          (num classes, height, width, depth).
        class_num (int): number between 0 - (num_classes -1) which says
                         which prediction class to compute statistics
                         for.

    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """

    class_pred = pred[class_num]
    class_label = label[class_num]

    
    tp = np.sum( (class_pred == 1) * (class_label == 1))
    tn = np.sum( (class_pred == 0) * (class_label == 0))
    fp = np.sum( (class_pred == 1) * (class_label == 0))
    fn = np.sum( (class_pred == 0) * (class_label == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def get_sens_spec_df(pred, label):
    patch_metrics = pd.DataFrame(
        columns = ['Edema', 
                   'Non-Enhancing Tumor', 
                   'Enhancing Tumor'], 
        index = ['Sensitivity',
                 'Specificity'])
    
    for i, class_name in enumerate(patch_metrics.columns):
        sens, spec = compute_class_sens_spec(pred, label, i)
        patch_metrics.loc['Sensitivity', class_name] = round(sens,4)
        patch_metrics.loc['Specificity', class_name] = round(spec,4)

    return patch_metrics


def compute_metrics(label, pred):
    whole_scan_label = tf.keras.utils.to_categorical(label, num_classes = 4)
    whole_scan_pred = pred
    whole_scan_label = np.moveaxis(whole_scan_label, 3 ,0)[1:4]
    whole_scan_pred = np.moveaxis(whole_scan_pred, 3, 0)[1:4]
    whole_scan_df = get_sens_spec_df(whole_scan_pred, whole_scan_label)
    return whole_scan_df