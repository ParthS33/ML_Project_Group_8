################################################################
# visualization.py
#
# implemented visualization methods:
#
# - plot_confusion_matrix
# - plot_roc_curve
# - plot_f1_scores
################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, f1_score

################################################################
# Dataset imports
################################################################

test_NLI_M = pd.read_csv(r"data\sentihood\bert-pair\test_NLI_M.tsv", delimiter='\t')

def read_txt(dataset_name, mapping={0: 'None', 1: 'Positive', 2: 'Negative'}):
    """
    Auxiliary method that transforms result .txt datasets into adequate pandas dataframe.

    Params
    ------
    dataset_name: .txt file
        File containing the results (ouput of the BERT-pair NLI-M classifier).
    mapping: dict
        Dictionary mapping the classes in the output file to the class names in the ground truth file for consistency.
     """
    file_path = r"results\sentihood\NLI_M_256_24_2e-05_4.0_1_2\{}.txt".format(dataset_name)
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.columns = ["col1", "col2", "col3", "col4"]
    
    df.iloc[:, 0] = df.iloc[:, 0].map(mapping)
    
    return df

test_ep_1 = read_txt('test_ep_1')
test_ep_2 = read_txt('test_ep_2')
test_ep_3 = read_txt('test_ep_3')
test_ep_4 = read_txt('test_ep_4')

################################################################
# Confusion matrix
################################################################

def plot_confusion_matrix(actual_dataset, prediction_dataset, epoch_number, classes=['Negative', 'None', 'Positive']):

    """
    Method that create a confusion matrix for one specified epoch.
    
    Params
    ------
    actual_dataset: pandas dataframe
        Dataset containing the gound truth values.
    prediction_dataset: pandas dataframes
        Dataset containing the predicted values.
    epoch_number: int
        Number of the considered epoch.
    classes:
        Classes for the polarity of the sentiment.
    """
    
    y_true = actual_dataset['label']
    y_pred = prediction_dataset["col1"]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for epoch {}'.format(epoch_number))
    plt.show()

################################################################
# ROC curve
################################################################

def map_classes(value, class_name):

    """
    Auxialiary method to create binary (0 or 1) ground truth and prediction vectors.
    
    Params
    ------
    value: str
        Class of a vector. It can be 'Positive', 'Negative' or 'None'.
    class_name: str
        Class of interested in the 'one-vs-all' approach. It can be 'Positive', 'Negative' or 'None'.
    """
    if value == class_name:
        return 1
    else:
        return 0

def plot_roc_curve(actual_dataset, prediction_datasets, class_name):
    """
    Method that creates a ROC curve for a specified class versus all the other ones.
    
    Params
    ------
    actual_dataset: pandas dataframe
        Dataset containing the gound truth values.
    prediction_datasets: list of pandas dataframes
        Each dataframe corresponds to one epoch, with the predictions for each observation. 
    """
    fpr, tpr = [0.], [0.]
    
    for dataset in prediction_datasets:
        y_true = actual_dataset['label']
        y_pred = dataset["col1"]
        y_true_altered = y_true.apply(lambda x: map_classes(x, class_name))
        y_pred_altered = y_pred.apply(lambda x: map_classes(x, class_name))
        epoch_fpr, epoch_tpr, _ = roc_curve(y_true_altered, y_pred_altered)
        fpr = np.append(fpr, epoch_fpr[1])
        tpr = np.append(tpr, epoch_tpr[1])
    fpr = np.append(fpr, 1)
    tpr = np.append(tpr, 1)
    
    combined_values = zip(fpr, tpr)
    
    sorted_values = sorted(combined_values, key=lambda x: x[0])
    sorted_fpr, sorted_tpr = zip(*sorted_values)

    # the sorted indices represent the epoch, and label points (fpr, tpr) on the plot
    sorted_indices = sorted(range(len(sorted_fpr)), key=lambda i: fpr[i])
    sorted_indices[0], sorted_indices[5] = ' ', ' '

    plt.figure()
    plt.plot(sorted_fpr, sorted_tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for i, label in enumerate(sorted_indices):
        plt.text(sorted_fpr[i], sorted_tpr[i], label, ha='center', va='bottom')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: {} vs rest'.format(class_name))
    plt.legend(loc="lower right")
    plt.show()

################################################################
# F1 scores barplot
################################################################

def plot_f1_scores(actual_dataset, prediction_datasets, epoch_names):

    """
    Method that plot F1 scores for each epoch.
    
    Params
    ------
    actual_dataset: pandas dataframe
        Dataset containing the gound truth values.
    prediction_datasets: list of pandas dataframes
        Each dataframe corresponds to one epoch, with the predictions for each observation.
    epoch_names: list of str
        List of numbered epochs to label the plot.
    """
    f1_scores=[]
    for dataset in prediction_datasets:
        y_true = actual_dataset['label']
        y_pred = dataset["col1"]
        f1_scores.append(f1_score(y_true, y_pred, average='weighted'))

    plt.figure(figsize=(10, 6))
    plt.bar(epoch_names, f1_scores, color='skyblue')

    for i, f1 in enumerate(f1_scores):
        plt.plot(i, f1, 'bo')
        plt.text(i, f1, "{:.3f}".format(f1), ha='right', va='bottom')

    plt.plot(range(len(f1_scores)), f1_scores, 'b-')

    plt.xlabel('epoch')
    plt.ylabel('weighted F1 scores')
    plt.title('Weighted F1 Scores for each Epoch')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
