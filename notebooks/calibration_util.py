import os
import tensorflow as tf
from util import constants
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.trainer import Trainer
from absl import app
from absl import flags
import numpy as np
from util.models import MODELS
from util.tasks import TASKS
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
sns.set_style("whitegrid")

from tqdm import tqdm

def test_for_calibration(model, task, n_bins=10):
    preds = []
    correct_class_probs = []
    predicted_class_probs = []
    pred_logits = []
    y_trues = []
    batch_count = task.n_valid_batches
    for x, y in task.valid_dataset:
        logits = model(x)
        pred_logits.extend(logits.numpy())
        pred = tf.argmax(logits, axis=-1)
        prob = task.get_probs_fn()(logits, labels=y, temperature=1)
        preds.extend(pred.numpy())
        y_trues.extend(y.numpy())
        batch_indexes = tf.cast(tf.range(len(y), dtype=tf.int32), dtype=tf.int32)
        true_indexes = tf.concat([batch_indexes[:,None], y[:,None]], axis=1)
        pred_indexes = tf.concat([batch_indexes[:,None], tf.cast(pred[:,None], tf.int32)], axis=1)

        correct_class_probs.extend(tf.gather_nd(prob, true_indexes).numpy())
        predicted_class_probs.extend(tf.gather_nd(prob, pred_indexes).numpy())

        batch_count -= 1
        if batch_count == 0:
            break

    model_accuracy = np.asarray(preds) == np.asarray(y_trues)

    return model_accuracy, predicted_class_probs, correct_class_probs, pred_logits, y_trues

def plot_calibration(model_accuracy, predicted_class_probs, correct_class_probs, n_bins=10):
    p_confidence_bins = np.zeros(n_bins)
    n_confidence_bins = np.zeros(n_bins)
    total_confidence_bins = np.zeros(n_bins)
    
    denominator = 100.0 / n_bins
    for i in np.arange(len(model_accuracy)):
        if model_accuracy[i]:
            p_confidence_bins[min(int(predicted_class_probs[i]*100 // denominator),n_bins-1)] += 1.0
        else:
            n_confidence_bins[min(int(predicted_class_probs[i]*100 // denominator),n_bins-1)] -= 1.0
            
        total_confidence_bins[min(int(predicted_class_probs[i]*100 // denominator),n_bins-1)] += 1

    #sns.stripplot(model_accuracy,predicted_class_probs, color='blue', alpha=0.5, jitter=True)
    #sns.stripplot(model_accuracy,correct_class_probs, color='green', alpha=0.2, jitter=True)
    #sns.swarmplot(model_accuracy,predicted_class_probs, color='blue', alpha=0.5)
    #plt.show()
   
    sns.barplot(x=np.arange(0,n_bins)*denominator, 
                y=np.arange(0,n_bins)/n_bins, 
                color='green', alpha=0.2, edgecolor='black')
    ax = sns.barplot(x=np.arange(0,n_bins)*denominator, 
                    y=p_confidence_bins/total_confidence_bins, 
                    color='red', alpha=0.5, edgecolor='black')
    
    x_ticks = np.arange(0,n_bins,2)
    x_tick_labels = x_ticks / np.float32(n_bins)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=10)
    
def expected_calibration_error(teacher_accuracy, teacher_predicted_class_probs):
    raise NotImplemented