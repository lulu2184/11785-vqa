import matplotlib.pyplot as plt
import numpy as np


def draw_graph(y, title, name):
    n = len(y)
    x = np.arange(1, n + 1)
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.grid(True)
    plt.title(title)
    plt.savefig(name + '.png')
    plt.gcf().clear()


def draw_scores(y2, title, name):
    n = len(y2)
    x = np.arange(1, n + 1)
    plt.plot(x, y2, label='score on dev set')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.grid(True)
    plt.title(title)
    plt.savefig(name + '.png')
    plt.gcf().clear()


# baseline
baseline_loss = [8.75, 3.50, 3.20, 3.01, 2.87, 2.75, 2.65, 2.56, 2.47, 2.39,
                 2.31, 2.24, 2.18, 2.11, 2.05, 2.00, 1.95, 1.91, 1.87, 1.83,
                 1.79, 1.75, 1.72, 1.69, 1.67, 1.64, 1.62, 1.59, 1.57, 1.55]
# draw_graph(baseline_loss[0:15], 'Model Loss', 'Loss')
baseline_dev_score = [49.93, 55.31, 57.69, 59.20, 60.41, 60.81, 61.42, 61.71,
                      61.75, 61.96, 61.79, 61.88, 62.02, 61.95, 61.91, 61.91,
                      61.71, 61.78, 61.74, 61.69, 61.64, 61.56, 61.50, 61.53,
                      61.44, 61.40, 61.42, 61.26, 61.33, 61.33]
baseline_train_score = [40.48, 52.53, 57.04, 60.09, 62.31, 64.35, 66.21, 67.81,
                        69.41, 70.91, 72.35, 73.59, 74.81, 75.90, 76.94, 77.79,
                        78.62, 79.30, 79.91, 80.45, 81.07, 81.56, 81.96, 82.40,
                        82.75, 83.09, 83.43, 83.79, 83.99, 84.29]
# draw_scores(baseline_dev_score[0:15], 'Score on Validation Set', 'Score')

# new attention
new_att_loss = [9.58, 3.53, 3.22, 3.03, 2.89, 2.78, 2.67, 2.58, 2.43, 2.35,
                2.29, 2.23, 2.17, 2.07, 2.02]
new_att_score = [50.34, 55.14, 57.43, 59.38, 60.13, 60.84, 61.44, 61.76, 61.93,
                 62.17, 62.18, 62.30, 62.25, 62.24, 62.19]

# draw_graph(new_att_loss, 'Model Loss', 'Loss')
# draw_scores(new_att_score, 'Score on Validation Set', 'Score')

# multi-head attention
multi_head_att_loss = [7.74, 3.62, 3.93, 3.24, 3.12, 3.01, 2.92, 2.83, 2.75,
                       2.68, 2.61, 2.54, 2.47, 2.41, 2.35]
multi_head_att_scores = [47.31, 50.99, 53.13, 54.57, 55.44, 56.61, 57.05, 57.26,
                         57.61, 57.60, 57.69, 57.65, 57.66, 57.49, 56.63]

draw_graph(multi_head_att_loss, 'Model Loss', 'Loss')
draw_scores(multi_head_att_scores, 'Score on Validation Set', 'Score')
