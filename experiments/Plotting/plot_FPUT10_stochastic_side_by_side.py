# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:51:09 2020

@author: pccom
"""

import os, sys
import numpy as np

sys.path.append("../..")

from auxiliary_functions import load_pickled_object
import re

from plot_noise_regularization_results import load_files_and_compute_metrics

file_directory = os.path.join(os.getcwd(), "..", "Results", "FPUT10_v7")

default_names = [
    "BCG",
    "BCG_constraint",
    "BCG_integral",
    "BCG_integral_constraint",
    "CVXOPT",
    "CVXOPT_constraint",
    "CVXOPT_integral",
    "CVXOPT_integral_constraint",
    "FISTA",
    "FISTA_integral",
    "SINDy",
    "SINDy_integral",
    "SR3_constrained_l0",
    "SR3_constrained_l1",
    "SR3_l0",
    "SR3_l1",
    "SR3_constrained_l0_integral",
    "SR3_constrained_l1_integral",
    "SR3_l0_integral",
    "SR3_l1_integral",
]

data = load_files_and_compute_metrics(file_directory, default_names)

dimension = 10
max_noise_level = len(data["noise"])
size_marker = 10
font_size_title = 30
font_size_axes = 20
legend_size = 10
tick_size = 20
marker_interval = 2

x_axis = data["noise"][:max_noise_level].squeeze()


import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": tick_size})
fig, axs = plt.subplots(
    5, 2, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]}
)

list_data_left = [
    data["BCG_recovery"][:max_noise_level],
    data["SINDy_recovery"][:max_noise_level],
    data["FISTA_recovery"][:max_noise_level],
    data["CVXOPT_recovery"][:max_noise_level],
    data["BCG_constraint_recovery"][:max_noise_level],
    data["CVXOPT_constraint_recovery"][:max_noise_level],
    data["SR3_constrained_l0_recovery"][:max_noise_level],
    data["SR3_constrained_l1_recovery"][:max_noise_level],
]
list_data_right = [
    data["BCG_integral_recovery"][:max_noise_level],
    data["SINDy_integral_recovery"][:max_noise_level],
    data["FISTA_integral_recovery"][:max_noise_level],
    data["CVXOPT_integral_recovery"][:max_noise_level],
    data["BCG_integral_constraint_recovery"][:max_noise_level],
    data["CVXOPT_integral_constraint_recovery"][:max_noise_level],
    data["SR3_constrained_l0_integral_recovery"][:max_noise_level],
    data["SR3_constrained_l1_integral_recovery"][:max_noise_level],
]
list_labels = [
    "CINDy",
    "SINDy",
    "FISTA",
    "IPM",
    "CINDy (c)",
    "IPM (c)",
    r"SR3 (c-$\ell_0$)",
    r"SR3 (c-$\ell_1$)",
]
colors = ["k", "m", "c", "r", "b", "g", "tab:orange", "maroon"]
markers = ["o", "s", "P", "*", "^", "D", "X", "v"]
y_min = 10000.0
y_max = 0.0
for i in range(len(list_data_left)):
    mean = np.mean(list_data_left[i], axis=1)
    std_dev = np.std(list_data_left[i], axis=1)
    axs[0, 0].loglog(
        x_axis,
        np.divide(mean, x_axis),
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

for i in range(len(list_data_right)):
    mean = np.mean(list_data_right[i], axis=1)
    std_dev = np.std(list_data_right[i], axis=1)
    axs[0, 1].loglog(
        x_axis,
        np.divide(mean, x_axis),
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

ymin1, ymax1 = axs[0, 0].get_ylim()
ymin2, ymax2 = axs[0, 1].get_ylim()
y_min = min(ymin1, ymin2)
y_max = max(ymax1, ymax2)

axs[0, 0].set_ylim(
    [
        y_min,
        y_max,
    ]
)
axs[0, 1].set_ylim(
    [
        y_min,
        y_max,
    ]
)

axs[0, 0].set_ylabel(r"$\mathcal{E}_{R} / \eta$", fontsize=font_size_axes)
axs[0, 0].set_title("Differential", fontsize=font_size_title)
axs[0, 1].set_title("Integral", fontsize=font_size_title)
axs[0, 1].legend(fontsize=legend_size, ncol=2)

list_data_left = [
    data["BCG_derivative"][:max_noise_level],
    data["SINDy_derivative"][:max_noise_level],
    data["FISTA_derivative"][:max_noise_level],
    data["CVXOPT_derivative"][:max_noise_level],
    data["BCG_constraint_derivative"][:max_noise_level],
    data["CVXOPT_constraint_derivative"][:max_noise_level],
    data["SR3_constrained_l0_derivative"][:max_noise_level],
    data["SR3_constrained_l1_derivative"][:max_noise_level],
]
list_data_right = [
    data["BCG_integral_derivative"][:max_noise_level],
    data["SINDy_integral_derivative"][:max_noise_level],
    data["FISTA_integral_derivative"][:max_noise_level],
    data["CVXOPT_integral_derivative"][:max_noise_level],
    data["BCG_integral_constraint_derivative"][:max_noise_level],
    data["CVXOPT_integral_constraint_derivative"][:max_noise_level],
    data["SR3_constrained_l0_integral_derivative"][:max_noise_level],
    data["SR3_constrained_l1_integral_derivative"][:max_noise_level],
]
y_min = 10000.0
y_max = 0.0
for i in range(len(list_data_left)):
    mean = np.mean(list_data_left[i], axis=1)
    std_dev = np.std(list_data_left[i], axis=1)
    axs[1, 0].loglog(
        x_axis,
        np.divide(mean, x_axis),
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

for i in range(len(list_data_right)):
    mean = np.mean(list_data_right[i], axis=1)
    std_dev = np.std(list_data_right[i], axis=1)
    axs[1, 1].loglog(
        x_axis,
        np.divide(mean, x_axis),
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )


ymin1, ymax1 = axs[1, 0].get_ylim()
ymin2, ymax2 = axs[1, 1].get_ylim()
y_min = min(ymin1, ymin2)
y_max = max(ymax1, ymax2)

axs[1, 0].set_ylim(
    [
        y_min,
        y_max,
    ]
)
axs[1, 1].set_ylim(
    [
        y_min,
        y_max,
    ]
)

axs[1, 0].set_ylabel(r"$\mathcal{E}_{D} / \eta$", fontsize=font_size_axes)


list_data_left = [
    data["BCG_trajectory"][:max_noise_level],
    data["SINDy_trajectory"][:max_noise_level],
    data["FISTA_trajectory"][:max_noise_level],
    data["CVXOPT_trajectory"][:max_noise_level],
    data["BCG_constraint_trajectory"][:max_noise_level],
    data["CVXOPT_constraint_trajectory"][:max_noise_level],
    data["SR3_constrained_l0_trajectory"][:max_noise_level],
    data["SR3_constrained_l1_trajectory"][:max_noise_level],
]
list_data_right = [
    data["BCG_integral_trajectory"][:max_noise_level],
    data["SINDy_integral_trajectory"][:max_noise_level],
    data["FISTA_integral_trajectory"][:max_noise_level],
    data["CVXOPT_integral_trajectory"][:max_noise_level],
    data["BCG_integral_constraint_trajectory"][:max_noise_level],
    data["CVXOPT_integral_constraint_trajectory"][:max_noise_level],
    data["SR3_constrained_l0_integral_trajectory"][:max_noise_level],
    data["SR3_constrained_l1_integral_trajectory"][:max_noise_level],
]
y_min = 10000.0
y_max = 0.0
for i in range(len(list_data_left)):
    mean = np.mean(list_data_left[i], axis=1)
    std_dev = np.std(list_data_left[i], axis=1)
    axs[2, 0].loglog(
        x_axis,
        np.divide(mean, x_axis),
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )
    #     y_max = np.max(mean + std_dev)

for i in range(len(list_data_right)):
    mean = np.mean(list_data_right[i], axis=1)
    std_dev = np.std(list_data_right[i], axis=1)
    axs[2, 1].loglog(
        x_axis,
        np.divide(mean, x_axis),
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

ymin1, ymax1 = axs[2, 0].get_ylim()
ymin2, ymax2 = axs[2, 1].get_ylim()
y_min = min(ymin1, ymin2)
y_max = max(ymax1, ymax2)

axs[2, 0].set_ylim(
    [
        y_min,
        y_max,
    ]
)
axs[2, 1].set_ylim(
    [
        y_min,
        y_max,
    ]
)

axs[2, 0].set_ylabel(r"$\mathcal{E}_{T}/ \eta$", fontsize=font_size_axes)

list_data_left = [
    data["BCG_extra"][:max_noise_level],
    data["SINDy_extra"][:max_noise_level],
    data["FISTA_extra"][:max_noise_level],
    data["CVXOPT_extra"][:max_noise_level],
    data["BCG_constraint_extra"][:max_noise_level],
    data["CVXOPT_constraint_extra"][:max_noise_level],
    data["SR3_constrained_l0_extra"][:max_noise_level],
    data["SR3_constrained_l1_extra"][:max_noise_level],
]
list_data_right = [
    data["BCG_integral_extra"][:max_noise_level],
    data["SINDy_integral_extra"][:max_noise_level],
    data["FISTA_integral_extra"][:max_noise_level],
    data["CVXOPT_integral_extra"][:max_noise_level],
    data["BCG_integral_constraint_extra"][:max_noise_level],
    data["CVXOPT_integral_constraint_extra"][:max_noise_level],
    data["SR3_constrained_l0_integral_extra"][:max_noise_level],
    data["SR3_constrained_l1_integral_extra"][:max_noise_level],
]
y_min = 10000.0
y_max = 0.0
for i in range(len(list_data_left)):
    mean = np.mean(list_data_left[i], axis=1)
    std_dev = np.std(list_data_left[i], axis=1)
    axs[3, 0].semilogx(
        x_axis,
        mean,
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

for i in range(len(list_data_right)):
    mean = np.mean(list_data_right[i], axis=1)
    std_dev = np.std(list_data_right[i], axis=1)
    axs[3, 1].semilogx(
        x_axis,
        mean,
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

ymin1, ymax1 = axs[3, 0].get_ylim()
ymin2, ymax2 = axs[3, 1].get_ylim()
y_min = min(ymin1, ymin2)
y_max = max(ymax1, ymax2)

axs[3, 0].set_ylim(
    [
        y_min,
        y_max,
    ]
)
axs[3, 1].set_ylim(
    [
        y_min,
        y_max,
    ]
)

axs[3, 0].set_ylabel(r"$\mathcal{S}_E$", fontsize=font_size_axes)

list_data_left = [
    data["BCG_missing"][:max_noise_level],
    data["SINDy_missing"][:max_noise_level],
    data["FISTA_missing"][:max_noise_level],
    data["CVXOPT_missing"][:max_noise_level],
    data["BCG_constraint_missing"][:max_noise_level],
    data["CVXOPT_constraint_missing"][:max_noise_level],
    data["SR3_constrained_l0_missing"][:max_noise_level],
    data["SR3_constrained_l1_missing"][:max_noise_level],
]
list_data_right = [
    data["BCG_integral_missing"][:max_noise_level],
    data["SINDy_integral_missing"][:max_noise_level],
    data["FISTA_integral_missing"][:max_noise_level],
    data["CVXOPT_integral_missing"][:max_noise_level],
    data["BCG_integral_constraint_missing"][:max_noise_level],
    data["CVXOPT_integral_constraint_missing"][:max_noise_level],
    data["SR3_constrained_l0_integral_missing"][:max_noise_level],
    data["SR3_constrained_l1_integral_missing"][:max_noise_level],
]
y_min = 10000.0
y_max = 0.0
for i in range(len(list_data_left)):
    mean = np.mean(list_data_left[i], axis=1)
    std_dev = np.std(list_data_left[i], axis=1)
    axs[4, 0].semilogx(
        x_axis,
        mean,
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

for i in range(len(list_data_right)):
    mean = np.mean(list_data_right[i], axis=1)
    std_dev = np.std(list_data_right[i], axis=1)
    axs[4, 1].semilogx(
        x_axis,
        mean,
        colors[i],
        marker=markers[i],
        markersize=size_marker,
        markerfacecolor="None",
        markeredgecolor=colors[i],
        markeredgewidth=1,
        markevery=marker_interval,
        linewidth=2.0,
        label=list_labels[i],
    )

ymin1, ymax1 = axs[4, 0].get_ylim()
ymin2, ymax2 = axs[4, 1].get_ylim()
y_min = min(ymin1, ymin2)
y_max = max(ymax1, ymax2)

axs[4, 0].set_ylim(
    [
        y_min,
        y_max,
    ]
)
axs[4, 1].set_ylim(
    [
        y_min,
        y_max,
    ]
)

axs[4, 0].set_ylabel(r"$\mathcal{S}_M$", fontsize=font_size_axes)

for ax in axs.flat:
    ax.grid()

for ax in axs.flat:
    ax.set(xlabel=r"$\eta$ (Noise level)")

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.set_figheight(15)
fig.set_figwidth(12)
plt.tight_layout()

plt.savefig(
    os.path.join(
        os.getcwd(),
        "..",
        "Images",
        "FPUT10_v7",
        "FPUT_" + str(dimension) + "_dim_v7.pdf",
    ),
    bbox_inches="tight",
)
plt.close()

from plot_noise_regularization_results import plot_stochastic

# Plot the difference between exact dynamic and recovered dynamic
list_data = [
    data["BCG_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["BCG_constraint_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SINDy_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["FISTA_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_constraint_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l0_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l1_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["exact_derivative_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
]
list_labels = [
    "CINDy",
    "CINDy (c)",
    "SINDy",
    "FISTA",
    "IPM",
    "IPM (c)",
    r"SR3 (c-$\ell_0$)",
    r"SR3 (c-$\ell_1$)",
    "Exact dynamic",
]
colors = ["k", "b", "m", "c", "r", "g", "tab:orange", "maroon", "y"]
markers = ["o", "^", "s", "P", "*", "D", "X", "v", "H"]
plot_stochastic(
    data["noise"][:, 0][:max_noise_level],
    list_data,
    [],
    "Differential",
    r"$\eta$ (Noise level)",
    r"$	\Vert \dot{Y}_{\mathrm{training}} -  \Omega^T \Psi(Y_{\mathrm{training}}) \Vert_F / \eta$",
    colors,
    markers,
    legend_location="upper right",
    fill_between_lines=False,
    log_y=False,
    save_figure=os.path.join(
        os.getcwd(),
        "..",
        "Images",
        "FPUT10_v7",
        "FPUT_differential_training_" + str(dimension) + "_dim_v7.pdf",
    ),
)
plt.close()

# Plot the difference between exact dynamic and recovered dynamic
list_data = [
    data["BCG_integral_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["BCG_integral_constraint_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SINDy_integral_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["FISTA_integral_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_integral_constraint_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l0_integral_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l1_integral_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["exact_trajectory_training"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
]
plot_stochastic(
    data["noise"][:, 0][:max_noise_level],
    list_data,
    [],
    "Integral",
    r"$\eta$ (Noise level)",
    r"$	\Vert \delta Y_{\mathrm{training}} -  \Omega^T \Gamma(Y_{\mathrm{training}}) \Vert_F / \eta$",
    colors,
    markers,
    legend_location="upper right",
    fill_between_lines=False,
    log_y=False,
    save_figure=os.path.join(
        os.getcwd(),
        "..",
        "Images",
        "FPUT10_v7",
        "FPUT_integral_training_" + str(dimension) + "_dim_v7.pdf",
    ),
)
plt.close()

# Plot the difference between exact dynamic and recovered dynamic
list_data = [
    data["BCG_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["BCG_constraint_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SINDy_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["FISTA_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_constraint_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l0_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l1_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["exact_derivative_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
]
plot_stochastic(
    data["noise"][:, 0][:max_noise_level],
    list_data,
    [],
    "Differential",
    r"$\eta$ (Noise level)",
    r"$	\Vert \dot{Y}_{\mathrm{testing}} -  \Omega^T \Psi(Y_{\mathrm{testing}}) \Vert_F / \eta$",
    colors,
    markers,
    legend_location="upper right",
    fill_between_lines=False,
    log_y=False,
    save_figure=os.path.join(
        os.getcwd(),
        "..",
        "Images",
        "FPUT10_v7",
        "FPUT_differential_validation_" + str(dimension) + "_dim_v7.pdf",
    ),
)
plt.close()

# Plot the difference between exact dynamic and recovered dynamic
list_data = [
    data["BCG_integral_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["BCG_integral_constraint_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SINDy_integral_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["FISTA_integral_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_integral_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["CVXOPT_integral_constraint_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l0_integral_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["SR3_constrained_l1_integral_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
    data["exact_trajectory_validation"][:max_noise_level]
    / data["noise"][:, 0][:max_noise_level][:, None],
]
plot_stochastic(
    data["noise"][:, 0][:max_noise_level],
    list_data,
    list_labels,
    "Integral",
    r"$\eta$ (Noise level)",
    r"$	\Vert \delta Y_{\mathrm{testing}} -  \Omega^T \Gamma(Y_{\mathrm{testing}}) \Vert_F / \eta$",
    colors,
    markers,
    legend_location="upper right",
    fill_between_lines=False,
    log_y=False,
    save_figure=os.path.join(
        os.getcwd(),
        "..",
        "Images",
        "FPUT10_v7",
        "FPUT_integral_validation_" + str(dimension) + "_dim_v7.pdf",
    ),
)
plt.close()
