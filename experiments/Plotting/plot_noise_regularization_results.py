# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:51:09 2020

@author: pccom
"""

import os, sys

sys.path.append("..")

from auxiliary_functions import load_pickled_object

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

fontsize = 19

fontsize_legend = 13

# data = load_pickled_object(os.path.join(os.getcwd(), 'results_kuramoto_L1reg.pickle'))

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
]


def load_files_and_compute_metrics(file_directory, algorithm_names=default_names):
    # Create the structure of the dataframe with the first file
    file_data = load_pickled_object(
        os.path.join(file_directory, os.listdir(file_directory)[0])
    )
    data = {
        "noise": file_data["noise"],
    }
    num_noise_levels = len(file_data["noise"])
    exact_dynamic = file_data["exact_dynamic"]
    # Compute the recovery metric for CINDy.
    algorithm_names
    for file in os.listdir(file_directory):
        file_data = load_pickled_object(os.path.join(file_directory, file))
        exact_dynamic = file_data["exact_dynamic"]
        for name in algorithm_names:
            data = compute_individual_metrics(name, exact_dynamic, data, file_data)

    # Training metrics on exact data
    metrics = [
        np.linalg.norm(Y_matrix - exact_dynamic.dot(matrix))
        for Y_matrix, matrix in zip(
            file_data["Y_train_data"], file_data["psi_train_data"]
        )
    ]
    if "exact_derivative_training" in data:
        data["exact_derivative_training"] = np.hstack(
            (data["exact_derivative_training"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data["exact_derivative_training"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.linalg.norm(Y_matrix - exact_dynamic.dot(matrix))
        for Y_matrix, matrix in zip(
            file_data["delta_train_data"], file_data["matrix_train_data"]
        )
    ]
    if "exact_trajectory_training" in data:
        data["exact_trajectory_training"] = np.hstack(
            (data["exact_trajectory_training"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data["exact_trajectory_training"] = np.asarray(metrics)[:, np.newaxis]

    # Training metrics on exact data
    metrics = [
        np.linalg.norm(Y_matrix - exact_dynamic.dot(matrix))
        for Y_matrix, matrix in zip(
            file_data["Y_validation_data"], file_data["psi_validation_data"]
        )
    ]
    if "exact_derivative_validation" in data:
        data["exact_derivative_validation"] = np.hstack(
            (data["exact_derivative_validation"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data["exact_derivative_validation"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.linalg.norm(Y_matrix - exact_dynamic.dot(matrix))
        for Y_matrix, matrix in zip(
            file_data["delta_validation_data"], file_data["matrix_validation_data"]
        )
    ]
    if "exact_trajectory_validation" in data:
        data["exact_trajectory_validation"] = np.hstack(
            (data["exact_trajectory_validation"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data["exact_trajectory_validation"] = np.asarray(metrics)[:, np.newaxis]

    return data


def compute_individual_metrics(name, exact_dynamic, data, problem_data):
    metrics = [
        np.linalg.norm(dynamic - exact_dynamic)
        for dynamic in problem_data[name + "_dynamic"]
    ]
    if name + "_recovery" in data:
        data[name + "_recovery"] = np.hstack(
            (data[name + "_recovery"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_recovery"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.linalg.norm((dynamic - exact_dynamic).dot(matrix))
        for dynamic, matrix in zip(
            problem_data[name + "_dynamic"], problem_data["psi_validation_data"]
        )
    ]
    if name + "_derivative" in data:
        data[name + "_derivative"] = np.hstack(
            (data[name + "_derivative"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_derivative"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.linalg.norm((dynamic - exact_dynamic).dot(matrix))
        for dynamic, matrix in zip(
            problem_data[name + "_dynamic"], problem_data["matrix_validation_data"]
        )
    ]
    if name + "_trajectory" in data:
        data[name + "_trajectory"] = np.hstack(
            (data[name + "_trajectory"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_trajectory"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.count_nonzero(np.multiply(exact_dynamic == 0.0, dynamic != 0.0))
        for dynamic in problem_data[name + "_dynamic"]
    ]
    if name + "_extra" in data:
        data[name + "_extra"] = np.hstack(
            (data[name + "_extra"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_extra"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.count_nonzero(np.multiply(exact_dynamic != 0.0, dynamic == 0.0))
        for dynamic in problem_data[name + "_dynamic"]
    ]
    if name + "_missing" in data:
        data[name + "_missing"] = np.hstack(
            (data[name + "_missing"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_missing"] = np.asarray(metrics)[:, np.newaxis]

    # Training metrics
    metrics = [
        np.linalg.norm(Y_matrix - dynamic.dot(matrix))
        for dynamic, Y_matrix, matrix in zip(
            problem_data[name + "_dynamic"],
            problem_data["Y_train_data"],
            problem_data["psi_train_data"],
        )
    ]
    if name + "_derivative_training" in data:
        data[name + "_derivative_training"] = np.hstack(
            (data[name + "_derivative_training"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_derivative_training"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.linalg.norm(Y_matrix - dynamic.dot(matrix))
        for dynamic, Y_matrix, matrix in zip(
            problem_data[name + "_dynamic"],
            problem_data["delta_train_data"],
            problem_data["matrix_train_data"],
        )
    ]
    if name + "_trajectory_training" in data:
        data[name + "_trajectory_training"] = np.hstack(
            (data[name + "_trajectory_training"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_trajectory_training"] = np.asarray(metrics)[:, np.newaxis]

    # Validation metrics
    metrics = [
        np.linalg.norm(Y_matrix - dynamic.dot(matrix))
        for dynamic, Y_matrix, matrix in zip(
            problem_data[name + "_dynamic"],
            problem_data["Y_validation_data"],
            problem_data["psi_validation_data"],
        )
    ]
    if name + "_derivative_validation" in data:
        data[name + "_derivative_validation"] = np.hstack(
            (data[name + "_derivative_validation"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_derivative_validation"] = np.asarray(metrics)[:, np.newaxis]
    metrics = [
        np.linalg.norm(Y_matrix - dynamic.dot(matrix))
        for dynamic, Y_matrix, matrix in zip(
            problem_data[name + "_dynamic"],
            problem_data["delta_validation_data"],
            problem_data["matrix_validation_data"],
        )
    ]
    if name + "_trajectory_validation" in data:
        data[name + "_trajectory_validation"] = np.hstack(
            (data[name + "_trajectory_validation"], np.asarray(metrics)[:, np.newaxis])
        )
    else:
        data[name + "_trajectory_validation"] = np.asarray(metrics)[:, np.newaxis]

    return data


def cm_to_inch(value):
    return value / 2.54


def plot_stochastic(
    x_axis,
    list_data,
    list_legend,
    title,
    x_label,
    y_label,
    colors,
    markers,
    log_x=True,
    log_y=True,
    fill_between_lines=True,
    save_figure=None,
    legend_location=None,
    outside_legend=False,
):
    plt.rcParams.update({"font.size": 19})
    # plt.figure(figsize=(cm_to_inch(12),cm_to_inch(14)))
    size_marker = 10
    for i in range(len(list_data)):
        mean = np.mean(list_data[i], axis=1)
        std_dev = np.std(list_data[i], axis=1)
        if list_legend != []:
            if log_x and log_y:
                plt.loglog(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if log_x and not log_y:
                plt.semilogx(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and log_y:
                plt.semilogy(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and not log_y:
                plt.plot(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if fill_between_lines:
                plt.fill_between(
                    x_axis, mean - std_dev, mean + std_dev, color=colors[i], alpha=0.2
                )
        else:
            if log_x and log_y:
                plt.loglog(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                )
            if log_x and not log_y:
                plt.semilogx(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                )
            if not log_x and log_y:
                plt.semilogy(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                )
            if not log_x and not log_y:
                plt.plot(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    markerfacecolor="None",
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                    linewidth=2.0,
                )
            if fill_between_lines:
                plt.fill_between(
                    x_axis, mean - std_dev, mean + std_dev, color=colors[i], alpha=0.2
                )
    plt.title(title, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    if list_legend != []:
        if legend_location is not None:
            plt.legend(fontsize=fontsize_legend, loc=legend_location, ncol=2)
        else:
            if outside_legend:
                plt.legend(
                    fontsize=fontsize_legend, loc="center left", bbox_to_anchor=(1, 0.5)
                )
            else:
                plt.legend(fontsize=fontsize_legend)
    plt.tight_layout()
    plt.grid(True, which="both")
    if save_figure is None:
        plt.show()
    else:
        if ".pdf" in save_figure:
            plt.savefig(save_figure, bbox_inches="tight")
        if ".png" in save_figure:
            plt.savefig(save_figure, dpi=600, format="png", bbox_inches="tight")
        # plt.savefig(save_figure)
        plt.close()


def plot_stochastic_side_by_side(
    x_axis,
    list_data_left,
    list_data_right,
    list_legend,
    title_left,
    title_right,
    x_label,
    y_label,
    colors,
    markers,
    linestyle_type=None,
    log_x=True,
    log_y=True,
    fill_between_lines=True,
    figure_size=None,
    save_figure=None,
    legend_location=None,
    outside_legend=False,
):
    plt.rcParams.update({"font.size": 20})
    # plt.figure(figsize=(cm_to_inch(12),cm_to_inch(14)))
    size_marker = 10
    font_size_title = 30
    # fig, axs = plt.subplots(1, 2, figsize=(14,7))
    if figure_size is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=figure_size)
    # fig, axs = plt.subplots(1, 2, figsize=(10,5))
    if linestyle_type is None:
        linestyle_type = ["-"] * len(list_data_left)
    for i in range(len(list_data_left)):
        mean = np.mean(list_data_left[i], axis=1)
        std_dev = np.std(list_data_left[i], axis=1)
        if list_legend != []:
            if log_x and log_y:
                axs[0].loglog(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if log_x and not log_y:
                axs[0].semilogx(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and log_y:
                axs[0].semilogy(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and not log_y:
                axs[0].plot(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if fill_between_lines:
                axs[0].fill_between(
                    x_axis, mean - std_dev, mean + std_dev, color=colors[i], alpha=0.2
                )
        else:
            if log_x and log_y:
                axs[0].loglog(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if log_x and not log_y:
                axs[0].semilogx(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if not log_x and log_y:
                axs[0].semilogy(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if not log_x and not log_y:
                axs[0].plot(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if fill_between_lines:
                axs[0].fill_between(
                    x_axis, mean - std_dev, mean + std_dev, color=colors[i], alpha=0.2
                )

    for i in range(len(list_data_right)):
        mean = np.mean(list_data_right[i], axis=1)
        std_dev = np.std(list_data_right[i], axis=1)
        if list_legend != []:
            if log_x and log_y:
                axs[1].loglog(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if log_x and not log_y:
                axs[1].semilogx(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and log_y:
                axs[1].semilogy(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and not log_y:
                axs[1].plot(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if fill_between_lines:
                axs[1].fill_between(
                    x_axis, mean - std_dev, mean + std_dev, color=colors[i], alpha=0.2
                )
        else:
            if log_x and log_y:
                axs[1].loglog(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if log_x and not log_y:
                axs[1].semilogx(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if not log_x and log_y:
                axs[1].semilogy(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    linestyle=linestyle_type[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if not log_x and not log_y:
                axs[1].plot(
                    x_axis,
                    mean,
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if fill_between_lines:
                axs[1].fill_between(
                    x_axis, mean - std_dev, mean + std_dev, color=colors[i], alpha=0.2
                )

    for ax in axs.flat:
        if x_label == "":
            ax.set(ylabel=y_label, fontsize=25)
        else:
            ax.set(xlabel=x_label, ylabel=y_label)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    if title_left != "" and title_right != "":
        axs[0].set_title(title_left, fontsize=font_size_title)
        axs[1].set_title(title_right, fontsize=font_size_title)
    # plt.ylabel(y_label, fontsize=fontsize)
    # plt.xlabel(x_label, fontsize=fontsize)
    if list_legend != []:
        if legend_location is not None:
            plt.legend(fontsize=fontsize_legend, loc=legend_location)
        else:
            if outside_legend:
                # handles, labels = axs[1].get_legend_handles_labels()
                # fig.legend(handles, labels, bbox_to_anchor=(0.85, 1.05))
                plt.legend(
                    fontsize=fontsize_legend, loc="center left", bbox_to_anchor=(1, 0.5)
                )
            else:
                plt.legend(fontsize=fontsize_legend)
    plt.tight_layout()
    # axs[0].set_yscale('log')
    # axs[1].set_yscale('log')
    axs[0].grid(True, which="both")
    axs[1].grid(True, which="both")
    if save_figure is None:
        plt.show()
    else:
        if ".pdf" in save_figure:
            plt.savefig(save_figure, bbox_inches="tight")
        if ".png" in save_figure:
            plt.savefig(save_figure, dpi=600, format="png", bbox_inches="tight")
        # plt.savefig(save_figure)
        plt.close()


def plot_stochastic_improvement(
    x_axis,
    reference_data,
    list_data,
    list_legend,
    title,
    x_label,
    y_label,
    colors,
    markers,
    log_x=True,
    log_y=True,
    save_figure=None,
    legend_location=None,
    outside_legend=False,
):
    plt.rcParams.update({"font.size": 19})
    plt.figure(figsize=(cm_to_inch(12), cm_to_inch(14)))
    size_marker = 10
    mean_reference = np.mean(reference_data, axis=1)
    for i in range(len(list_data)):
        mean = np.mean(list_data[i], axis=1)
        if list_legend != []:
            if log_x and log_y:
                plt.loglog(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if log_x and not log_y:
                plt.semilogx(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and log_y:
                plt.semilogy(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
            if not log_x and not log_y:
                plt.plot(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                    label=list_legend[i],
                )
        else:
            if log_x and log_y:
                plt.loglog(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if log_x and not log_y:
                plt.semilogx(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if not log_x and log_y:
                plt.semilogy(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
            if not log_x and not log_y:
                plt.plot(
                    x_axis,
                    np.divide(mean, mean_reference),
                    colors[i],
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=2.0,
                )
    plt.title(title, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    if list_legend != []:
        if legend_location is not None:
            plt.legend(fontsize=fontsize_legend, loc=legend_location)
        else:
            if outside_legend:
                plt.legend(
                    fontsize=fontsize_legend, loc="center left", bbox_to_anchor=(1, 0.5)
                )
            else:
                plt.legend(fontsize=fontsize_legend)
    plt.tight_layout()
    plt.grid(True, which="both")
    if save_figure is None:
        plt.show()
    else:
        plt.savefig(save_figure, bbox_inches="tight")
        # plt.savefig(save_figure)
        plt.close()


def plot_heatmaps(
    x_axis,
    y_axis,
    list_data,
    list_legend,
    title,
    labels,
    log_x=True,
    log_z=True,
    log_y=False,
    save_figure=None,
    label_heatmap="Error (decimal log)",
    color_min=None,
    color_max=None,
    minimum_values=None,
):
    interpolation_method = "cubic"
    plt.rcParams.update({"font.size": 19})
    title_font_size = 19
    ylabel_font_size = 15
    colormap = mpl.cm.viridis
    # plt.xkcd()
    if log_x:
        x = np.log10(x_axis.flatten())
    else:
        x = x_axis.flatten()

    if log_y:
        y = np.log10(y_axis.flatten())
    else:
        y = y_axis.flatten()

    # define grid.
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    dict_parameters = {"fontsize": 20}
    if len(list_data) == 1:

        if log_z:
            z_1 = np.log10(list_data[0].flatten())
        else:
            z_1 = list_data[0].flatten()
        zi_1 = griddata(
            (x, y), z_1, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        min_val = np.min(zi_1)
        max_val = np.max(zi_1)
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(title)
        axs.contour(
            xi, yi, zi_1, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax1 = axs.contourf(xi, yi, zi_1, 15, cmap=colormap, vmin=min_val, vmax=max_val)
        axs.set_title(list_legend[0], fontdict=dict_parameters, position=(0.5, 0.6))
        axs.xaxis.set_visible(False)
        axs.set_ylabel(labels[1], fontsize=fontsize)
        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.05
        )
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        cmap = colormap
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cbar.ax.set_ylabel("# of contacts", rotation=270)

    if len(list_data) == 2:
        if log_z:
            z_1 = np.log10(list_data[0].flatten())
            z_2 = np.log10(list_data[1].flatten())
        else:
            z_1 = list_data[0].flatten()
            z_2 = list_data[1].flatten()
        zi_1 = griddata(
            (x, y), z_1, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_2 = griddata(
            (x, y), z_2, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        min_val = min(np.min(zi_1), np.min(zi_2))
        max_val = max(np.max(zi_1), np.max(zi_2))
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(title)
        axs[0].contour(
            xi, yi, zi_1, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax1 = axs[0].contourf(
            xi, yi, zi_1, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0].text(
            0.5,
            0.8,
            list_legend[0],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[0].transAxes,
        )
        axs[0].set_xlabel(labels[0], fontsize=fontsize)
        axs[0].set_ylabel(labels[1], fontsize=fontsize)
        axs[1].contour(
            xi, yi, zi_2, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax2 = axs[1].contourf(
            xi, yi, zi_2, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1].text(
            0.5,
            0.8,
            list_legend[1],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[1].transAxes,
        )
        axs[1].yaxis.set_visible(False)
        axs[1].set_xlabel(labels[0], fontsize=fontsize)
        cb_ax = fig.add_axes()
        cmap = colormap
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(label_heatmap, rotation=270)

    if len(list_data) == 3:
        if log_z:
            z_1 = np.log10(list_data[0].flatten())
            z_2 = np.log10(list_data[1].flatten())
            z_3 = np.log10(list_data[2].flatten())
        else:
            z_1 = list_data[0].flatten()
            z_2 = list_data[1].flatten()
            z_3 = list_data[2].flatten()
        zi_1 = griddata(
            (x, y), z_1, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_2 = griddata(
            (x, y), z_2, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_3 = griddata(
            (x, y), z_3, (xi[None, :], yi[:, None]), method=interpolation_method
        )

        if color_min is None and color_max is None:
            min_val = min(np.min(zi_1), np.min(zi_2), np.min(zi_3))
            max_val = max(np.max(zi_1), np.max(zi_2), np.max(zi_3))
        else:
            min_val = color_min
            max_val = color_max

        fig, axs = plt.subplots(3, 1, figsize=(5, 10))
        fig.suptitle(title, fontsize=title_font_size)
        axs[0].contour(
            xi, yi, zi_1, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax1 = axs[0].contourf(
            xi, yi, zi_1, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0].text(
            0.5,
            0.8,
            list_legend[0],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[0].transAxes,
        )
        axs[0].xaxis.set_visible(False)
        axs[0].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[1].contour(
            xi, yi, zi_2, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax2 = axs[1].contourf(
            xi, yi, zi_2, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1].text(
            0.5,
            0.8,
            list_legend[1],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[1].transAxes,
        )
        axs[1].xaxis.set_visible(False)
        axs[1].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[2].contour(
            xi, yi, zi_3, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax3 = axs[2].contourf(
            xi, yi, zi_3, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[2].text(
            0.5,
            0.8,
            list_legend[2],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[2].transAxes,
        )
        axs[2].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[2].set_xlabel(labels[0], fontsize=fontsize)
        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.05
        )
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        cmap = colormap
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cbar.ax.get_yaxis().labelpad = 24
        cbar.ax.set_ylabel(label_heatmap, rotation=270, fontsize=fontsize)

    if len(list_data) == 4:
        if log_z:
            z_1 = np.log10(list_data[0].flatten())
            z_2 = np.log10(list_data[1].flatten())
            z_3 = np.log10(list_data[2].flatten())
            z_4 = np.log10(list_data[3].flatten())
        else:
            z_1 = list_data[0].flatten()
            z_2 = list_data[1].flatten()
            z_3 = list_data[2].flatten()
            z_4 = list_data[3].flatten()
        zi_1 = griddata(
            (x, y), z_1, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_2 = griddata(
            (x, y), z_2, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_3 = griddata(
            (x, y), z_3, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_4 = griddata(
            (x, y), z_4, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        min_val = min(np.min(zi_1), np.min(zi_2), np.min(zi_3), np.min(zi_4))
        max_val = max(np.max(zi_1), np.max(zi_2), np.max(zi_3), np.max(zi_4))
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(title)
        axs[0, 0].contour(
            xi, yi, zi_1, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax1 = axs[0, 0].contourf(
            xi, yi, zi_1, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0, 0].text(
            0.5,
            0.8,
            list_legend[0],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[0, 0].transAxes,
        )
        axs[0, 0].xaxis.set_visible(False)
        axs[0, 0].set_ylabel(labels[1])
        axs[0, 1].contour(
            xi, yi, zi_2, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax2 = axs[0, 1].contourf(
            xi, yi, zi_2, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0, 1].text(
            0.5,
            0.8,
            list_legend[1],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[0, 1].transAxes,
        )
        axs[0, 1].xaxis.set_visible(False)
        axs[0, 1].yaxis.set_visible(False)
        axs[1, 0].contour(
            xi, yi, zi_3, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax3 = axs[1, 0].contourf(
            xi, yi, zi_3, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1, 0].text(
            0.5,
            0.8,
            list_legend[2],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[1, 0].transAxes,
        )
        axs[1, 0].set_ylabel(labels[1], fontsize=fontsize)
        axs[1, 0].set_xlabel(labels[0], fontsize=fontsize)
        axs[1, 1].contour(
            xi, yi, zi_4, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax4 = axs[1, 1].contourf(
            xi, yi, zi_4, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1, 1].text(
            0.5,
            0.8,
            list_legend[3],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[1, 1].transAxes,
        )
        axs[1, 1].yaxis.set_visible(False)
        axs[1, 1].set_xlabel(labels[0], fontsize=fontsize)
        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.05
        )
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        cmap = colormap
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(label_heatmap, rotation=270)

    if len(list_data) == 6:
        dict_parameters = {"fontsize": 14}
        if log_z:
            z_1 = np.log10(list_data[0].flatten())
            z_2 = np.log10(list_data[1].flatten())
            z_3 = np.log10(list_data[2].flatten())
            z_4 = np.log10(list_data[3].flatten())
            z_5 = np.log10(list_data[4].flatten())
            z_6 = np.log10(list_data[5].flatten())
        else:
            z_1 = list_data[0].flatten()
            z_2 = list_data[1].flatten()
            z_3 = list_data[2].flatten()
            z_4 = list_data[3].flatten()
            z_5 = list_data[4].flatten()
            z_6 = list_data[5].flatten()
        zi_1 = griddata(
            (x, y), z_1, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_2 = griddata(
            (x, y), z_2, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_3 = griddata(
            (x, y), z_3, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_4 = griddata(
            (x, y), z_4, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_5 = griddata(
            (x, y), z_5, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_6 = griddata(
            (x, y), z_6, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        min_val = min(
            np.min(zi_1),
            np.min(zi_2),
            np.min(zi_3),
            np.min(zi_4),
            np.min(zi_5),
            np.min(zi_6),
        )
        max_val = max(
            np.max(zi_1),
            np.max(zi_2),
            np.max(zi_3),
            np.max(zi_4),
            np.max(zi_5),
            np.max(zi_6),
        )
        fig, axs = plt.subplots(2, 3)
        fig.suptitle(title)
        axs[0, 0].contour(
            xi, yi, zi_1, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax1 = axs[0, 0].contourf(
            xi, yi, zi_1, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0, 0].set_title(
            list_legend[0], fontdict=dict_parameters, position=(0.5, 0.2)
        )
        axs[0, 0].xaxis.set_visible(False)
        axs[0, 0].set_ylabel(labels[1], fontsize=fontsize)

        axs[0, 1].contour(
            xi, yi, zi_2, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax2 = axs[0, 1].contourf(
            xi, yi, zi_2, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0, 1].set_title(
            list_legend[1], fontdict=dict_parameters, position=(0.5, 0.2)
        )
        axs[0, 1].xaxis.set_visible(False)
        axs[0, 1].yaxis.set_visible(False)

        axs[0, 2].contour(
            xi, yi, zi_3, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax3 = axs[0, 2].contourf(
            xi, yi, zi_3, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0, 2].set_title(
            list_legend[2], fontdict=dict_parameters, position=(0.5, 0.2)
        )
        axs[0, 2].xaxis.set_visible(False)
        axs[0, 2].yaxis.set_visible(False)

        axs[1, 0].contour(
            xi, yi, zi_4, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax4 = axs[1, 0].contourf(
            xi, yi, zi_4, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1, 0].set_title(
            list_legend[3], fontdict=dict_parameters, position=(0.5, 0.2)
        )
        axs[1, 0].set_ylabel(labels[1], fontsize=fontsize)
        axs[1, 0].set_xlabel(labels[0], fontsize=fontsize)

        axs[1, 1].contour(
            xi, yi, zi_5, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax5 = axs[1, 1].contourf(
            xi, yi, zi_5, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1, 1].set_title(
            list_legend[4], fontdict=dict_parameters, position=(0.5, 0.2)
        )
        axs[1, 1].yaxis.set_visible(False)
        axs[1, 1].set_xlabel(labels[0], fontsize=fontsize)

        axs[1, 2].contour(
            xi, yi, zi_6, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax6 = axs[1, 2].contourf(
            xi, yi, zi_6, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1, 2].set_title(
            list_legend[5], fontdict=dict_parameters, position=(0.5, 0.2)
        )
        axs[1, 2].yaxis.set_visible(False)
        axs[1, 2].set_xlabel(labels[0], fontsize=fontsize)

        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.05
        )
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        cmap = colormap
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(label_heatmap, rotation=270)

    if save_figure is None:
        plt.show()
    else:
        plt.savefig(save_figure, format="pdf", bbox_inches="tight")
        plt.close()


def plot_heatmaps_small(
    x_axis,
    y_axis,
    list_data,
    list_legend,
    title,
    labels,
    log_x=True,
    log_z=True,
    log_y=False,
    save_figure=None,
    label_heatmap="Error (decimal log)",
    color_min=None,
    color_max=None,
    minimum_values=None,
):
    interpolation_method = "cubic"
    plt.rcParams.update({"font.size": 19})
    title_font_size = 19
    ylabel_font_size = 15
    colormap = mpl.cm.viridis
    # plt.xkcd()
    if log_x:
        x = np.log10(x_axis.flatten())
    else:
        x = x_axis.flatten()

    if log_y:
        y = np.log10(y_axis.flatten())
    else:
        y = y_axis.flatten()

    # define grid.
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    dict_parameters = {"fontsize": 20}
    if len(list_data) == 3:
        if log_z:
            z_1 = np.log10(list_data[0].flatten())
            z_2 = np.log10(list_data[1].flatten())
            z_3 = np.log10(list_data[2].flatten())
        else:
            z_1 = list_data[0].flatten()
            z_2 = list_data[1].flatten()
            z_3 = list_data[2].flatten()
        zi_1 = griddata(
            (x, y), z_1, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_2 = griddata(
            (x, y), z_2, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_3 = griddata(
            (x, y), z_3, (xi[None, :], yi[:, None]), method=interpolation_method
        )

        if color_min is None and color_max is None:
            min_val = min(np.min(zi_1), np.min(zi_2), np.min(zi_3))
            max_val = max(np.max(zi_1), np.max(zi_2), np.max(zi_3))
        else:
            min_val = color_min
            max_val = color_max

        fig, axs = plt.subplots(3, 1)
        # fig, axs = plt.subplots(3, 1,figsize=(5,10))
        fig.suptitle(title, fontsize=title_font_size)
        axs[0].contour(
            xi, yi, zi_1, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax1 = axs[0].contourf(
            xi, yi, zi_1, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0].text(
            0.5,
            0.8,
            list_legend[0],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[0].transAxes,
        )
        axs[0].xaxis.set_visible(False)
        # axs[0].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[1].contour(
            xi, yi, zi_2, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax2 = axs[1].contourf(
            xi, yi, zi_2, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1].text(
            0.5,
            0.8,
            list_legend[1],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[1].transAxes,
        )
        axs[1].xaxis.set_visible(False)
        axs[1].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[2].contour(
            xi, yi, zi_3, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax3 = axs[2].contourf(
            xi, yi, zi_3, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[2].text(
            0.5,
            0.8,
            list_legend[2],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[2].transAxes,
        )
        # axs[2].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[2].set_xlabel(labels[0], fontsize=fontsize)
        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.05
        )
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        cmap = colormap
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cbar.ax.get_yaxis().labelpad = 24
        cbar.ax.set_ylabel(label_heatmap, rotation=270, fontsize=fontsize)

    if save_figure is None:
        plt.show()
    else:
        plt.savefig(save_figure, format="pdf", bbox_inches="tight")
        plt.close()


def plot_heatmaps_blogpost(
    x_axis,
    y_axis,
    list_data,
    list_legend,
    title,
    labels,
    log_x=True,
    log_z=True,
    log_y=False,
    save_figure=None,
    label_heatmap="Error (decimal log)",
    color_min=None,
    color_max=None,
    minimum_values=None,
):

    plt.xkcd()
    interpolation_method = "cubic"
    plt.rcParams.update({"font.size": 19})
    title_font_size = 19
    ylabel_font_size = 15
    colormap = mpl.cm.viridis
    # plt.xkcd()
    if log_x:
        x = np.log10(x_axis.flatten())
    else:
        x = x_axis.flatten()

    if log_y:
        y = np.log10(y_axis.flatten())
    else:
        y = y_axis.flatten()

    # define grid.
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    dict_parameters = {"fontsize": 20}

    if len(list_data) == 3:
        if log_z:
            z_1 = np.log10(list_data[0].flatten())
            z_2 = np.log10(list_data[1].flatten())
            z_3 = np.log10(list_data[2].flatten())
        else:
            z_1 = list_data[0].flatten()
            z_2 = list_data[1].flatten()
            z_3 = list_data[2].flatten()
        zi_1 = griddata(
            (x, y), z_1, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_2 = griddata(
            (x, y), z_2, (xi[None, :], yi[:, None]), method=interpolation_method
        )
        zi_3 = griddata(
            (x, y), z_3, (xi[None, :], yi[:, None]), method=interpolation_method
        )

        if color_min is None and color_max is None:
            min_val = min(np.min(zi_1), np.min(zi_2), np.min(zi_3))
            max_val = max(np.max(zi_1), np.max(zi_2), np.max(zi_3))
        else:
            min_val = color_min
            max_val = color_max

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=title_font_size)
        axs[0].contour(
            xi, yi, zi_1, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax1 = axs[0].contourf(
            xi, yi, zi_1, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[0].text(
            0.5,
            0.8,
            list_legend[0],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[0].transAxes,
        )
        # axs[0].xaxis.set_visible(False)
        axs[0].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[0].set_xticks([-7, -5, -3])
        axs[0].set_xlabel(labels[0], fontsize=fontsize)

        axs[1].contour(
            xi, yi, zi_2, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax2 = axs[1].contourf(
            xi, yi, zi_2, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[1].text(
            0.5,
            0.8,
            list_legend[1],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[1].transAxes,
        )
        # axs[1].xaxis.set_visible(False)
        axs[1].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[1].set_xlabel(labels[0], fontsize=fontsize)
        axs[1].set_xticks([-7, -5, -3])
        axs[2].contour(
            xi, yi, zi_3, 15, linewidths=0.5, colors="k", vmin=min_val, vmax=max_val
        )
        ax3 = axs[2].contourf(
            xi, yi, zi_3, 15, cmap=colormap, vmin=min_val, vmax=max_val
        )
        axs[2].text(
            0.5,
            0.8,
            list_legend[2],
            fontdict=dict_parameters,
            horizontalalignment="center",
            transform=axs[2].transAxes,
        )
        axs[2].set_ylabel(labels[1], fontsize=ylabel_font_size)
        axs[2].set_xlabel(labels[0], fontsize=fontsize)
        axs[2].set_xticks([-7, -5, -3])
        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.05
        )
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        cmap = colormap
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cbar.ax.get_yaxis().labelpad = 24
        cbar.ax.set_ylabel(label_heatmap, rotation=270, fontsize=fontsize)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    if save_figure is None:
        plt.show()
    else:
        plt.savefig(save_figure, format="png", bbox_inches="tight")
        plt.close()
