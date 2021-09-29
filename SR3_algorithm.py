# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:20:24 2021

@author: pccom
"""
import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from pysindy.utils import prox_l1, prox_l0


def update_full_coef(cho, psi_transpose_y, coef_sparse, nu, iters):
    """Update the unregularized weight vector"""
    b = psi_transpose_y + coef_sparse / nu
    coef_full = cho_solve(cho, b)
    return coef_full, iters + 1


def update_full_coef_constraints(
    H,
    psi_transpose_y,
    coef_sparse,
    nu,
    equality_constraint_matrix,
    equality_constraint_vector,
    iters,
):
    inv1 = np.linalg.inv(H)
    inv1_mod = np.kron(np.eye(coef_sparse.shape[1]), inv1)
    inv2 = np.linalg.inv(
        equality_constraint_matrix.dot(inv1_mod).dot(equality_constraint_matrix.T)
    )
    RHS_val = -equality_constraint_vector + equality_constraint_matrix.dot(
        inv1_mod
    ).dot(psi_transpose_y.flatten(order="F"))
    phi = inv2.dot(RHS_val)
    xi = inv1_mod.dot(
        -equality_constraint_matrix.T.dot(phi) + psi_transpose_y.flatten(order="F")
    )
    xi_v2 = xi.reshape(coef_sparse.T.shape, order="C").T
    return xi_v2, iters + 1


def update_sparse_coef(coef_full, threshold, type_penalty="l0"):
    """Update the regularized weight vector"""
    if type_penalty == "l1":
        coef_sparse = prox_l1(coef_full, threshold)
    else:
        coef_sparse = prox_l0(coef_full, threshold)
    return coef_sparse


def evaluate_objective(
    Psi, Y, coef_full, coef_sparse, nu, threshold, type_penalty="l0"
):
    """objective function"""
    R2 = np.linalg.norm(Y - np.dot(Psi, coef_full)) ** 2
    D2 = np.linalg.norm(coef_full - coef_sparse) ** 2
    if type_penalty == "l1":
        return (
            0.5 * np.sum(R2)
            + regularization_l1(coef_full, 0.5 * threshold ** 2 / nu)
            + 0.5 * np.sum(D2) / nu
        )
    else:
        return (
            0.5 * np.sum(R2)
            + regularization_l0(coef_full, 0.5 * threshold ** 2 / nu)
            + 0.5 * np.sum(D2) / nu
        )


def regularization_l1(x, lam):
    return lam * np.sum(np.abs(x))


def regularization_l0(x, lam):
    return lam * np.count_nonzero(x)


def convergence_criterion(data, nu):
    """Calculate the convergence criterion for the optimization"""
    this_coef = data[-1]
    if len(data) > 1:
        last_coef = data[-2]
    else:
        last_coef = np.zeros_like(this_coef)
    err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2)) / nu
    return err_coef


def run_SR3(
    Psi,
    Y,
    parameters,
    regularization="l0",
    max_iter=30,
    tolerance=1.0e-14,
    equality_constraint_matrix=None,
    equality_constraint_vector=None,
):
    nu = parameters[0]
    threshold = parameters[1]

    Psi = Psi.T
    Y = Y.T
    n_samples, n_features = Psi.shape
    _, n_dimension = Y.shape
    coef_sparse = np.zeros((n_features, n_dimension))
    # Precompute some objects for upcoming least-squares solves.
    # Assumes that self.nu is fixed throughout optimization procedure.
    H = np.dot(Psi.T, Psi) + np.diag(np.full(n_features, 1.0 / nu))
    psi_transpose_y = np.dot(Psi.T, Y)
    if equality_constraint_matrix is None and equality_constraint_vector is None:
        cho = cho_factor(H)
    obj_his = []
    iters = 0
    for i in range(max_iter):
        if equality_constraint_matrix is None and equality_constraint_vector is None:
            coef_full, iters = update_full_coef(
                cho, psi_transpose_y, coef_sparse, nu, iters
            )
        else:
            coef_full, iters = update_full_coef_constraints(
                H,
                psi_transpose_y,
                coef_sparse,
                nu,
                equality_constraint_matrix,
                equality_constraint_vector,
                iters,
            )
        coef_sparse = update_sparse_coef(
            coef_full, threshold, type_penalty=regularization
        )
        obj_his.append(
            evaluate_objective(
                Psi,
                Y,
                coef_full,
                coef_sparse,
                nu,
                threshold,
                type_penalty=regularization,
            )
        )
        if convergence_criterion(obj_his, nu) < tolerance:
            break
    coef_ = coef_sparse.T
    # coef_full_ = coef_full.T
    # obj_his = obj_his
    return coef_


# Select the parameter that minimizes the loss_function.
def regularization_selection_SR3(
    loss_function,
    type_algorithm,
    options,
    show_progress=False,
    skip_minimization_check=False,
):
    if type_algorithm == "minimization":
        from scipy.optimize import minimize

        reg = minimize(
            lambda x: loss_function(x),
            np.asarray([1.0e3, 0.00]),
            method="SLSQP",
            bounds=[(0.01, np.inf), (0.0, np.inf)],
            tol=1.0e-12,
            options={"maxiter": 1000, "ftol": 1e-09},
        )

        # reg = minimize_scalar(lambda x: loss_function(x), method='bounded', bounds=(options['min'], options['max']),  options = {'xatol': options['xatol'], 'maxiter': options['number_evaluations']})
        if skip_minimization_check:
            if show_progress:
                print(
                    "Final parameter selected: ",
                    reg["x"],
                    " Number of calls: ",
                    reg,
                    " Max number calls: ",
                )
            return reg["x"]
        else:
            # Compare to the loss if we have zero regularization.
            if loss_function(reg["x"]) <= loss_function(0.0):
                if show_progress:
                    print(
                        "Final parameter selected: ",
                        reg["x"],
                        " Number of calls: ",
                        reg,
                        " Max number calls: ",
                        options["number_evaluations"],
                    )
                return reg["x"]
            else:
                if show_progress:
                    print(
                        "Final parameter selected: ",
                        0.0,
                        " Number of calls: ",
                        reg,
                        " Max number calls: ",
                        options["number_evaluations"],
                    )
                return 0.0

    if type_algorithm == "bayesian":
        from hyperopt import fmin, hp, tpe, Trials
        from auxiliary_functions import plot_params_tried

        if options["distribution"] == "log-uniform":
            trls = Trials()
            # SPACE = [hp.uniform(str(dim), 1.0e-3, 1.0e5) for dim in range(2)]
            SPACE = {"x": hp.loguniform("x", 4, 24), "y": hp.loguniform("y", -20, -3)}

            def modified_loss_function(x):
                vect = np.asarray([x["x"], x["y"]])
                return loss_function(vect)

            reg = fmin(
                modified_loss_function,
                space=SPACE,
                trials=trls,
                max_evals=options["number_evaluations"],
                algo=tpe.suggest,
                show_progressbar=False,
            )

            print("Regularization parameter: ", reg)
            if show_progress:
                plot_params_tried(SPACE, trls, dist_plots=True)
            return np.asarray([reg["x"], reg["y"]])

    assert False, "Type of algorithm selected is incorrect"


def train_SR3(
    Psi,
    Y,
    testing_function,
    type_regularization="l0",
    equality_constraint_matrix=None,
    equality_constraint_vector=None,
    type_parameter_search="bayesian",
    options={
        "min": 0.0,
        "max": 1.0,
        "number_evaluations": 2000,
        "distribution": "log-uniform",
        "inner_iterations": 5000,
    },
    show_progress=False,
    skip_minimization_check=True,
):
    alpha = regularization_selection_SR3(
        lambda x: testing_function.evaluate(
            run_SR3(
                Psi,
                Y,
                x,
                regularization=type_regularization,
                equality_constraint_matrix=equality_constraint_matrix,
                equality_constraint_vector=equality_constraint_vector,
            )
        ),
        type_parameter_search,
        options,
        show_progress=show_progress,
        skip_minimization_check=True,
    )
    return run_SR3(
        Psi,
        Y,
        alpha,
        equality_constraint_matrix=equality_constraint_matrix,
        equality_constraint_vector=equality_constraint_vector,
    )
