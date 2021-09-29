import numpy as np
import time

ts = time.time()
from scipy.optimize import minimize_scalar
import pickle
import math

"""# Miscelaneous Functions

"""


def display_statistics(dynamic, exact_dynamic, noise, formulation_matrices):
    print("Noise level: " + str(noise))
    print(
        "Recovery error :"
        + str(np.linalg.norm(dynamic - exact_dynamic))
        + "\tDerivative inference :"
        + str(
            np.linalg.norm(
                np.dot(dynamic - exact_dynamic, formulation_matrices["Psi_validation"])
            )
        )
        + "\tTrajectory inference :"
        + str(
            np.linalg.norm(
                np.dot(
                    dynamic - exact_dynamic, formulation_matrices["matrix_validation"]
                )
            )
        )
        + "\tPrimal gap (derivative training): "
        + str(
            np.linalg.norm(
                formulation_matrices["Y_train"]
                - np.dot(dynamic, formulation_matrices["Psi_train"])
            )
        )
        + "\tPrimal gap (trajectory training): "
        + str(
            np.linalg.norm(
                formulation_matrices["delta_train"]
                - np.dot(dynamic, formulation_matrices["matrix_train"])
            )
        )
        + "\tExtra terms :"
        + str(np.count_nonzero(np.multiply(exact_dynamic == 0.0, dynamic != 0.0)))
        + "\tMissing terms :"
        + str(np.count_nonzero(np.multiply(exact_dynamic != 0.0, dynamic == 0.0)))
    )
    print()
    return


def evaluate_dynamic(
    data,
    dynamic,
    exact_dynamic,
    noise,
    noise_levels,
    formulation_matrices,
    show_output=False,
):
    index = noise_levels.index(noise)
    assert (
        "Psi_validation" in formulation_matrices
        and "matrix_validation" in formulation_matrices
    ), "Missing validation matrices to compute metrics."
    assert (
        "Y_validation" in formulation_matrices
        and "delta_validation" in formulation_matrices
    ), "Missing validation matrices to compute metrics."
    assert (
        "Psi_train" in formulation_matrices and "matrix_train" in formulation_matrices
    ), "Missing training matrices to compute metrics."
    assert (
        "Y_train" in formulation_matrices and "delta_train" in formulation_matrices
    ), "Missing training matrices to compute metrics."

    # Recovery_error
    if "accuracy_recovery" in data:
        data["accuracy_recovery"][index] = np.linalg.norm(dynamic - exact_dynamic)
    else:
        data["accuracy_recovery"] = np.zeros(len(noise_levels))
        data["accuracy_recovery"][index] = np.linalg.norm(dynamic - exact_dynamic)
    # Inference of derivatives.
    if "accuracy_data" in data:
        data["accuracy_data"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["Psi_validation"])
        )
    else:
        data["accuracy_data"] = np.zeros(len(noise_levels))
        data["accuracy_data"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["Psi_validation"])
        )
    # Inference of trajectories
    if "accuracy_trajectory" in data:
        data["accuracy_trajectory"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["matrix_validation"])
        )
    else:
        data["accuracy_trajectory"] = np.zeros(len(noise_levels))
        data["accuracy_trajectory"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["matrix_validation"])
        )
    # Extra terms
    if "extra" in data:
        data["extra"][index] = np.count_nonzero(
            np.multiply(exact_dynamic == 0.0, dynamic != 0.0)
        )
    else:
        data["extra"] = np.zeros(len(noise_levels))
        data["extra"][index] = np.count_nonzero(
            np.multiply(exact_dynamic == 0.0, dynamic != 0.0)
        )
    # Missing terms
    if "missing" in data:
        data["missing"][index] = np.count_nonzero(
            np.multiply(exact_dynamic != 0.0, dynamic == 0.0)
        )
    else:
        data["missing"] = np.zeros(len(noise_levels))
        data["missing"][index] = np.count_nonzero(
            np.multiply(exact_dynamic != 0.0, dynamic == 0.0)
        )
    # Inference of derivatives (old measure).
    if "accuracy_data_old" in data:
        data["accuracy_data_old"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["Psi_validation"])
            - formulation_matrices["Y_validation"]
        )
    else:
        data["accuracy_data_old"] = np.zeros(len(noise_levels))
        data["accuracy_data_old"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["Psi_validation"])
            - formulation_matrices["Y_validation"]
        )
    # Inference of trajectories (old measure).
    if "accuracy_trajectory_old" in data:
        data["accuracy_trajectory_old"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["matrix_validation"])
            - formulation_matrices["delta_validation"]
        )
    else:
        data["accuracy_trajectory_old"] = np.zeros(len(noise_levels))
        data["accuracy_trajectory_old"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["matrix_validation"])
            - formulation_matrices["delta_validation"]
        )
    # Inference of derivatives (training_data)
    if "accuracy_data_train" in data:
        data["accuracy_data_train"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["Psi_train"])
        )
    else:
        data["accuracy_data_train"] = np.zeros(len(noise_levels))
        data["accuracy_data_train"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["Psi_train"])
        )
    # Inference of trajectories (training_data)
    if "accuracy_trajectory_train" in data:
        data["accuracy_trajectory_train"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["matrix_train"])
        )
    else:
        data["accuracy_trajectory_train"] = np.zeros(len(noise_levels))
        data["accuracy_trajectory_train"][index] = np.linalg.norm(
            np.dot(dynamic - exact_dynamic, formulation_matrices["matrix_train"])
        )
    # Inference of derivatives (old measure training data).
    if "accuracy_data_old_train" in data:
        data["accuracy_data_old_train"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["Psi_train"])
            - formulation_matrices["Y_train"]
        )
    else:
        data["accuracy_data_old_train"] = np.zeros(len(noise_levels))
        data["accuracy_data_old_train"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["Psi_train"])
            - formulation_matrices["Y_train"]
        )
    # Inference of trajectories (old measure).
    if "accuracy_trajectory_old_train" in data:
        data["accuracy_trajectory_old_train"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["matrix_train"])
            - formulation_matrices["delta_train"]
        )
    else:
        data["accuracy_trajectory_old_train"] = np.zeros(len(noise_levels))
        data["accuracy_trajectory_old_train"][index] = np.linalg.norm(
            np.dot(dynamic, formulation_matrices["matrix_train"])
            - formulation_matrices["delta_train"]
        )
    if show_output:
        print(
            "Recovery error :"
            + str(data["accuracy_recovery"][index])
            + "\tDerivative inference :"
            + str(data["accuracy_data"][index])
            + "\tTrajectory inference :"
            + str(data["accuracy_trajectory"][index])
            + "\tExtra terms :"
            + str(data["extra"][index])
            + "\tMissing terms :"
            + str(data["missing"][index])
        )
    return data


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def plot_params_tried(p_space, trials, dist_plots=True, **kwargs):
    """
    Plots the parameters tried by hyperopt.
    p_space: Hyperopt parameter space.
    trials: Hyperopt trials object.
    dist_plots: If True, plots the distribution of values tried for each parameter.
    If False, plots a scatter plot of loss vs parameter values. Default True.
    params: Parameters to be plotted. Default = All parameters in p_space.
    ncols: Number of columns in the figure. Default = 3
    figsize: Figure size. Default = (15, 5*nrows).
    constrained_layout: Constrained layout. Default = True.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from hyperopt import space_eval

    defaults = {"params": p_space.keys(), "ncols": 2, "constrained_layout": True}
    defaults = {k: v for k, v in defaults.items() if k not in kwargs}
    kwargs = {**kwargs, **defaults}
    params = kwargs["params"]
    del kwargs["params"]
    results_df = pd.DataFrame(columns=["tid", *params, "loss", "status"])
    for t in trials.trials:
        result_dict = t["misc"]["vals"]
        result_dict = {d: v[0] for d, v in result_dict.items()}
        result_dict = space_eval(p_space, result_dict)
        result_dict["tid"] = t["tid"]
        result_dict.update(t["result"])
        results_df = pd.concat(
            [results_df, pd.DataFrame(result_dict, index=[0])],
            axis=0,
            ignore_index=True,
        )
    results_df = results_df[results_df.status == "ok"]
    results_df = results_df.infer_objects()
    nrows = int(math.ceil((len(params) + 1) / kwargs["ncols"]))
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (15, 5 * nrows)
    f, axs = plt.subplots(nrows=nrows, **kwargs)
    axs = trim_axs(axs, len(params) + 1)
    f.suptitle("Parameter Space tried by Hyperopt & Loss")
    for ax, p in zip(axs, [*params, "loss"]):
        if p == "loss":
            sns.scatterplot(x=results_df.tid, y=results_df.loc[:, p], alpha=0.8, ax=ax)
            ax.set_title("Scatterplot of Loss vs trial number.")
        elif dist_plots:
            if pd.api.types.is_bool_dtype(
                results_df.loc[:, p]
            ) or pd.api.types.is_string_dtype(results_df.loc[:, p]):
                sns.countplot(results_df.loc[:, p], ax=ax)
            else:
                sns.distplot(results_df.loc[:, p].astype("float"), bins=10, ax=ax)
            ax.set_title(p)
        else:
            sns.scatterplot(x=results_df.loc[:, p], y=results_df.loss, alpha=0.8, ax=ax)
            ax.set_title(f"{p} vs loss")
    plt.show()
    return results_df


# Select the parameter that minimizes the loss_function.
def regularization_selection(
    loss_function,
    type_algorithm,
    options,
    show_progress=False,
    skip_minimization_check=False,
):
    if type_algorithm == "brute":
        x = np.linspace(options["min"], options["max"], options["number_evaluations"])
        min_error = loss_function(x[0])
        min_regularization = x[0]
        error = [loss_function(x[0])]
        for i in range(1, options["number_evaluations"]):
            error.append(loss_function(x[i]))
            if min_error > error[-1]:
                min_error = error[-1]
                min_regularization = x[i]
        if show_progress:
            import matplotlib.pyplot as plt

            plt.plot(x, error)
            plt.show()
        return min_regularization

    if type_algorithm == "minimization":
        from scipy.optimize import minimize_scalar

        reg = minimize_scalar(
            lambda x: loss_function(x),
            method="bounded",
            bounds=(options["min"], options["max"]),
            options={
                "xatol": options["xatol"],
                "maxiter": options["number_evaluations"],
            },
        )
        if reg["success"]:
            if skip_minimization_check:
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
        else:
            return regularization_selection(loss_function, "brute", options)

    if type_algorithm == "bayesian":
        from hyperopt import fmin, hp, tpe, Trials
        from auxiliary_functions import plot_params_tried

        if options["distribution"] == "uniform":
            trls = Trials()
            SPACE = {
                "regularization": hp.uniform(
                    "regularization", options["min"], options["max"]
                )
            }
            reg = fmin(
                lambda x: loss_function(x["regularization"]),
                space=SPACE,
                trials=trls,
                max_evals=options["number_evaluations"],
                algo=tpe.suggest,
                show_progressbar=False,
            )
            print("Regularization parameter: ", reg["regularization"])
            if show_progress:
                plot_params_tried(SPACE, trls, dist_plots=True)
            if skip_minimization_check:
                return reg["regularization"]
            else:
                if loss_function(reg["regularization"]) <= loss_function(
                    options["max"]
                ):
                    return reg["regularization"]
                else:
                    return options["max"]

        if options["distribution"] == "log-uniform":
            trls = Trials()
            SPACE = {
                "regularization": hp.loguniform(
                    "regularization", options["min"], options["max"]
                )
            }
            reg = fmin(
                lambda x: loss_function(x["regularization"]),
                space=SPACE,
                trials=trls,
                max_evals=options["number_evaluations"],
                algo=tpe.suggest,
                show_progressbar=False,
            )
            if show_progress:
                plot_params_tried(SPACE, trls, dist_plots=True)
            return reg["regularization"]
            # return np.log(reg['regularization'])

        if options["distribution"] == "log-normal":
            trls = Trials()
            SPACE = {
                "regularization": hp.lognormal(
                    "regularization", options["mean"], options["sigma"]
                )
            }
            reg = fmin(
                lambda x: loss_function(x["regularization"]),
                space=SPACE,
                trials=trls,
                max_evals=options["number_evaluations"],
                algo=tpe.suggest,
                show_progressbar=False,
            )
            if show_progress:
                plot_params_tried(SPACE, trls, dist_plots=True)
            return reg["regularization"]
            # return np.log(reg['regularization'])
    assert False, "Type of algorithm selected is incorrect"


def load_pickled_object(filepath):
    with open(filepath, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object


def dump_pickled_object(filepath, target_object):
    with open(filepath, "wb") as f:
        pickle.dump(target_object, f)


# Defines the type of maximum vertex dot product that we'll return.
def max_vertex(grad, activeVertex):
    # See which extreme point in the active set gives greater inner product.
    maxProd = activeVertex[0].T.dot(grad)
    maxInd = 0
    for i in range(len(activeVertex)):
        aux = activeVertex[i].T.dot(grad)
        if aux > maxProd:
            maxProd = aux
            maxInd = i
    return activeVertex[maxInd], maxInd


# Finds the step with the maximum and minimum inner product.
def max_min_vertex(grad, activeVertex):
    # See which extreme point in the active set gives greater inner product.
    maxProd = np.dot(activeVertex[0], grad)
    minProd = np.dot(activeVertex[0], grad)
    maxInd = 0
    minInd = 0
    for i in range(len(activeVertex)):
        if np.dot(activeVertex[i], grad) > maxProd:
            maxProd = np.dot(activeVertex[i], grad)
            maxInd = i
        else:
            if np.dot(activeVertex[i], grad) < minProd:
                minProd = np.dot(activeVertex[i], grad)
                minInd = i
    return activeVertex[maxInd], maxInd, activeVertex[minInd], minInd


def new_vertex_fail_fast(x, extremePoints):
    for i in range(len(extremePoints)):
        # Compare succesive indices.
        for j in range(len(extremePoints[i])):
            if extremePoints[i][j] != x[j]:
                break
        if j == len(extremePoints[i]) - 1:
            return False, i
    return True, np.nan


# Finds if x is already in the extremePoint list.
# Returns True if vertex is new, otherwise false and the index.
def check_new_vertex(vertex, active_set):
    for i in range(len(active_set)):
        if np.array_equal(active_set[i], vertex):
            return False, i
    return True, np.nan


# Deletes the extremepoint from the representation.
def delete_vertex_index(index, extremePoints, weights):
    del extremePoints[index]
    del weights[index]
    return


# Sort projection for the simplex.
def projection_simplex_sort(x, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = x.shape  # will raise ValueError if v is not 1-D
    if x.sum() == s and np.alltrue(x >= 0):
        return x
    v = x - np.max(x)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - s)) - 1
    theta = float(cssv[rho] - s) / (rho + 1)
    w = (v - theta).clip(min=0)
    return w


# Pick a stepsize.
def step_size(function, d, grad, x, typeStep="EL", maxStep=None):
    if typeStep == "Armijo":
        from scipy.optimize.linesearch import line_search_armijo

        alpha, fc, phi1 = line_search_armijo(
            function.f, x.flatten(), d.flatten(), grad.flatten(), function.f(x)
        )
        return alpha
    if typeStep == "SS":
        return -np.dot(grad, d) / (function.largest_eigenvalue() * np.dot(d, d))
    else:
        if typeStep == "GS":
            options = {"xatol": 1e-08, "maxiter": 500000, "disp": 0}

            def InnerFunction(t):  # Hidden from outer code
                return function.f(x + t * d)

            if maxStep is None:
                res = minimize_scalar(
                    InnerFunction, bounds=(0, 1), method="bounded", options=options
                )
            else:
                res = minimize_scalar(
                    InnerFunction,
                    bounds=(0, maxStep),
                    method="bounded",
                    options=options,
                )
            return res.x
        else:
            if maxStep is None:
                return function.line_search(grad, d, x, maxStep=1.0)
            else:
                return function.line_search(grad, d, x, maxStep=maxStep)


# Once the problem has been solved to a high accuracy, solve the problem.
def exportsolution(filepath, formatString, fOpt, xOpt, tolerance, size):
    with open(filepath, "wb") as f:
        np.savetxt(f, [np.array(formatString)], fmt="%s", delimiter=",")
        np.savetxt(f, np.array([fOpt]), fmt="%.15f")
        np.savetxt(f, [xOpt.T], fmt="%.11f", delimiter=",")
        np.savetxt(f, np.array([tolerance]), fmt="%.15f")
        np.savetxt(f, np.array([size]), fmt="%.15f")
    return


# Once the problem has been solved to a high accuracy, solve the problem.
def importSolution(filepath):
    with open(filepath) as f:
        _ = f.readline()
        fOpt = float(f.readline().rstrip())
        xOpt = np.asarray(f.readline().rstrip().split(",")).astype(float)
        tolerance = float(f.readline().rstrip())
        size = int(float(f.readline().rstrip()))
    return fOpt, xOpt, tolerance, size


def evaluate_polynomial_backup(X, polinomial):
    Psi = polinomial.fit_transform(X.T).T
    return Psi, Psi.shape[0]


def evaluate_polynomial(X, polinomial):
    Psi = polinomial.fit_transform(X.T).T
    return Psi


def polynomial_differentiation(u, x, deg=3, diff=1, width=5):

    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2 * width, diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n - width):

        points = np.arange(j - width, j + width)

        # Fit to a Chebyshev polynomial
        # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points], u[points], deg)

        # Take derivatives
        for d in range(1, diff + 1):
            du[j - width, d - 1] = poly.deriv(m=d)(x[j])
    return du


def polynomial_integration(u, x, deg=4, diff=1, width=5):

    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2 * width, diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n - width):

        points = np.arange(j - width, j + width)

        # Fit to a Chebyshev polynomial
        # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points], u[points], deg)

        # Create integral polynomial
        integral_poly = poly.integ()
        # Computes integral between x[j-1] and x[j]
        for d in range(1, diff + 1):
            du[j - width, d - 1] = integral_poly(x[j]) - integral_poly(x[j - 1])
    return np.cumsum(du)


def derivative(X, time, method="poly", deg=2, diff=1, width=5):
    if method == "poly":
        dim, num_samples = X.shape
        derivative = np.zeros((dim, int(num_samples - 2.0 * width)))
        if diff == 2:
            second_derivative = np.zeros((dim, int(num_samples - 2.0 * width)))
        for i in range(dim):
            if diff == 1:
                derivative[i] = polynomial_differentiation(
                    X[i], time, deg, diff, width
                ).squeeze()
            else:
                if diff == 2:
                    val = polynomial_differentiation(X[i], time, deg, diff, width)
                    derivative[i] = val[:, 0].squeeze()
                    second_derivative[i] = val[:, 1].squeeze()
                else:
                    assert (
                        False
                    ), "Have only built infrastructure to handle first and second derivative."
        if diff == 1:
            return derivative, X[:, width:-width], time[width:-width]
        if diff == 2:
            # print(len(time[width:-width]), len(time))
            # quit()
            return derivative, second_derivative, X[:, width:-width], time[width:-width]
    if method == "central":
        #        deriv = np.gradient(X, time, axis = 1)
        if diff == 1:
            return (
                (X[:, 2:] - X[:, :-2]) / (time[2:] - time[:-2]),
                X[:, 1:-1],
                time[1:-1],
            )
        if diff == 2:
            return (
                (X[:, 2:] - X[:, :-2]) / (time[2:] - time[:-2]),
                (X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2])
                / np.square(0.5 * (time[2:] - time[:-2])),
                X[:, 1:-1],
                time[1:-1],
            )
    if method == "forward":
        return np.diff(X, axis=1) / np.diff(time)[None, :], X[:, :-1], time[1:]


def derivative_from_list(list_X, list_time, method="poly", deg=2, diff=1, width=5):
    assert diff == 1 or diff == 2, "Order of differentiation not implemented."
    list_Y = []
    list_first_deriv = []
    list_X_updated = []
    list_time_updated = []
    for i in range(len(list_X)):
        if diff == 1:
            Y, X, t = derivative(
                list_X[i], list_time[i], method=method, deg=deg, diff=diff, width=width
            )
        if diff == 2:
            first_deriv, Y, X, t = derivative(
                list_X[i], list_time[i], method=method, deg=deg, diff=diff, width=width
            )
            list_first_deriv.append(first_deriv)
        list_Y.append(Y)
        list_X_updated.append(X)
        list_time_updated.append(t)
    if diff == 1:
        return list_Y, list_X_updated, list_time_updated
    else:
        return list_Y, list_first_deriv, list_X_updated, list_time_updated


def compute_integral_formulation_matrix(
    list_Psi,
    list_X,
    list_t,
    polinomial,
    type_of_integration="Simpsons",
    width=5,
    order=3,
):

    list_matrix = []
    list_delta = []
    for psi_val, x_val, t_val in zip(list_Psi, list_X, list_t):
        mat, x_values = cumulative_integration(
            psi_val,
            x_val,
            t_val,
            type_integration=type_of_integration,
            width=width,
            order=order,
        )
        list_matrix.append(mat)
        list_delta.append(x_values)
    return list_delta, list_matrix


def compute_exact_kuramoto_integral_formulation_matrix(
    list_X,
    list_t,
    intrinsic_frequencies,
    polinomial,
    num_basis_functions,
    number_of_samples=1000,
):
    from dynamics import kuramoto_time_individual
    from scipy import integrate

    list_exact = []
    for k in range(len(list_t)):
        matrix = np.zeros((num_basis_functions, int(len(list_t[k]) - 1)))
        # First interval.
        for l in range(1, len(list_t[k])):
            position, time_stamp = kuramoto_time_individual(
                intrinsic_frequencies,
                list_X[k][:, l - 1],
                list_t[k][l - 1],
                list_t[k][l],
                number_of_samples,
            )
            position_sine_cosines = np.vstack((np.cos(position), np.sin(position)))
            psi_evaluation = evaluate_polynomial(position_sine_cosines, polinomial)
            matrix[:, l - 1] = integrate.simps(psi_evaluation, time_stamp)
        list_exact.append(np.cumsum(matrix, axis=1))
    return list_exact


def compute_exact_FPUT_integral_formulation_matrix(
    list_X,
    list_t,
    exact_solution,
    polinomial,
    num_basis_functions,
    number_of_samples=1000,
):
    from dynamics import fermi_pasta_ulam_time_individual
    from scipy import integrate

    # Assume the initial velocities are zero
    list_exact = []
    for k in range(len(list_t)):
        matrix = np.zeros((num_basis_functions, int(len(list_t[k]) - 1)))
        # First interval.
        for l in range(1, len(list_t[k])):
            position, time_stamp = fermi_pasta_ulam_time_individual(
                exact_solution,
                polinomial,
                list_X[k][:, l - 1],
                list_t[k][l - 1],
                list_t[k][l],
                number_of_samples,
            )
            psi_evaluation = evaluate_polynomial(position, polinomial)
            matrix[:, l - 1] = integrate.simps(psi_evaluation, time_stamp)
        list_exact.append(np.cumsum(matrix, axis=1))
    return list_exact


def compute_exact_FPUT_integral_formulation_matrix_correct(
    list_X,
    list_first_deriv,
    list_t,
    exact_solution,
    polinomial,
    num_basis_functions,
    number_of_samples=1000,
):
    from dynamics import fermi_pasta_ulam_time_individual_correct
    from scipy import integrate

    list_exact = []
    for k in range(len(list_t)):
        matrix = np.zeros((num_basis_functions, int(len(list_t[k]) - 1)))
        for l in range(1, len(list_t[k])):
            (
                position,
                velocity_matrix,
                time_stamp,
            ) = fermi_pasta_ulam_time_individual_correct(
                exact_solution,
                polinomial,
                list_X[k][:, l - 1],
                list_first_deriv[k][:, l - 1],
                list_t[k][l - 1],
                list_t[k][l],
                number_of_samples,
            )
            psi_evaluation = evaluate_polynomial(position, polinomial)
            matrix[:, l - 1] = integrate.simps(psi_evaluation, time_stamp)
        list_exact.append(np.cumsum(matrix, axis=1))
    return list_exact


# Compute the cumulative integral of Psi in terms of t.
def cumulative_integration(Psi, x, t, type_integration="Simpsons", width=5, order=3):
    assert (
        type_integration == "Simpsons"
        or type_integration == "Trapezoid"
        or type_integration == "Poly"
    ), "Wrong integration rule."
    from scipy import integrate

    if type_integration == "Simpsons":
        x_output = (x - x[:, 0][:, np.newaxis])[:, 1:]
        num_basis_functions, number_of_samples = Psi.shape
        matrix = np.zeros((num_basis_functions, number_of_samples))
        for i in range(num_basis_functions):
            for j in range(0, number_of_samples):
                matrix[i, j] = integrate.simps(Psi[i, : j + 1], t[: j + 1])
        matrix = matrix[:, 1:]
    if type_integration == "Trapezoid":
        x_output = (x - x[:, 0][:, np.newaxis])[:, 1:]
        matrix = integrate.cumtrapz(Psi, t)
    if type_integration == "Poly":
        x_output = x[:, width:-width] - x[:, width - 1][:, np.newaxis]
        num_basis_functions, number_of_samples = Psi.shape
        matrix = np.zeros((num_basis_functions, int(number_of_samples - 2 * width)))
        for i in range(num_basis_functions):
            matrix[i, :] = polynomial_integration(Psi[i, :], t, deg=order, width=width)
    return matrix, x_output


def training_testing_validation_split(
    Psi, Y, proportion_train_data=0.7, proportion_testing_data=0.2
):
    from sklearn.model_selection import train_test_split

    # Split into training and auxiliary
    Psi_train, Psi_aux, Y_train, Y_aux = train_test_split(
        Psi.T, Y.T, test_size=1.0 - proportion_train_data
    )
    Psi_validation, Psi_test, Y_validation, Y_test = train_test_split(
        Psi_aux,
        Y_aux,
        test_size=1 - proportion_testing_data / (1.0 - proportion_train_data),
    )
    return (
        Psi_train.T,
        Y_train.T,
        Psi_validation.T,
        Y_validation.T,
        Psi_test.T,
        Y_test.T,
    )


# Given a dynamic, test how closely it resembles the testing data.
class testing_function:
    def __init__(
        self, Psi_test, Y_test, l0_penalty, exact_dynamic, normalization_factors
    ):
        self.Psi_test = Psi_test.copy()
        self.Y_test = Y_test.copy()
        self.l0_penalty = l0_penalty
        self.num_basis, self.num_samples = self.Psi_test.shape
        self.dimension, _ = Y_test.shape
        self.exact_dynamic = exact_dynamic.copy()
        self.normalization_factors = normalization_factors
        return

    def evaluate(self, dynamic):
        from scipy.sparse import isspmatrix_csr

        if isspmatrix_csr(dynamic):
            penalty = self.l0_penalty * dynamic.count_nonzero()
        else:
            penalty = self.l0_penalty * np.count_nonzero(dynamic)
        if dynamic.shape != self.exact_dynamic.shape:
            aux = dynamic.reshape(self.dimension, self.num_basis)
            return np.linalg.norm(aux.dot(self.Psi_test) - self.Y_test) + penalty
        else:
            return np.linalg.norm(dynamic.dot(self.Psi_test) - self.Y_test) + penalty

    def compare_exact_backup(self, dynamic):
        if dynamic.shape != self.exact_dynamic.shape:
            return np.linalg.norm(
                self.exact_dynamic - dynamic.reshape(self.exact_dynamic.shape)
            )
        else:
            return np.linalg.norm(self.exact_dynamic - dynamic)

    def compare_exact(self, dynamic):
        unnormalized = (
            dynamic.reshape(self.exact_dynamic.shape).T / self.normalization_factors
        ).T
        return np.linalg.norm(self.exact_dynamic - unnormalized)

    def return_Psi(self):
        return self.Psi_test

    def return_Y(self):
        return self.Y_test

# Given a dynamic, test how closely it resembles the testing data.
class testing_function_simplex:
    def __init__(
        self, Psi_test, Y_test, l0_penalty, exact_dynamic, normalization_factors
    ):
        self.Psi_test = Psi_test.copy()
        self.Y_test = Y_test.copy()
        self.l0_penalty = l0_penalty
        self.num_basis, self.num_samples = self.Psi_test.shape
        self.dimension, _ = Y_test.shape
        self.exact_dynamic = exact_dynamic.copy()
        self.normalization_factors = normalization_factors
        return

    def evaluate(self, dynamic):
        from scipy.sparse import isspmatrix_csr

        dynamic = dynamic[:len(dynamic)//2] - dynamic[len(dynamic)//2:]

        if isspmatrix_csr(dynamic):
            penalty = self.l0_penalty * dynamic.count_nonzero()
        else:
            penalty = self.l0_penalty * np.count_nonzero(dynamic)
        if dynamic.shape != self.exact_dynamic.shape:
            aux = dynamic.reshape(self.dimension, self.num_basis)
            return np.linalg.norm(aux.dot(self.Psi_test) - self.Y_test) + penalty
        else:
            return np.linalg.norm(dynamic.dot(self.Psi_test) - self.Y_test) + penalty

    def compare_exact_backup(self, dynamic):
        dynamic = dynamic[:len(dynamic)//2] - dynamic[len(dynamic)//2:]
        
        if dynamic.shape != self.exact_dynamic.shape:
            return np.linalg.norm(
                self.exact_dynamic - dynamic.reshape(self.exact_dynamic.shape)
            )
        else:
            return np.linalg.norm(self.exact_dynamic - dynamic)

    def compare_exact(self, dynamic):
        dynamic = dynamic[:len(dynamic)//2] - dynamic[len(dynamic)//2:]
        unnormalized = (
            dynamic.reshape(self.exact_dynamic.shape).T / self.normalization_factors
        ).T
        return np.linalg.norm(self.exact_dynamic - unnormalized)

    def return_Psi(self):
        return self.Psi_test

    def return_Y(self):
        return self.Y_test

def polish_solution(
    Psi,
    y,
    active_set,
    barycentric_coordinates,
    tolerance=1.0e-10,
    threshold=0.0,
    type_criterion="FW",
    time_limit=60.0,
    max_steps=100,
):
    num_basis_functions = Psi.shape[0]
    dimension = y.shape[0]

    from scipy.sparse import csr_matrix
    from scipy.sparse import vstack
    matrix = csr_matrix(
        active_set[0].reshape((dimension, num_basis_functions)).dot(Psi).flatten()
    )
    for i in range(1, len(active_set)):
        matrix = vstack(
            (
                matrix,
                csr_matrix(
                    active_set[i]
                    .reshape((dimension, num_basis_functions))
                    .dot(Psi)
                    .flatten()
                ),
            )
        )

    quadratic = matrix.dot(matrix.T)
    linear = -matrix.dot(y.flatten())
    
    # print(quadratic)
    # print(linear)

    
    # Create objective function and feasible region.
    from feasible_regions import probability_simplex
    from functions import solution_polishing
    feas_reg = probability_simplex(len(active_set))
    fun_polishing = solution_polishing(quadratic, linear)
    from CINDy_algorithm import accelerated_projected_gradient_descent

    (
        x,
        polished_barycentric_coordinates,
        gap_values1,
    ) = accelerated_projected_gradient_descent(
        fun_polishing,
        feas_reg,
        active_set,
        tolerance,
        barycentric_coordinates,
        time_limit=time_limit,
        type_criterion=type_criterion,
        max_iteration=max_steps,
    )
    # return x, active_set, polished_barycentric_coordinates
    return remove_vertives(active_set, polished_barycentric_coordinates, threshold)


def remove_vertives(active_set, barycentric_coordinates, threshold):
    new_active_set = []
    new_barycentric_coordinates = []
    # print(barycentric_coordinates)
    for i in range(len(active_set)):
        if barycentric_coordinates[i] > threshold:
            new_active_set.append(active_set[i])
            new_barycentric_coordinates.append(barycentric_coordinates[i])
    aux = sum(new_barycentric_coordinates)
    new_barycentric_coordinates = [
        x + (1.0 - aux) / len(new_barycentric_coordinates)
        for x in new_barycentric_coordinates
    ]
    x = np.zeros(active_set[0].shape)
    for i in range(len(new_active_set)):
        x += new_barycentric_coordinates[i] * new_active_set[i]
    # print(x[np.where(x != 0.0)], np.where(x != 0.0), new_barycentric_coordinates)
    
    return x, new_active_set, new_barycentric_coordinates


# Given a dynamic, test how closely it resembles the testing data.
class normalization_and_recovery:
    def __init__(self, X):
        self.normalization_factors = np.linalg.norm(X, axis=1)[:, None]
        self.num_basis_functions = X.shape[0]
        return

    # Divide each collection of samples by the norm of the samples.
    def normalize(self, X):
        return X / self.normalization_factors

    def unnormalize(self, X):
        return X * self.normalization_factors

    def return_normalization_factors(self):
        return self.normalization_factors

    # Divide each collection of samples by the norm of the samples.
    def recover_solution(self, Sol):
        if Sol.ndim == 2 and Sol.shape[1] == self.num_basis_functions:
            return (Sol.T / self.normalization_factors).T
        else:
            dimension = int(Sol.shape[0] / self.num_basis_functions)
            return (
                Sol.reshape(dimension, self.num_basis_functions).T
                / self.normalization_factors
            ).T

    # Divide each collection of samples by the norm of the samples.
    def unnormalize_solution(self, Sol):
        if Sol.ndim == 2 and Sol.shape[1] == self.num_basis_functions:
            return (Sol.T * self.normalization_factors).T
        else:
            dimension = int(Sol.shape[0] / self.num_basis_functions)
            return (
                Sol.reshape(dimension, self.num_basis_functions).T
                * self.normalization_factors
            ).T


# Divide each collection of samples by the norm of the samples.
def normalize_data(X):
    return X / np.linalg.norm(X, axis=1)[:, None]


# Divide each collection of samples by the norm of the samples.
def transform_solution(Sol, X):
    return (Sol.T * np.linalg.norm(X, axis=1)[:, None]).T


def repeat_along_diag(a, r):
    m, n = a.shape
    out = np.zeros((r, m, r, n), dtype=a.dtype)
    diag = np.einsum("ijik->ijk", out)
    diag[:] = a
    return out.reshape(-1, n * r)
