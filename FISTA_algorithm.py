import numpy as np
from auxiliary_functions import regularization_selection


def FISTA(function, x0, regularization, maxit):
    t = 1
    x = x0.copy()
    z = x0.copy()
    L = function.largest_eigenvalue()
    for iter_num in range(maxit):
        xold = x.copy()
        grad = function.gradient(z)
        x = function.proximal_operator(z - grad / L, regularization / L)
        t0 = t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        z = x + ((t0 - 1.0) / t) * (x - xold)
    return x


def Train_LASSO(
    training_function,
    testing_function,
    maxit=50000,
    type_parameter_search="minimization",
    options={
        "min": 0.0,
        "max": 1.0,
        "mean": 0.0,
        "sigma": 2.0,
        "number_evaluations": 200,
        "distribution": "uniform",
        "inner_iterations": 5000,
    },
    show_progress=False,
):
    psi = training_function.psi_val()
    y = training_function.y_val()
    starting_point = y.dot(np.linalg.pinv(psi)).flatten()
    alpha = regularization_selection(
        lambda x: testing_function.evaluate(
            FISTA(training_function, starting_point, x, options["inner_iterations"])
        ),
        type_parameter_search,
        options,
        show_progress=show_progress,
    )
    return FISTA(training_function, starting_point, alpha, maxit)
