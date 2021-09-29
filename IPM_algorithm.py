import numpy as np
from auxiliary_functions import regularization_selection
from cvxopt import matrix, sparse, solvers


def Train_QP_cvxopt(
    function,
    testing_function,
    equality_constraints=None,
    inequality_constraints=None,
    show_progress=False,
    type_parameter_search="minimization",
    maximum_number_iterations_search=100,
    maximum_number_iterations_final=100,
    options={
        "min": 0.0,
        "max": 1.0,
        "number_evaluations": 200,
        "distribution": "uniform",
        "inner_iterations": 5000,
    },
    absolute_tolerance=1.0e-7,
    relative_tolerance=1.0e-6,
    feasibility_tolernace=1.0e-7,
):
    from auxiliary_functions import repeat_along_diag

    num_basis_functions = function.number_basis_functions()
    dimension = function.number_dimensions()

    reference_point = np.random.rand(int(dimension * num_basis_functions))
    gradient_val = function.gradient(reference_point)
    hessian_matrix = repeat_along_diag(function.hessian(), dimension)

    # Get the quadratic and the linear part of the "flattened" problem (i.e. the problem in vector form)
    quadratic = hessian_matrix
    linear = (
        gradient_val
        - 0.5 * reference_point.dot(quadratic)
        - 0.5 * quadratic.dot(reference_point)
    )

    # Transform to a problem over the simplex.
    simplex_quadratic = np.block([[quadratic, -quadratic], [-quadratic, quadratic]])
    simplex_linear = np.hstack((linear, -linear))

    simplex_dimensionality = 2 * int(dimension * num_basis_functions)

    if equality_constraints is not None:
        # Create the constraint matrix.
        num_constraints = len(equality_constraints)
        constraint_matrix = np.zeros((num_constraints, simplex_dimensionality))
        constraint_vector = np.zeros(num_constraints)
        for i in range(num_constraints):
            for key, value in equality_constraints[i].items():
                if key != "constant":
                    index = int(key.replace("x", ""))
                    constraint_matrix[i, index] = value
                    constraint_matrix[
                        i, index + int(num_basis_functions * dimension)
                    ] = -value
                else:
                    constraint_vector[i] = value
        A = sparse(
            matrix(
                np.vstack((constraint_matrix, np.ones(simplex_dimensionality))),
                (1 + num_constraints, simplex_dimensionality),
            )
        )
    else:
        A = matrix(np.ones(simplex_dimensionality), (1, simplex_dimensionality))
        constraint_vector = None

    if inequality_constraints is not None:
        # Create the inequality constraint matrix.
        num_ineq_constraints = len(inequality_constraints)
        ineq_constraint_matrix = np.zeros(
            (num_ineq_constraints, simplex_dimensionality)
        )
        ineq_constraint_vector = np.zeros(num_ineq_constraints)
        for i in range(num_ineq_constraints):
            for key, value in inequality_constraints[i].items():
                if key != "constant":
                    index = int(key.replace("x", ""))
                    ineq_constraint_matrix[i, index] = value
                    ineq_constraint_matrix[
                        i, index + int(num_basis_functions * dimension)
                    ] = -value
                else:
                    ineq_constraint_vector[i] = value
        G = sparse(
            matrix(
                np.vstack(
                    (ineq_constraint_matrix, -np.identity(simplex_dimensionality))
                ),
                (simplex_dimensionality + num_ineq_constraints, simplex_dimensionality),
            )
        )
        h = matrix(
            np.append(ineq_constraint_vector, np.zeros(simplex_dimensionality)),
            (simplex_dimensionality + num_ineq_constraints, 1),
        )
    else:
        G = -sparse(
            matrix(
                np.identity(simplex_dimensionality),
            )
        )
        h = matrix(np.zeros(simplex_dimensionality), (simplex_dimensionality, 1))

    # Create the matrices that will be used in the solution process
    P = sparse(matrix(simplex_quadratic))
    q = matrix(simplex_linear, (simplex_dimensionality, 1))

    # Select the best hyperparameters
    solvers.options["show_progress"] = show_progress
    solvers.options["absolute_tolerance"] = absolute_tolerance
    solvers.options["relative_tolerance"] = relative_tolerance
    solvers.options["feasibility_tolernace"] = feasibility_tolernace
    alpha = regularization_selection(
        lambda x: testing_function.evaluate(
            QP_cvxopt(P, q, G, h, A, x, constraint_vector=constraint_vector)
        ),
        type_parameter_search,
        options,
        show_progress=show_progress,
        skip_minimization_check=True,
    )
    print("Regularization choosen (CVXOPT): ", alpha)
    return QP_cvxopt(P, q, G, h, A, alpha, constraint_vector=constraint_vector)


def QP_cvxopt(P, q, G, h, A, l1_regularization, constraint_vector=None):
    if constraint_vector is None:
        b = matrix(l1_regularization)
    else:
        b = matrix(
            np.append(constraint_vector, l1_regularization).tolist(),
            (1 + len(constraint_vector), 1),
            "d",
        )
    sol = solvers.qp(P, q, G, h, A, b)
    optimum = np.array(sol["x"])
    return (
        optimum[: int(len(optimum) / 2)] - optimum[int(len(optimum) / 2) :]
    ).flatten()
