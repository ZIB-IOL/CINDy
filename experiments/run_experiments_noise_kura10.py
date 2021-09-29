# General imports
import numpy as np
import os, sys, time, datetime

sys.path.append("..")

from dynamics import exact_solution_kuramoto

from functions import quadratic_function_fast

import matplotlib.pyplot as plt

# Input all the parameters to the file.
TIME_LIMIT_INNER = 1600

# TIME_LIMIT_INNER= 20
tolerance_outer = 1.0e-6
tolerance_inner = 1.0e-7
number_of_oscillators = 10
number_of_samples = 6000
number_of_experiments = 40
max_degree_polynomial = 2
num_repetitions = 20
noise_parameter = np.power(10, np.linspace(-8, -2, 13)).tolist()
radius_multiplier = 1.5
num_repetitions = 1
num_sets = 20

order_polynomial_integration = 8
order_polynomial_differentiation = 8
type_differentiation = "poly"
type_integration = "Poly"
width_derivative = 5

# Parameters used in the dynamic
beta = 0.7
K = 2.0
forcing_param = 0.2

# Create the polinomial basis functions
from sklearn.preprocessing import PolynomialFeatures

polinomial = PolynomialFeatures(max_degree_polynomial)
# Generate all possible polinomials given number_of_oscillators points.
polinomial.fit_transform(np.ones((1, int(2.0 * number_of_oscillators))))

print("List of all the polynomials used as basis functions")
print(polinomial.get_feature_names())
print("Number of basis functions: " + str(len(polinomial.get_feature_names())))

intrinsic_frequencies = np.random.rand(number_of_oscillators)
# Create the exact solution to the dynamic given the polynomials above
exact_solution = exact_solution_kuramoto(
    number_of_oscillators,
    intrinsic_frequencies,
    polinomial,
    K=K,
    forcing_param=forcing_param,
)
print("Exact matrix that represents the dynamic")
print(exact_solution)
print(
    "Number of non-zero elements in the matrix: "
    + str(np.count_nonzero(exact_solution))
)
print(
    "Total size of the matrix: "
    + str(int(exact_solution.shape[0] * exact_solution.shape[1]))
)


# Compute the solution using the STRidge algorithm.
from SINDy_algorithm import TrainSTRidge

# Generate the exact solution to the problem, according to the previously defined basis.
from auxiliary_functions import (
    normalization_and_recovery,
    training_testing_validation_split,
    evaluate_polynomial,
)

from dynamics import kuramoto_time

for _ in range(num_sets):
    SINDy_dynamic = []
    SINDy_integral_dynamic = []
    FISTA_dynamic = []
    FISTA_integral_dynamic = []
    BCG_dynamic = []
    BCG_constraint_dynamic = []
    BCG_integral_dynamic = []
    BCG_integral_constraint_dynamic = []
    CVXOPT_dynamic = []
    CVXOPT_constraint_dynamic = []
    CVXOPT_integral_dynamic = []
    CVXOPT_integral_constraint_dynamic = []
    SR3_constraint_l0_dynamic = []
    SR3_constraint_l1_dynamic = []
    SR3_l0_dynamic = []
    SR3_l1_dynamic = []
    SR3_constraint_l0_integral_dynamic = []
    SR3_constraint_l1_integral_dynamic = []
    SR3_l0_integral_dynamic = []
    SR3_l1_integral_dynamic = []
    psi_train_data = []
    psi_test_data = []
    psi_validation_data = []
    Y_train_data = []
    Y_test_data = []
    Y_validation_data = []
    matrix_train_data = []
    matrix_test_data = []
    matrix_validation_data = []
    delta_train_data = []
    delta_test_data = []
    delta_validation_data = []

    # Noise and regularization
    noise = np.zeros((len(noise_parameter), num_repetitions))
    first_derivative_norm = np.zeros((len(noise_parameter), num_repetitions))
    delta_norm = np.zeros((len(noise_parameter), num_repetitions))

    for i in range(num_repetitions):
        for j in range(len(noise_parameter)):

            # Save the noise parameter
            noise[j, i] = noise_parameter[j]

            # theta_init = np.random.rand(number_of_oscillators)
            max_time = 10.0
            # max_time = 2.0
            list_X, list_Y, list_t = kuramoto_time(
                intrinsic_frequencies,
                max_time,
                int(number_of_samples / number_of_experiments),
                number_of_experiments=number_of_experiments,
                K=2.0,
                forcing_param=0.2,
            )
            Y = np.hstack(list_Y)

            width_derivative = 5
            # Corrupt the data with a little bit of noise.
            list_X_noisy = [
                x
                + np.random.normal(
                    scale=noise_parameter[j] * np.sqrt(np.var(x, axis=1)),
                    size=x.T.shape,
                ).T
                for x in list_X
            ]

            # Infer the derivatives from the noisy data.
            from auxiliary_functions import derivative_from_list

            # Y_list_cropped, X_noisy_cropped, t_cropped  = derivative_from_list(list_X_noisy, list_t, method="poly", deg=3, width = width_derivative)
            # Y_list_cropped, X_noisy_cropped, t_cropped = derivative_from_list(
            #     list_X_noisy, list_t, method="poly"
            # )

            Y_list_cropped, X_noisy_cropped, t_cropped = derivative_from_list(
                list_X_noisy,
                list_t,
                method=type_differentiation,
                deg=order_polynomial_differentiation,
                diff=1,
                width=width_derivative,
            )

            # In case we want to compare to the performance with clean derivatives.
            width = int((list_Y[0].shape[1] - Y_list_cropped[0].shape[1]) / 2.0)
            Y_clean_cropped = np.hstack([x[:, width:-width] for x in list_Y])

            # Aggregate the data into a single numpy array.
            X_noisy_cropped = np.hstack(X_noisy_cropped)
            Y_noisy_cropped = np.hstack(Y_list_cropped)
            X_noisy_cropped_transformed = np.vstack(
                (np.cos(X_noisy_cropped), np.sin(X_noisy_cropped))
            )

            # Evaluate the polynomials
            Psi_noisy_cropped_transformed = evaluate_polynomial(
                X_noisy_cropped_transformed, polinomial
            )
            num_basis_functions = Psi_noisy_cropped_transformed.shape[0]

            # Compute the matrices for the integral approach.
            # Create the list of psi.
            list_Psi = [
                evaluate_polynomial(np.vstack((np.cos(x), np.sin(x))), polinomial)
                for x in list_X_noisy
            ]

            from auxiliary_functions import compute_integral_formulation_matrix

            list_delta, list_matrix = compute_integral_formulation_matrix(
                list_Psi,
                list_X_noisy,
                list_t,
                polinomial,
                type_of_integration=type_integration,
                width=width,
                order=order_polynomial_integration,
            )

            if type_integration == "Poly":
                matrix = np.hstack(
                    [
                        np.divide(matrix, time[1 : int(-2.0 * width + 1)])
                        for matrix, time in zip(list_matrix, list_t)
                    ]
                )
                delta = np.hstack(
                    [
                        np.divide(delta, time[1 : int(-2.0 * width + 1)])
                        for delta, time in zip(list_delta, list_t)
                    ]
                )
            else:
                matrix = np.hstack(
                    [
                        np.divide(matrix, time[1:])
                        for matrix, time in zip(list_matrix, list_t)
                    ]
                )
                delta = np.hstack(
                    [
                        np.divide(delta, time[1:])
                        for delta, time in zip(list_delta, list_t)
                    ]
                )

            # Split into testing and training data for the CG approach
            (
                Psi_train,
                Y_train,
                Psi_validation,
                Y_validation,
                Psi_test,
                Y_test,
            ) = training_testing_validation_split(
                Psi_noisy_cropped_transformed,
                Y_noisy_cropped,
                proportion_train_data=0.7,
                proportion_testing_data=0.2,
            )
            (
                matrix_train,
                delta_train,
                matrix_validation,
                delta_validation,
                matrix_test,
                delta_test,
            ) = training_testing_validation_split(
                matrix,
                delta,
                proportion_train_data=0.7,
                proportion_testing_data=0.2,
            )

            formulation_matrices = {
                "Psi_validation": Psi_validation,
                "matrix_validation": matrix_validation,
                "Psi_train": Psi_train,
                "matrix_train": matrix_train,
                "Y_train": Y_train,
                "delta_train": delta_train,
            }

            # Save the data so that we can output the results later.
            psi_train_data.append(Psi_train)
            psi_test_data.append(Psi_test)
            psi_validation_data.append(Psi_validation)
            Y_train_data.append(Y_train)
            Y_test_data.append(Y_test)
            Y_validation_data.append(Y_validation)
            matrix_train_data.append(matrix_train)
            matrix_test_data.append(matrix_test)
            matrix_validation_data.append(matrix_validation)
            delta_train_data.append(delta_train)
            delta_test_data.append(delta_test)
            delta_validation_data.append(delta_validation)

            # Normalize the data afterwards. Differential approach.
            normalize_data_class = normalization_and_recovery(Psi_train)
            Psi_train = normalize_data_class.normalize(Psi_train)
            Psi_test = normalize_data_class.normalize(Psi_test)
            normalization_factors = normalize_data_class.return_normalization_factors()
            radius_differential = radius_multiplier * np.sum(
                np.abs(normalize_data_class.unnormalize_solution(exact_solution))
            )

            # Normalize the data afterwards. Integral approach.
            normalize_data_class_integral = normalization_and_recovery(matrix_train)
            matrix_train = normalize_data_class_integral.normalize(matrix_train)
            matrix_test = normalize_data_class_integral.normalize(matrix_test)
            normalization_factors_integral = (
                normalize_data_class_integral.return_normalization_factors()
            )
            radius_integral = radius_multiplier * np.sum(
                np.abs(
                    normalize_data_class_integral.unnormalize_solution(exact_solution)
                )
            )

            # Create the data for the SINDy approach (get rid of all the cos^2 terms for example.)
            indices_to_eliminate = []
            feature_names = polinomial.get_feature_names()
            for l in range(number_of_oscillators):
                indices_to_eliminate.append(feature_names.index("x" + str(l) + "^2"))
                indices_to_eliminate.append(
                    feature_names.index("x" + str(l + number_of_oscillators) + "^2")
                )
            indices_to_use = [
                x
                for x in list(np.arange(num_basis_functions))
                if x not in indices_to_eliminate
            ]

            print("Kura10 Running SINDy constrained.")
            # Compute the solution using the STRidge algorithm.
            sol_STRidge = np.zeros((number_of_oscillators, num_basis_functions))
            for k in range(number_of_oscillators):
                sol_STRidge[k, indices_to_use] = TrainSTRidge(
                    Psi_train.T[:, indices_to_use],
                    Y_train[k].reshape((Psi_train.shape[1], 1)),
                    Psi_test.T[:, indices_to_use],
                    Y_test[k].reshape((Psi_test.shape[1], 1)),
                    0.0,
                    0.01,
                ).squeeze()
            sol_STRidge = normalize_data_class.recover_solution(sol_STRidge)
            SINDy_dynamic.append(sol_STRidge)
            from auxiliary_functions import display_statistics

            display_statistics(
                sol_STRidge, exact_solution, noise_parameter[j], formulation_matrices
            )
            # print("Derivative training and validation error: ", np.linalg.norm(Y_validation - sol_STRidge.dot(Psi_validation)), np.linalg.norm(Y_train_data[-1] - sol_STRidge.dot(psi_train_data[-1])))

            size = int(number_of_oscillators * num_basis_functions)
            from auxiliary_functions import testing_function

            # Normal approach
            testing_fun = testing_function(
                Psi_test, Y_test, 0.0, exact_solution, normalization_factors
            )
            fun = quadratic_function_fast(Psi_train, Y_train, alpha=0.0)

            print("Running FISTA - LASSO.")
            from FISTA_algorithm import Train_LASSO
            from functions import quadratic_LASSO

            x_LASSO = np.zeros((number_of_oscillators, num_basis_functions))
            fun_LASSO = quadratic_LASSO(Psi_train[indices_to_use, :], Y_train)
            testing_fun_LASSO = testing_function(
                Psi_test[indices_to_use, :],
                Y_test,
                0.0,
                exact_solution[:, indices_to_use],
                normalization_factors[indices_to_use],
            )
            options_LASSO = {
                "min": -20.0,
                "max": -3.0,
                "number_evaluations": 100,
                "distribution": "log-uniform",
                "inner_iterations": 2000,
                "xatol": 1e-07,
            }
            time_ref = time.time()
            x_LASSO[:, indices_to_use] = Train_LASSO(
                fun_LASSO,
                testing_fun_LASSO,
                30000,
                type_parameter_search="bayesian",
                options=options_LASSO,
                show_progress=True,
            ).reshape((number_of_oscillators, len(indices_to_use)))
            x_LASSO = normalize_data_class.recover_solution(x_LASSO)
            FISTA_dynamic.append(x_LASSO)
            display_statistics(
                x_LASSO, exact_solution, noise_parameter[j], formulation_matrices
            )
            print("Time to compute: " + str(time.time() - time_ref))

            from CINDy_algorithm import CINDy

            from dynamics import (
                add_constraints_symmetry_kuramoto,
                add_constraints_symmetry_kuramoto_simple,
            )
            from feasible_regions import l1_ball_pyscipopt

            equality_constraints = add_constraints_symmetry_kuramoto(
                polinomial, number_of_oscillators, normalization_factors
            )

            feasibleRegion = l1_ball_pyscipopt(
                size,
                radius_differential,
                linear_equality_constraints=equality_constraints,
            )

            print("Running FCFW differential constrained.")
            (
                x,
                testing_loss,
                real_loss,
                timing,
                true_losses,
                testing_losses,
                timing,
                gap_values,
            ) = CINDy(
                fun,
                feasibleRegion,
                tolerance_outer,
                tolerance_inner,
                TIME_LIMIT_INNER,
                testing_fun,
                primal_improvement=1.0e-9,
                type_criterion="BCG",
            )
            # Output the results for FCFW
            x = x.reshape(exact_solution.shape)
            x = normalize_data_class.recover_solution(x)
            BCG_constraint_dynamic.append(x)
            display_statistics(
                x, exact_solution, noise_parameter[j], formulation_matrices
            )

            from IPM_algorithm import Train_QP_cvxopt

            print("Running QP differential constrained")
            options_CVXOPT = {
                "min": -1.0,
                "max": 13.0,
                "number_evaluations": 100,
                "distribution": "log-uniform",
                "xatol": 1e-07,
            }
            x_QP_CVXOPT_constraint = Train_QP_cvxopt(
                fun,
                testing_fun,
                equality_constraints=equality_constraints,
                show_progress=False,
                type_parameter_search="bayesian",
                options=options_CVXOPT,
            )
            x_QP_CVXOPT_constraint = x_QP_CVXOPT_constraint.reshape(
                exact_solution.shape
            )
            x_QP_CVXOPT_constraint[:, indices_to_eliminate] = 0.0
            x_QP_CVXOPT_constraint = normalize_data_class.recover_solution(
                x_QP_CVXOPT_constraint
            )
            CVXOPT_constraint_dynamic.append(x_QP_CVXOPT_constraint)
            display_statistics(
                x_QP_CVXOPT_constraint,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            equality_constraints_simple = add_constraints_symmetry_kuramoto_simple(
                polinomial, number_of_oscillators, normalization_factors
            )

            feasibleRegion = l1_ball_pyscipopt(
                size,
                radius_differential,
                linear_equality_constraints=equality_constraints_simple,
            )

            print("Running FCFW differential.")
            (
                x,
                testing_loss,
                real_loss,
                timing,
                true_losses,
                testing_losses,
                timing,
                gap_values,
            ) = CINDy(
                fun,
                feasibleRegion,
                tolerance_outer,
                tolerance_inner,
                TIME_LIMIT_INNER,
                testing_fun,
                primal_improvement=1.0e-9,
                type_criterion="BCG",
            )
            # Output the results for FCFW
            x = x.reshape(exact_solution.shape)
            x = normalize_data_class.recover_solution(x)
            BCG_dynamic.append(x)
            display_statistics(
                x, exact_solution, noise_parameter[j], formulation_matrices
            )

            print("Running QP differential")
            x_QP_CVXOPT = Train_QP_cvxopt(
                fun,
                testing_fun,
                equality_constraints=equality_constraints_simple,
                show_progress=False,
                type_parameter_search="bayesian",
                options=options_CVXOPT,
            )
            x_QP_CVXOPT = x_QP_CVXOPT.reshape(exact_solution.shape)
            x_QP_CVXOPT[:, indices_to_eliminate] = 0.0
            x_QP_CVXOPT = normalize_data_class.recover_solution(x_QP_CVXOPT)
            CVXOPT_dynamic.append(x_QP_CVXOPT)
            display_statistics(
                x_QP_CVXOPT, exact_solution, noise_parameter[j], formulation_matrices
            )

            testing_fun_integral = testing_function(
                matrix_test,
                delta_test,
                0.0,
                exact_solution,
                normalization_factors_integral,
            )
            fun_integral = quadratic_function_fast(matrix_train, delta_train, alpha=0.0)

            print("Running Integral SINDy.")
            # Compute the solution using the STRidge algorithm.
            sol_STRidge_integral = np.zeros(
                (number_of_oscillators, num_basis_functions)
            )
            for k in range(number_of_oscillators):
                sol_STRidge_integral[k, indices_to_use] = TrainSTRidge(
                    matrix_train.T[:, indices_to_use],
                    delta_train[k].reshape((matrix_train.shape[1], 1)),
                    matrix_test.T[:, indices_to_use],
                    delta_test[k].reshape((matrix_test.shape[1], 1)),
                    0.0,
                    1.0e-6,
                ).squeeze()
            sol_STRidge_integral = normalize_data_class_integral.recover_solution(
                sol_STRidge_integral
            )
            SINDy_integral_dynamic.append(sol_STRidge_integral)
            display_statistics(
                sol_STRidge_integral,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            print("Running FISTA - LASSO integral.")
            x_LASSO_integral = np.zeros((number_of_oscillators, num_basis_functions))
            fun_LASSO_integral = quadratic_LASSO(
                matrix_train[indices_to_use, :], delta_train
            )
            testing_fun_LASSO_integral = testing_function(
                matrix_test[indices_to_use, :],
                delta_test,
                0.0,
                exact_solution[:, indices_to_use],
                normalization_factors_integral[indices_to_use],
            )
            time_ref = time.time()
            x_LASSO_integral[:, indices_to_use] = Train_LASSO(
                fun_LASSO_integral,
                testing_fun_LASSO_integral,
                30000,
                type_parameter_search="bayesian",
                options=options_LASSO,
                show_progress=True,
            ).reshape((number_of_oscillators, len(indices_to_use)))
            x_LASSO_integral = normalize_data_class_integral.recover_solution(
                x_LASSO_integral
            )
            FISTA_integral_dynamic.append(x_LASSO_integral)
            display_statistics(
                x_LASSO_integral,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )
            print("Time to compute: " + str(time.time() - time_ref))

            equality_constraints_integral = add_constraints_symmetry_kuramoto(
                polinomial, number_of_oscillators, normalization_factors_integral
            )
            feasibleRegion_integral = l1_ball_pyscipopt(
                size,
                radius_integral,
                linear_equality_constraints=equality_constraints_integral,
            )
            print("Running FCFW integral constrained.")
            (
                x_integral,
                testing_loss_integral,
                real_loss_integral,
                timing,
                true_losses,
                testing_losses,
                timing,
                gap_values,
            ) = CINDy(
                fun_integral,
                feasibleRegion_integral,
                tolerance_outer,
                tolerance_inner,
                TIME_LIMIT_INNER,
                testing_fun_integral,
                primal_improvement=1.0e-9,
                type_criterion="BCG",
            )
            # Output the results for FCFW
            x_integral = x_integral.reshape(exact_solution.shape)
            x_integral = normalize_data_class_integral.recover_solution(x_integral)
            BCG_integral_constraint_dynamic.append(x_integral)
            display_statistics(
                x_integral, exact_solution, noise_parameter[j], formulation_matrices
            )

            print("Running QP integral constrained")
            x_QP_CVXOPT_integral_constraint = Train_QP_cvxopt(
                fun_integral,
                testing_fun_integral,
                equality_constraints=equality_constraints_integral,
                show_progress=False,
                type_parameter_search="bayesian",
                options=options_CVXOPT,
            )
            x_QP_CVXOPT_integral_constraint = x_QP_CVXOPT_integral_constraint.reshape(
                exact_solution.shape
            )
            x_QP_CVXOPT_integral_constraint[:, indices_to_eliminate] = 0.0
            x_QP_CVXOPT_integral_constraint = (
                normalize_data_class_integral.recover_solution(
                    x_QP_CVXOPT_integral_constraint
                )
            )
            CVXOPT_integral_constraint_dynamic.append(x_QP_CVXOPT_integral_constraint)
            display_statistics(
                x_QP_CVXOPT_integral_constraint,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            equality_constraints_integral_simple = (
                add_constraints_symmetry_kuramoto_simple(
                    polinomial, number_of_oscillators, normalization_factors_integral
                )
            )
            feasibleRegion_integral = l1_ball_pyscipopt(
                size,
                radius_integral,
                linear_equality_constraints=equality_constraints_integral_simple,
            )

            print("Running FCFW integral.")
            (
                x_integral,
                testing_loss_integral,
                real_loss_integral,
                timing,
                true_losses,
                testing_losses,
                timing,
                gap_values,
            ) = CINDy(
                fun_integral,
                feasibleRegion_integral,
                tolerance_outer,
                tolerance_inner,
                int(2.0 * TIME_LIMIT_INNER),
                testing_fun_integral,
                primal_improvement=1.0e-9,
                type_criterion="BCG",
            )
            # Output the results for FCFW
            x_integral = x_integral.reshape(exact_solution.shape)
            x_integral = normalize_data_class_integral.recover_solution(x_integral)
            BCG_integral_dynamic.append(x_integral)
            display_statistics(
                x_integral, exact_solution, noise_parameter[j], formulation_matrices
            )

            print("Running QP integral")
            x_QP_CVXOPT_integral = Train_QP_cvxopt(
                fun_integral,
                testing_fun_integral,
                equality_constraints=equality_constraints_integral_simple,
                show_progress=False,
                type_parameter_search="bayesian",
                options=options_CVXOPT,
            )
            x_QP_CVXOPT_integral = x_QP_CVXOPT_integral.reshape(exact_solution.shape)
            x_QP_CVXOPT_integral[:, indices_to_eliminate] = 0.0
            x_QP_CVXOPT_integral = normalize_data_class_integral.recover_solution(
                x_QP_CVXOPT_integral
            )
            CVXOPT_integral_dynamic.append(x_QP_CVXOPT_integral)
            display_statistics(
                x_QP_CVXOPT_integral,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            # Run the algorithm
            from SR3_algorithm import train_SR3

            equality_constraints = add_constraints_symmetry_kuramoto(
                polinomial, number_of_oscillators, normalization_factors
            )

            # Convert the constraints to vector matrix form.
            constraint_matrix = np.zeros(
                (
                    len(equality_constraints),
                    int(len(feature_names) * number_of_oscillators),
                )
            )
            constraint_vector = np.zeros(len(equality_constraints))
            for k in range(len(equality_constraints)):
                for key, value in equality_constraints[k].items():
                    if key != "constant":
                        constraint_matrix[k, int(key.replace("x", ""))] = value
                    else:
                        constraint_vector[k] = value

            # Keep only the coefficients that we need
            true_constraint_matrix = np.zeros(
                (
                    len(equality_constraints),
                    int(len(indices_to_use) * number_of_oscillators),
                )
            )
            list_nonzero_indices = []
            for k in range(number_of_oscillators):
                list_nonzero_indices += [
                    elem + len(feature_names) * k for elem in indices_to_use
                ]
            true_constraint_matrix = constraint_matrix[:, list_nonzero_indices]
            true_constraint_vector = constraint_vector[
                ~np.all(true_constraint_matrix == 0, axis=1)
            ]
            true_constraint_matrix = true_constraint_matrix[
                ~np.all(true_constraint_matrix == 0, axis=1)
            ]

            testing_fun = testing_function(
                Psi_test[indices_to_use],
                Y_test,
                0.0,
                exact_solution[:, indices_to_use],
                normalization_factors[indices_to_use],
            )

            print("SR3 constrained l0 (differential)")
            dynamic_SR3_constraint_l0 = np.zeros(exact_solution.shape)
            dynamic_SR3_constraint_l0[:, indices_to_use] = train_SR3(
                Psi_train[indices_to_use, :],
                Y_train,
                testing_fun,
                type_regularization="l0",
                equality_constraint_matrix=true_constraint_matrix,
                equality_constraint_vector=true_constraint_vector,
                show_progress=False,
            )
            dynamic_SR3_constraint_l0 = normalize_data_class.recover_solution(
                dynamic_SR3_constraint_l0
            )
            SR3_constraint_l0_dynamic.append(dynamic_SR3_constraint_l0)
            display_statistics(
                dynamic_SR3_constraint_l0,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            print("SR3 constrained l1 (differential)")
            dynamic_SR3_constraint_l1 = np.zeros(exact_solution.shape)
            dynamic_SR3_constraint_l1[:, indices_to_use] = train_SR3(
                Psi_train[indices_to_use, :],
                Y_train,
                testing_fun,
                type_regularization="l1",
                equality_constraint_matrix=true_constraint_matrix,
                equality_constraint_vector=true_constraint_vector,
                show_progress=False,
            )
            dynamic_SR3_constraint_l1 = normalize_data_class.recover_solution(
                dynamic_SR3_constraint_l1
            )
            SR3_constraint_l1_dynamic.append(dynamic_SR3_constraint_l1)
            display_statistics(
                dynamic_SR3_constraint_l1,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            print("SR3 l0 (differential)")
            dynamic_SR3_l0 = np.zeros(exact_solution.shape)
            dynamic_SR3_l0[:, indices_to_use] = train_SR3(
                Psi_train[indices_to_use, :],
                Y_train,
                testing_fun,
                type_regularization="l0",
                show_progress=False,
            )
            dynamic_SR3_l0 = normalize_data_class.recover_solution(dynamic_SR3_l0)
            SR3_l0_dynamic.append(dynamic_SR3_l0)
            display_statistics(
                dynamic_SR3_l0, exact_solution, noise_parameter[j], formulation_matrices
            )

            print("SR3 l1 (differential)")
            dynamic_SR3_l1 = np.zeros(exact_solution.shape)
            dynamic_SR3_l1[:, indices_to_use] = train_SR3(
                Psi_train[indices_to_use, :],
                Y_train,
                testing_fun,
                type_regularization="l1",
                show_progress=False,
            )
            dynamic_SR3_l1 = normalize_data_class.recover_solution(dynamic_SR3_l1)
            SR3_l1_dynamic.append(dynamic_SR3_l1)
            display_statistics(
                dynamic_SR3_l1, exact_solution, noise_parameter[j], formulation_matrices
            )

            equality_constraints_integral = add_constraints_symmetry_kuramoto(
                polinomial, number_of_oscillators, normalization_factors_integral
            )

            # Convert the constraints to vector matrix form.
            constraint_matrix_integral = np.zeros(
                (
                    len(equality_constraints),
                    int(len(feature_names) * number_of_oscillators),
                )
            )
            constraint_vector_integral = np.zeros(len(equality_constraints))
            for k in range(len(equality_constraints_integral)):
                for key, value in equality_constraints_integral[k].items():
                    if key != "constant":
                        constraint_matrix_integral[k, int(key.replace("x", ""))] = value
                    else:
                        constraint_vector_integral[k] = value

            # Keep only the coefficients that we need
            true_constraint_matrix_integral = np.zeros(
                (
                    len(equality_constraints_integral),
                    int(len(indices_to_use) * number_of_oscillators),
                )
            )
            list_nonzero_indices = []
            for k in range(number_of_oscillators):
                list_nonzero_indices += [
                    elem + len(feature_names) * k for elem in indices_to_use
                ]
            true_constraint_matrix_integral = constraint_matrix_integral[
                :, list_nonzero_indices
            ]
            true_constraint_vector_integral = constraint_vector_integral[
                ~np.all(true_constraint_matrix_integral == 0, axis=1)
            ]
            true_constraint_matrix_integral = true_constraint_matrix_integral[
                ~np.all(true_constraint_matrix_integral == 0, axis=1)
            ]

            testing_fun_integral = testing_function(
                matrix_test[indices_to_use],
                delta_test,
                0.0,
                exact_solution[:, indices_to_use],
                normalization_factors_integral[indices_to_use],
            )

            print("SR3 constrained l0 (integral)")
            dynamic_SR3_constraint_l0_integral = np.zeros(exact_solution.shape)
            dynamic_SR3_constraint_l0_integral[:, indices_to_use] = train_SR3(
                matrix_train[indices_to_use, :],
                delta_train,
                testing_fun_integral,
                type_regularization="l0",
                equality_constraint_matrix=true_constraint_matrix_integral,
                equality_constraint_vector=true_constraint_vector_integral,
                show_progress=False,
            )
            dynamic_SR3_constraint_l0_integral = (
                normalize_data_class_integral.recover_solution(
                    dynamic_SR3_constraint_l0_integral
                )
            )
            SR3_constraint_l0_integral_dynamic.append(
                dynamic_SR3_constraint_l0_integral
            )
            display_statistics(
                dynamic_SR3_constraint_l0_integral,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            print("SR3 constrained l1 (integral)")
            dynamic_SR3_constraint_l1_integral = np.zeros(exact_solution.shape)
            dynamic_SR3_constraint_l1_integral[:, indices_to_use] = train_SR3(
                matrix_train[indices_to_use, :],
                delta_train,
                testing_fun_integral,
                type_regularization="l1",
                equality_constraint_matrix=true_constraint_matrix_integral,
                equality_constraint_vector=true_constraint_vector_integral,
                show_progress=False,
            )
            dynamic_SR3_constraint_l1_integral = (
                normalize_data_class_integral.recover_solution(
                    dynamic_SR3_constraint_l1_integral
                )
            )
            SR3_constraint_l1_integral_dynamic.append(
                dynamic_SR3_constraint_l1_integral
            )
            display_statistics(
                dynamic_SR3_constraint_l1_integral,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            print("SR3 l0 (integral)")
            dynamic_SR3_l0_integral = np.zeros(exact_solution.shape)
            dynamic_SR3_l0_integral[:, indices_to_use] = train_SR3(
                matrix_train[indices_to_use, :],
                delta_train,
                testing_fun_integral,
                type_regularization="l0",
                show_progress=False,
            )
            dynamic_SR3_l0_integral = normalize_data_class_integral.recover_solution(
                dynamic_SR3_l0_integral
            )
            SR3_l0_integral_dynamic.append(dynamic_SR3_l0_integral)
            display_statistics(
                dynamic_SR3_l0_integral,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

            print("SR3 l1 (integral)")
            dynamic_SR3_l1_integral = np.zeros(exact_solution.shape)
            dynamic_SR3_l1_integral[:, indices_to_use] = train_SR3(
                matrix_train[indices_to_use, :],
                delta_train,
                testing_fun_integral,
                type_regularization="l1",
                show_progress=False,
            )
            dynamic_SR3_l1_integral = normalize_data_class_integral.recover_solution(
                dynamic_SR3_l1_integral
            )
            SR3_l1_dynamic.append(dynamic_SR3_l1_integral)
            display_statistics(
                dynamic_SR3_l1_integral,
                exact_solution,
                noise_parameter[j],
                formulation_matrices,
            )

    results = {
        "SINDy_dynamic": SINDy_dynamic,
        "SINDy_integral_dynamic": SINDy_integral_dynamic,
        "FISTA_dynamic": FISTA_dynamic,
        "FISTA_integral_dynamic": FISTA_integral_dynamic,
        "BCG_dynamic": BCG_dynamic,
        "BCG_constraint_dynamic": BCG_constraint_dynamic,
        "BCG_integral_dynamic": BCG_integral_dynamic,
        "BCG_integral_constraint_dynamic": BCG_integral_constraint_dynamic,
        "CVXOPT_dynamic": CVXOPT_dynamic,
        "CVXOPT_constraint_dynamic": CVXOPT_constraint_dynamic,
        "CVXOPT_integral_dynamic": CVXOPT_integral_dynamic,
        "CVXOPT_integral_constraint_dynamic": CVXOPT_integral_constraint_dynamic,
        "SR3_constrained_l0_dynamic": SR3_constraint_l0_dynamic,
        "SR3_constrained_l1_dynamic": SR3_constraint_l1_dynamic,
        "SR3_l0_dynamic": SR3_l0_dynamic,
        "SR3_l1_dynamic": SR3_l1_dynamic,
        "SR3_constrained_l0_integral_dynamic": SR3_constraint_l0_integral_dynamic,
        "SR3_constrained_l1_integral_dynamic": SR3_constraint_l1_integral_dynamic,
        "SR3_l0_integral_dynamic": SR3_l0_integral_dynamic,
        "SR3_l1_integral_dynamic": SR3_l1_integral_dynamic,
        "psi_train_data": psi_train_data,
        "psi_test_data": psi_test_data,
        "psi_validation_data": psi_validation_data,
        "Y_train_data": Y_train_data,
        "Y_test_data": Y_test_data,
        "Y_validation_data": Y_validation_data,
        "matrix_train_data": matrix_train_data,
        "matrix_test_data": matrix_test_data,
        "matrix_validation_data": matrix_validation_data,
        "delta_train_data": delta_train_data,
        "delta_test_data": delta_test_data,
        "delta_validation_data": delta_validation_data,
        "noise": noise,
        "first_derivative_norm": first_derivative_norm,
        "delta_norm": delta_norm,
        "exact_dynamic": exact_solution,
    }

    from auxiliary_functions import dump_pickled_object

    ts = time.time()
    timestamp = (
        datetime.datetime.fromtimestamp(ts)
        .strftime("%Y-%m-%d %H:%M:%S")
        .replace(" ", "-")
        .replace(":", "-")
    )

    dump_pickled_object(
        os.path.join(
            os.getcwd(),
            "Results",
            "kura10_v7",
            "kura_v7_"
            + str(number_of_oscillators)
            + "osc_"
            + str(number_of_samples)
            + "samp_"
            + str(number_of_experiments)
            + "exp_"
            + str(max_degree_polynomial)
            + "degree_"
            + str(num_repetitions)
            + "rep_"
            + str(TIME_LIMIT_INNER)
            + "timein_"
            + str(order_polynomial_differentiation)
            + "int_"
            + str(order_polynomial_differentiation)
            + "diff_"
            + str(timestamp)
            + ".pickle",
        ),
        results,
    )
