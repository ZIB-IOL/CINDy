# General imports
import numpy as np
import sys, os

sys.path.append("..")
sys.path.append(os.path.join(os.getcwd(), "Plotting"))


from dynamics import exact_solution_fermi_pasta_ulam


# Number of oscillators we will simulate
number_of_oscillators = 5
# Number of data points we will use
number_of_samples = 6000
number_of_experiments = 60
# Number of data points we will use
max_degree_polynomial = 3
# beta parameter used in the dynamic
beta = 0.7
noise_parameter = [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]

order_polynomial_differentiation = 8
number_repetitions = 1
# num_sets = 20
num_sets = 1

# Create the polinomial basis functions
from sklearn.preprocessing import PolynomialFeatures

polinomial = PolynomialFeatures(max_degree_polynomial)
# Generate all possible polinomials given number_of_oscillators points.
polinomial.fit_transform(np.ones((1, number_of_oscillators)))

print("List of all the polynomials used as basis functions")
print(polinomial.get_feature_names())
print("Number of basis functions: " + str(len(polinomial.get_feature_names())))

feature_names = polinomial.get_feature_names()

# Create the exact solution to the dynamic given the polynomials above
exact_solution = exact_solution_fermi_pasta_ulam(
    number_of_oscillators, polinomial, beta
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


# Generate the exact solution to the problem, according to the previously defined basis.
from auxiliary_functions import (
    evaluate_polynomial,
)

from dynamics import fermi_pasta_ulam_time_correct

for _ in range(num_sets):
    error_Psi = np.zeros((len(noise_parameter), number_repetitions))
    error_X_error = np.zeros((len(noise_parameter), number_repetitions))
    performance_Simpson = np.zeros((len(noise_parameter), number_repetitions))
    performance_central_first_derivative = np.zeros(
        (len(noise_parameter), number_repetitions)
    )
    performance_central_second_derivative = np.zeros(
        (len(noise_parameter), number_repetitions)
    )
    performance_poly_first_derivative = np.zeros(
        (order_polynomial_differentiation, len(noise_parameter), number_repetitions)
    )
    performance_poly_second_derivative = np.zeros(
        (order_polynomial_differentiation, len(noise_parameter), number_repetitions)
    )
    performance_int = np.zeros(
        (order_polynomial_differentiation, len(noise_parameter), number_repetitions)
    )

    for k in range(number_repetitions):
        for j in range(len(noise_parameter)):
            max_time = 3.0
            (
                list_X,
                list_first_derivative_clean,
                list_second_derivative_clean,
                list_t,
            ) = fermi_pasta_ulam_time_correct(
                number_of_oscillators,
                exact_solution,
                polinomial,
                max_time,
                int(number_of_samples / number_of_experiments),
                number_of_experiments=number_of_experiments,
            )

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

            error_X_error[j, k] = np.linalg.norm(
                np.hstack(list_X) - np.hstack(list_X_noisy)
            ) / np.linalg.norm(np.hstack(list_X))
            print(
                "Error in X: "
                + str(np.linalg.norm(np.hstack(list_X) - np.hstack(list_X_noisy)))
            )

            list_Psi_exact = [evaluate_polynomial(x, polinomial) for x in list_X]

            # Compute the matrices for the integral approach.
            # Create the list of psi.
            list_Psi_noisy = [evaluate_polynomial(x, polinomial) for x in list_X_noisy]

            num_basis_functions = exact_solution.shape[1]

            # Infer the derivatives from the noisy data.
            from auxiliary_functions import derivative_from_list

            (
                list_second_derivative_central,
                list_first_derivative_central,
                _,
                list_t_central,
            ) = derivative_from_list(list_X_noisy, list_t, method="central", diff=2)

            # In case we want to compare to the performance with clean derivatives.
            width = int(
                (
                    list_second_derivative_clean[0].shape[1]
                    - list_second_derivative_central[0].shape[1]
                )
                / 2.0
            )

            list_second_derivative_central = [
                x[:, width:-width] for x in list_first_derivative_clean
            ]
            list_first_derivative_central = [
                x[:, width:-width] for x in list_first_derivative_clean
            ]
            second_derivative_central_clean = np.hstack(list_second_derivative_central)
            first_derivative_central_clean = np.hstack(list_first_derivative_central)
            second_derivative_central = np.hstack(list_second_derivative_central)
            first_derivative_central = np.hstack(list_first_derivative_central)

            from auxiliary_functions import (
                compute_exact_FPUT_integral_formulation_matrix_correct,
                compute_integral_formulation_matrix,
            )

            list_exact_integral_central = (
                compute_exact_FPUT_integral_formulation_matrix_correct(
                    [x[:, width:-width] for x in list_X],
                    [x[:, width:-width] for x in list_first_derivative_clean],
                    [x[width:-width] for x in list_t],
                    exact_solution,
                    polinomial,
                    num_basis_functions,
                )
            )

            print(
                "Error in Psi(Y): "
                + str(
                    np.linalg.norm(
                        np.hstack(list_Psi_noisy) - np.hstack(list_Psi_exact)
                    )
                )
            )
            error_Psi[j, k] = np.linalg.norm(
                np.hstack(list_Psi_noisy) - np.hstack(list_Psi_exact)
            ) / np.linalg.norm(np.hstack(list_Psi_exact))
            print(
                "Error in the second derivative using central difference: "
                + str(
                    np.linalg.norm(
                        second_derivative_central - second_derivative_central_clean
                    )
                )
            )
            performance_central_second_derivative[j, k] = np.linalg.norm(
                second_derivative_central - second_derivative_central_clean
            ) / np.linalg.norm(second_derivative_central_clean)
            print(
                "Error in the first derivative using central difference: "
                + str(
                    np.linalg.norm(
                        first_derivative_central - first_derivative_central_clean
                    )
                )
            )
            performance_central_first_derivative[j, k] = np.linalg.norm(
                first_derivative_central - first_derivative_central_clean
            ) / np.linalg.norm(first_derivative_central_clean)

            print()

            list_Psi_central = [x[:, width:-width] for x in list_Psi_noisy]
            (
                list_delta_central,
                list_matrix_central,
            ) = compute_integral_formulation_matrix(
                list_Psi_central,
                list_first_derivative_central,
                list_t_central,
                polinomial,
                type_of_integration="Simpsons",
            )
            print()
            print(
                "Simpsons int: "
                + str(
                    np.linalg.norm(
                        np.hstack(list_matrix_central)
                        - np.hstack(list_exact_integral_central)
                    )
                )
            )
            performance_Simpson[j, k] = np.linalg.norm(
                np.hstack(list_matrix_central) - np.hstack(list_exact_integral_central)
            ) / np.linalg.norm(np.hstack(list_exact_integral_central))

            print()
            width = 5
            list_exact_integral_poly = (
                compute_exact_FPUT_integral_formulation_matrix_correct(
                    [x[:, width:-width] for x in list_X],
                    [x[:, width:-width] for x in list_first_derivative_clean],
                    [x[width:-width] for x in list_t],
                    exact_solution,
                    polinomial,
                    num_basis_functions,
                )
            )

            for i in range(1, order_polynomial_differentiation + 1):
                (
                    list_second_derivative_poly,
                    list_first_derivative_poly,
                    _,
                    _,
                ) = derivative_from_list(
                    list_X_noisy,
                    list_t,
                    method="poly",
                    deg=i,
                    diff=2,
                    width=width,
                )
                # In case we want to compare to the performance with clean derivatives.
                width = int(
                    (
                        list_second_derivative_clean[0].shape[1]
                        - list_second_derivative_poly[0].shape[1]
                    )
                    / 2.0
                )
                second_derivative_poly_clean = np.hstack(
                    [x[:, width:-width] for x in list_second_derivative_clean]
                )
                first_derivative_poly_clean = np.hstack(
                    [x[:, width:-width] for x in list_first_derivative_clean]
                )
                # Aggregate the data into a single numpy array.
                second_derivative_poly = np.hstack(list_second_derivative_poly)
                first_derivative_poly = np.hstack(list_first_derivative_poly)
                print(
                    "Second derivative Poly diff error order "
                    + str(i)
                    + ": "
                    + str(
                        np.linalg.norm(
                            second_derivative_poly - second_derivative_poly_clean
                        )
                    )
                )
                performance_poly_second_derivative[i - 1, j, k] = np.linalg.norm(
                    second_derivative_poly - second_derivative_poly_clean
                ) / np.linalg.norm(second_derivative_poly_clean)
                print(
                    "First derivative Poly diff error order "
                    + str(i)
                    + ": "
                    + str(
                        np.linalg.norm(
                            first_derivative_poly - first_derivative_poly_clean
                        )
                    )
                )
                performance_poly_first_derivative[i - 1, j, k] = np.linalg.norm(
                    first_derivative_poly - first_derivative_poly_clean
                ) / np.linalg.norm(first_derivative_poly_clean)

                print(noise_parameter[j])
                SNR1 = np.linalg.norm(
                    np.cov(second_derivative_poly_clean)
                ) / np.linalg.norm(
                    np.cov(second_derivative_poly - second_derivative_poly_clean)
                )
                SNR2 = np.linalg.norm(
                    np.cov(second_derivative_poly_clean), ord=2
                ) / np.linalg.norm(
                    np.cov(second_derivative_poly - second_derivative_poly_clean), ord=2
                )
                print(SNR1, SNR1 / (1.0 + SNR1))
                print(SNR2, SNR2 / (1.0 + SNR2))

                print()

    results = {
        "error_Psi": error_Psi,
        "error_X_error": error_X_error,
        "noise": noise_parameter,
        "performance_central_first_derivative": performance_central_first_derivative,
        "performance_central_second_derivative": performance_central_second_derivative,
        "performance_poly_first_derivative": performance_poly_first_derivative,
        "performance_poly_second_derivative": performance_poly_second_derivative,
        "performance_Simpson": performance_Simpson,
        "performance_int": performance_int,
    }

    from auxiliary_functions import dump_pickled_object

    import time, datetime

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
            "FPUT_formulation_error_v3_"
            + str(number_of_oscillators)
            + "osc_"
            + str(number_of_samples)
            + "samp_"
            + str(number_of_experiments)
            + "exp_"
            + str(max_degree_polynomial)
            + "degree_"
            + str(number_repetitions)
            + "rep_"
            + str(timestamp)
            + ".pickle",
        ),
        results,
    )
