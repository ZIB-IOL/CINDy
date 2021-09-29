import numpy as np
import time
from scipy.sparse import isspmatrix_csr, issparse

# Import functions for the step_sizes and the updates.
from auxiliary_functions import step_size, polish_solution

# Import functions for active set management
from auxiliary_functions import (
    new_vertex_fail_fast,
    delete_vertex_index,
    max_min_vertex,
)


def CINDy(
    function,
    feasible_region,
    tolerance_outer,
    tolerance_inner,
    max_time_inner,
    testing_function,
    max_iterations=2000,
    type_criterion="FW",
    primal_improvement=1.0e-5,
):
    x = feasible_region.initial_point()
    loss_evolution = [testing_function.evaluate(x)]
    real_loss = [testing_function.compare_exact(x)]
    timing = [time.time()]
    testing_losses = []
    true_losses = []
    x = feasible_region.initial_point()
    if type_criterion == "FCFW":
        (
            x,
            testing_loss,
            true_loss,
            testing_running_avg,
            _,
            _,
            all_true_losses,
            all_testing_losses,
            all_running_avg,
            timing_values,
            gap_values,
        ) = approximate_fully_corrective_FW(
            function,
            x,
            [x.copy()],
            [1.0],
            feasible_region,
            tolerance_outer,
            tolerance_inner,
            max_time_inner,
            testing_function,
            max_iterations,
            type_criterion=type_criterion,
            primal_improvement=primal_improvement,
        )
    if type_criterion == "BCG":
        (
            x,
            testing_loss,
            true_loss,
            testing_running_avg,
            _,
            _,
            all_true_losses,
            all_testing_losses,
            all_running_avg,
            timing_values,
            gap_values,
        ) = blended_conditional_gradients(
            function,
            x,
            [x.copy()],
            [1.0],
            feasible_region,
            tolerance_outer,
            tolerance_inner,
            max_time_inner,
            testing_function,
            max_iterations,
            type_criterion=type_criterion,
            primal_improvement=primal_improvement,
        )
    testing_losses = testing_losses + all_testing_losses
    true_losses = true_losses + all_true_losses
    timing.append(time.time())
    loss_evolution.append(testing_loss)
    real_loss.append(true_loss)
    timing[:] = [t - timing[0] for t in timing]
    return (
        x,
        loss_evolution,
        real_loss,
        timing,
        true_losses,
        testing_losses,
        timing_values,
        gap_values,
    )


def approximate_fully_corrective_FW(
    function,
    x,
    active_set,
    lambdaVal,
    feasible_region,
    outer_tolerance,
    inner_tolerance,
    max_time,
    testing_function,
    max_iterations=2000,
    type_criterion="FW",
    threshold=1.0e-9,
    num_moving_average=5,
    primal_improvement=1.0e-5,
):
    time_ref = time.time()
    xbest = x.copy()
    active_set_best = active_set.copy()
    lambda_val_best = lambdaVal.copy()
    all_true_losses = [testing_function.compare_exact(x)]
    all_testing_losses = [testing_function.evaluate(x)]
    running_average = [testing_function.evaluate(x)]
    itCount = 1
    grad = function.gradient(x)
    gap = (x - feasible_region.linear_programming_oracle(grad)).T.dot(grad)
    timing = [time.time()]
    gap_values = [gap]
    while True:
        # print(x)
        for i in range(10):
            x, gap_aux = away_step_CG(
                function, feasible_region, x, active_set, lambdaVal, "EL"
            )
        grad = function.gradient(x)
        gap = (x - feasible_region.linear_programming_oracle(grad)).T.dot(grad)
        if len(active_set) >= 2 and gap != 0.0:
            x, active_set[:], lambdaVal[:] = polish_solution(
                function.psi_val(),
                function.y_val(),
                active_set,
                lambdaVal,
                tolerance=inner_tolerance,
                threshold=threshold,
                type_criterion="FW",
                time_limit=max_time,
                max_steps=max_iterations,
            )
        grad = function.gradient(x)
        gap = (x - feasible_region.linear_programming_oracle(grad)).T.dot(grad)
        timing.append(time.time())
        gap_values.append(gap)
        all_true_losses.append(testing_function.compare_exact(x))
        all_testing_losses.append(testing_function.evaluate(x))
        if len(all_testing_losses) < num_moving_average:
            running_average.append(np.mean(np.asarray(all_testing_losses)))
        else:
            running_average.append(
                np.mean(np.asarray(all_testing_losses)[-num_moving_average:])
            )

        if running_average[-1] < min(running_average[:-1]):
            xbest = x.copy()
            active_set_best = active_set.copy()
            lambda_val_best = lambdaVal.copy()

        if (
            time.time() - time_ref > max_time
            or itCount > max_iterations
            or gap < outer_tolerance
            or np.abs(all_testing_losses[-2] - all_testing_losses[-1])
            < primal_improvement
        ):
            xbest, active_set_best[:], lambda_val_best[:] = polish_solution(
                function.psi_val(),
                function.y_val(),
                active_set_best,
                lambda_val_best,
                tolerance=1.0e-6,
                threshold=1.0e-4,
                type_criterion="FW",
                time_limit=120,
                max_steps=np.inf,
            )
            xbest, active_set_best[:], lambda_val_best[:] = polish_solution(
                function.psi_val(),
                function.y_val(),
                active_set_best,
                lambda_val_best,
                tolerance=1.0e-6,
                threshold=0.0,
                type_criterion="FW",
                time_limit=120,
                max_steps=np.inf,
            )
            timing[:] = [t - timing[0] for t in timing]
            return (
                xbest,
                testing_function.evaluate(xbest),
                testing_function.compare_exact(xbest),
                min(running_average),
                active_set_best,
                lambda_val_best,
                all_true_losses,
                all_testing_losses,
                running_average,
                timing,
                gap_values,
            )
        itCount += 1


def blended_conditional_gradients(
    function,
    x,
    active_set,
    lambdaVal,
    feasible_region,
    outer_tolerance,
    inner_tolerance,
    max_time,
    testing_function,
    max_iterations=np.inf,
    type_criterion="FW",
    threshold=1.0e-9,
    K=4.0,
    num_moving_average=5,
    primal_improvement=1.0e-5,
):
    from auxiliary_functions import polish_solution

    time_ref = time.time()
    xbest = x.copy()
    active_set_best = active_set.copy()
    lambda_val_best = lambdaVal.copy()
    all_true_losses = [testing_function.compare_exact(x)]
    all_testing_losses = [testing_function.evaluate(x)]
    running_average = [testing_function.evaluate(x)]
    itCount = 1
    grad = function.gradient(x)
    gap = (x - feasible_region.linear_programming_oracle(grad)).T.dot(grad)
    phi_val = gap / 2.0
    timing = [time.time()]
    gap_values = [gap]
    phi_values = [phi_val]
    while True:
        x, active_set[:], lambdaVal[:] = polish_solution(
            function.psi_val(),
            function.y_val(),
            active_set,
            lambdaVal,
            tolerance=phi_val / 4.0,
            threshold=threshold,
            type_criterion="blended",
            time_limit=max_time,
        )
        grad = function.gradient(x)
        gap = (x - feasible_region.linear_programming_oracle(grad)).T.dot(grad)
        timing.append(time.time())
        gap_values.append(gap)
        phi_values.append(phi_val)
        if testing_function.evaluate(x) < min(all_testing_losses):
            xbest = x.copy()
            active_set_best = active_set.copy()
            lambda_val_best = lambdaVal.copy()
        all_true_losses.append(testing_function.compare_exact(x))
        all_testing_losses.append(testing_function.evaluate(x))
        if len(all_testing_losses) < num_moving_average:
            running_average.append(np.mean(np.asarray(all_testing_losses)))
        else:
            running_average.append(
                np.mean(np.asarray(all_testing_losses)[-num_moving_average:])
            )
        if running_average[-1] < min(running_average[:-1]):
            xbest = x.copy()
            active_set_best = active_set.copy()
            lambda_val_best = lambdaVal.copy()

        if (
            time.time() - time_ref > max_time
            or itCount > max_iterations
            or gap < outer_tolerance
        ):
            timing[:] = [t - timing[0] for t in timing]
            return (
                xbest,
                testing_function.evaluate(xbest),
                testing_function.compare_exact(xbest),
                min(running_average),
                active_set_best,
                lambda_val_best,
                all_true_losses,
                all_testing_losses,
                running_average,
                timing,
                gap_values,
            )
        if gap >= phi_val / K:
            x, gap_aux = away_step_CG(
                function, feasible_region, x, active_set, lambdaVal, "EL"
            )
        else:
            phi_val = gap / 2.0
        itCount += 1


def accelerated_projected_gradient_descent(
    f,
    feasible_region,
    active_set,
    tolerance,
    alpha0,
    time_limit=60,
    max_iteration=100,
    type_criterion="FW",
):
    """
    Run Nesterov's accelerated projected gradient descent.

    References
    ----------
    Nesterov, Y. (2018). Lectures on convex optimization (Vol. 137).
    Berlin, Germany: Springer. (Constant scheme II, Page 93)

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns projection oracle of a point x onto the feasible region,
        which are computed through the function feasible_region.project(x).
        Additionally, a LMO is used to compute the Frank-Wolfe gap (used as a
        stopping criterion) through the function
        feasible_region.linear_optimization_oracle(grad) function, which
        minimizes <x, grad> over the feasible region.
    tolerance : float
        Frank-Wolfe accuracy to which the solution is outputted.

    Returns
    -------
    x : numpy array
        Outputted solution with primal gap below the target tolerance
    """
    from collections import deque

    # Quantities we want to output.
    L = f.largest_eigenvalue()
    mu = f.smallest_eigenvalue()
    x = deque([np.asarray(alpha0)], maxlen=2)
    y = np.asarray(alpha0)
    q = mu / L
    if mu < 1.0e-3:
        alpha = deque([0], maxlen=2)
    else:
        alpha = deque([np.sqrt(q)], maxlen=2)
    grad = f.gradient(x[-1])
    if type_criterion == "FW":
        FWGap = grad.dot(x[-1] - feasible_region.linear_programming_oracle(grad))
    if type_criterion == "blended":
        away_vertex, _ = feasible_region.away_oracle(grad,[], x[-1])
        FWGap = grad.dot(
            away_vertex
            - feasible_region.linear_programming_oracle(grad)
        )
    time_ref = time.time()
    it_count = 0
    gap_values = [FWGap]
    while FWGap > tolerance:
        x.append(feasible_region.project(y - 1 / L * f.gradient(y)))
        if mu < 1.0e-3:
            alpha.append(0.5 * (1 + np.sqrt(1 + 4 * alpha[-1] * alpha[-1])))
            beta = (alpha[-2] - 1.0) / alpha[-1]
        else:
            root = np.roots([1, alpha[-1] ** 2 - q, -alpha[-1] ** 2])
            root = root[(root >= 0.0) & (root < 1.0)]
            assert len(root) != 0, "Root does not meet desired criteria.\n"
            alpha.append(root[0])
            beta = alpha[-2] * (1 - alpha[-2]) / (alpha[-2] ** 2 - alpha[-1])
        y = x[-1] + beta * (x[-1] - x[-2])
        grad = f.gradient(x[-1])
        if type_criterion == "FW":
            FWGap = grad.dot(x[-1] - feasible_region.linear_programming_oracle(grad))
        if type_criterion == "blended":
            away_vertex, _ = feasible_region.away_oracle(grad,[], x[-1])
            FWGap = grad.dot(
                away_vertex
                - feasible_region.linear_programming_oracle(grad)
            )
        it_count += 1
        if time.time() - time_ref > time_limit or it_count > max_iteration:
            break
        gap_values.append(FWGap)
    w = np.zeros(len(active_set[0]))
    for i in range(len(active_set)):
        w += x[-1][i] * active_set[i]
    return w, x[-1].tolist(), gap_values


def away_step_CG(function, feasible_region, x, active_set, lambdas, type_of_step):
    """
    Performs a single step of the ACG/AFW algorithm.

    Parameters
    ----------
    function: function being minimized
        Function that we will minimize.
    feasible_region : feasible region function.
        Returns LP oracles over feasible region.
    x : numpy array.
        Point.
    active_set : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    type_of_step : str
        Type of step size used.

    Returns
    -------
    x + alpha*d
        Output point
    FWGap
        FW gap at initial point.

    """
    grad = function.gradient(x)
    v = feasible_region.linear_programming_oracle(grad)
    a, indexMax = feasible_region.away_oracle(grad, active_set, x)
    # Choose FW direction, can overwrite index.
    FWGap = (x - feasible_region.linear_programming_oracle(grad)).T.dot(grad)
    away_gap = (a - x).T.dot(grad)
    if issparse(FWGap):
        FWGap = FWGap.todense().item()
        away_gap = away_gap.todense().item()
    if FWGap >= away_gap:
        d = v - x
        alphaMax = 1.0
        optStep = step_size(function, d, grad, x, type_of_step)
        alpha = min(optStep, alphaMax)
        if alpha != alphaMax:
            # newVertex returns true if vertex is new.
            flag, index = feasible_region.new_vertex(v, active_set)
            lambdas[:] = [i * (1 - alpha) for i in lambdas]
            if flag:
                active_set.append(v)
                lambdas.append(alpha)
            else:
                # Update existing weights
                lambdas[index] += alpha
        # Max step length away step, only one vertex now.
        else:
            active_set[:] = [v]
            lambdas[:] = [alphaMax]
    else:
        d = x - a
        alphaMax = lambdas[indexMax] / (1.0 - lambdas[indexMax])
        optStep = step_size(function, d, grad, x, type_of_step, maxStep=alphaMax)
        alpha = min(optStep, alphaMax)
        lambdas[:] = [i * (1 + alpha) for i in lambdas]
        # Max step, need to delete a vertex.
        if alpha != alphaMax:
            lambdas[indexMax] -= alpha
        else:
            delete_vertex_index(indexMax, active_set, lambdas)
    return x + alpha * d, FWGap
