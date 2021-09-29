import numpy as np


def STRidge(X, y, lam, maxit, tol, normalize=2, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """
    from sklearn.linear_model import Ridge

    n, d = X.shape

    w = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_.T

    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]
    # Threshold and continue
    for j in range(maxit):
        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                return w
            else:
                break
        biginds = new_biginds
        # Otherwise get a new guess
        w[smallinds] = 0
        w[biginds] = Ridge(alpha=lam, fit_intercept=False).fit(X[:, biginds], y).coef_.T
    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []:
        w[biginds] = Ridge(alpha=0.0, fit_intercept=False).fit(X[:, biginds], y).coef_.T
    return w


def TrainSTRidge(
    TrainR,
    TrainY,
    TestR,
    TestY,
    lam,
    d_tol,
    maxit=200,
    STR_iters=200,
    l0_penalty=0.0,
    print_best_tol=False,
):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them
    using a loss function on a holdout set.

    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """
    from sklearn.linear_model import Ridge

    D = TrainR.shape[1]

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None:
        l0_penalty = 0.001 * np.linalg.cond(TrainR)

    # Get the standard least squares estimator
    w = np.zeros((D, 1))
    w_best = Ridge(alpha=0.0, fit_intercept=False).fit(TrainR, TrainY).coef_.T
    err_best = np.linalg.norm(
        TestY - TestR.dot(w_best), 2
    ) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    errors = [
        np.linalg.norm(TestY - TestR.dot(w_best), 2)
        + l0_penalty * np.count_nonzero(w_best)
    ]
    tolerances = [0]

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(TrainR, TrainY, lam, STR_iters, tol)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty * np.count_nonzero(w)

        errors.append(err)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0, tol - 2 * d_tol])
            d_tol = 2 * d_tol / (maxit - iter)
            tol = tol + d_tol
        tolerances.append(tol)

    if print_best_tol:
        print("Optimal tolerance: " + str(tol_best))
    return w_best
