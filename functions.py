import numpy as np
from scipy.linalg import eigvalsh

"""# Objective Functions

"""


class quadratic_function_fast:
    def __init__(self, Psi, Y, alpha=0.0):
        self.sizePsi, _ = Psi.shape
        self.sizeY = Y.shape[0]
        self.Y = Y.copy()
        self.alpha = alpha
        self.Psi = Psi.copy()
        self.hessian_mat = None
        self.L = None
        self.mu = None
        self.y_dot_psi = np.dot(self.Y,self.Psi.T)
        self.psi_dot_psi = np.dot(self.Psi,self.Psi.T)
        return

    # Evaluate function.
    def f(self, A):
        return (
            np.linalg.norm(
                self.Y - np.dot(A.reshape(self.sizeY, self.sizePsi), self.Psi)
            )
            ** 2
            + self.alpha * np.linalg.norm(A) ** 2
        )
    
    def gradient(self, A):
        return (
            -2.0
            * self.y_dot_psi.flatten()
            +2.0
            * np.dot(A.reshape(self.sizeY, self.sizePsi), self.psi_dot_psi).flatten()
            + 2.0 * self.alpha * A.flatten()
        )
    
    # Evaluate hessian.
    def hessian(self):
        return 2.0 * self.psi_dot_psi + 2.0 * self.alpha * np.identity(len(self.psi_dot_psi))

    # Line Search.
    def line_search(self, grad, d, x, maxStep=1.0):
        grad_aux = grad.reshape(self.sizeY, self.sizePsi)
        d_aux = d.reshape(self.sizeY, self.sizePsi)
        return min(
            maxStep,
            -np.trace(np.dot(0.5 * grad_aux.T, d_aux))
            / (
                np.linalg.norm(np.dot(d_aux, self.Psi)) ** 2
                + self.alpha * np.linalg.norm(d_aux) ** 2
            ),
        )

    def psi_val(self):
        return self.Psi

    def y_val(self):
        return self.Y

    def number_basis_functions(self):
        return self.sizePsi

    def number_dimensions(self):
        return self.sizeY

    # Return largest eigenvalue.
    def largest_eigenvalue(self):
        if self.L is None or self.mu is None:
            w = eigvalsh(self.hessian())
            self.L = np.max(w)
            self.mu = np.min(w)
        return self.L

    # Return smallest eigenvalue.
    def smallest_eigenvalue(self):
        if self.L is None or self.mu is None:
            w = eigvalsh(self.hessian())
            self.L = np.max(w)
            self.mu = np.min(w)
        return self.mu

    def proximal_operator(self, A, lambda_val):
        return np.sign(A) * np.maximum(np.abs(A) - lambda_val, 0.0)

class quadratic_LASSO:
    def __init__(self, Psi, Y):
        self.sizePsi, _ = Psi.shape
        self.sizeY = Y.shape[0]
        self.Y = Y.copy()
        self.Psi = Psi.copy()
        self.hessian_mat = None
        self.L = None
        return

    # Evaluate function.
    #Where f = F + G, F is smooth convex and differentiable, and G is convex.
    def f_smooth(self, A):
        return (
           0.5* np.linalg.norm(
                self.Y - np.dot(A.reshape(self.sizeY, self.sizePsi), self.Psi)
            )
            ** 2
        )

    def f(self, A, regularization):
        return (
           0.5* np.linalg.norm(
                self.Y - np.dot(A.reshape(self.sizeY, self.sizePsi), self.Psi)
            )
            ** 2 + regularization*np.sum(np.abs(A))
        )

    # Evaluate gradient.
    def gradient_smooth(self, A):
        return (
            -np.dot(
                self.Y - np.dot(A.reshape(self.sizeY, self.sizePsi), self.Psi),
                self.Psi.T,
            )
        )

    # Evaluate hessian.
    def hessian(self):
        if self.hessian_mat is None:
            self.hessian_mat = np.dot(self.Psi, self.Psi.T)
        return self.hessian_mat

    # Return largest eigenvalue.
    def largest_eigenvalue_smooth(self):
        if(self.L is None):
            from scipy.sparse.linalg.eigen.arpack import eigsh
            hessian_mat = np.dot(self.Psi, self.Psi.T)
            self.L = eigsh(hessian_mat, 1, which='LM', return_eigenvectors = False)[0]
        return self.L
    
    def proximal_operator(self, A, lambda_val):
        return np.sign(A) * np.maximum(np.abs(A) - lambda_val, 0.)
    
    def starting_point(self):
        return np.zeros((self.sizeY, self.sizePsi))
    
    def return_Psi(self):
        return self.Psi
    
    def return_Y(self):
        return self.Y
    
    def return_shape(self):
        return (self.sizeY, self.sizePsi)

class solution_polishing:
    import numpy as np
    from scipy.sparse import issparse

    def __init__(self, Q, b):
        self.Q = Q.copy()
        self.b = b.copy()
        w = eigvalsh(self.Q.todense())
        self.L = np.max(w)
        self.Mu = np.min(w)
        return

    # Evaluate function.
    def f(self, x):
        return self.b.dot(x) + 0.5 * x.T.dot(self.Q.dot(x))

    # Evaluate gradient.
    def gradient(self, x):
        return self.b + self.Q.dot(x)

    # Line Search.
    def line_search(self, grad, d, x, maxStep=1.0):
        alpha = -d.dot(grad) / d.T.dot(self.Q.dot(d))
        return min(alpha, maxStep)

    def largest_eigenvalue(self):
        return self.L

    def smallest_eigenvalue(self):
        return self.Mu
