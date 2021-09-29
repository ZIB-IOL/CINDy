import numpy as np
from auxiliary_functions import max_vertex, projection_simplex_sort, check_new_vertex
from scipy.sparse import csr_matrix
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
import pyscipopt as pso
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

class l1_ball_pyscipopt:
    def __init__(
        self,
        dimension,
        lambda_value,
        linear_equality_constraints=None,
        linear_inequality_constraints=None,
        normalization_factors=None,
    ):
        self.dimension = dimension
        self.lambda_value = lambda_value
        self.linear_equality_constraints = linear_equality_constraints
        self.linear_inequality_constraints = linear_inequality_constraints
        self.normalization_factors = normalization_factors
        # Initialize model
        self.model = Model("polytope")
        # Let’s deactivate presolving and heuristic solutions
        #        self.model.setPresolve(pso.SCIP_PARAMSETTING.OFF)
        #        self.model.setHeuristics(pso.SCIP_PARAMSETTING.OFF)
        #        self.model.disablePropagation()
        #        # let’s use the primal simplex
        self.model.setCharParam("lp/initalgorithm", "p")
        self.model.hideOutput()
        self.model.setBoolParam("reoptimization/enable", True)
        # Create variables
        self.x = {}
        for i in range(int(2.0 * self.dimension)):
            self.x["x" + str(i)] = self.model.addVar(vtype="C", name="x%s" % i, lb=0.0)
        # Create l1 Ball constraint constraints
        self.c = {}
        self.c["l1_constraint"] = self.model.addCons(
            quicksum(self.x["x" + str(i)] for i in range(int(2.0 * self.dimension)))
            == lambda_value,
            name="l1_constraint",
        )

        if linear_equality_constraints is not None:
            for constraint in linear_equality_constraints:
                self.model.addCons(
                    quicksum(
                        value * self.x[key]
                        - value
                        * self.x["x" + str(self.dimension + int(key.replace("x", "")))]
                        for key, value in constraint.items()
                        if key != "constant"
                    )
                    == constraint["constant"]
                )

        if linear_inequality_constraints is not None:
            for constraint in linear_inequality_constraints:
                self.model.addCons(
                    quicksum(
                        value * self.x[key]
                        - value
                        * self.x["x" + str(self.dimension + int(key.replace("x", "")))]
                        for key, value in constraint.items()
                        if key != "constant"
                    )
                    <= constraint["constant"]
                )

        # self.write_formulation_to_file()
        # quit()
        return

    def linear_programming_oracle(self, coefficients):
        # Check out how to do it correctly here: https://imada.sdu.dk/~marco/Misc/PySCIPOpt/classpyscipopt_1_1scip_1_1Model.html#aba052bc4ef52e8e85b19a9e385356453
        # Use chgReoptObjective
        auxiliary_coefficients = np.concatenate((coefficients, -coefficients), axis=0)
        #        self.model.freeTransform()
        self.model.freeReoptSolve()
        self.model.chgReoptObjective(
            quicksum(
                auxiliary_coefficients[i] * self.x["x" + str(i)]
                for i in range(int(2.0 * self.dimension))
            ),
            "minimize",
        )
        self.model.optimize()
        solution = np.zeros(int(2.0 * self.dimension))
        if self.model.getStatus() == "optimal":
            for i in range(int(2.0 * self.dimension)):
                val = self.model.getVal(self.x["x" + str(i)])
                if abs(val) > 1.0e-12:
                    solution[i] = -val
                else:
                    solution[i] = 0.0
        else:
            print("The LP solver was not able to reach a solution.")
        # Transform the solution.
        auxiliary_solution = -solution[: self.dimension] + solution[self.dimension :]
        return auxiliary_solution

    # Input is the xor over which we calculate the inner product.
    def away_oracle(self, grad, active_set, x):
        return max_vertex(grad, active_set)

    # Generate a random initial point
    def initial_point(self):
        grad = np.zeros(self.dimension)
        grad[0] = 1.0
        # grad = np.random.rand(self.dimension)
        return self.linear_programming_oracle(grad)

    # Sort projection for the simplex.
    def new_vertex(self, vertex, active_set):
        return check_new_vertex(vertex, active_set)

    def write_formulation_to_file(
        self, name_param="param.set", name_cip="prod1_scip.cip"
    ):
        # Write the set of SCIP parameters and their settings.
        self.model.writeParams(name_param)
        # Write the instantiated model to a file
        self.model.writeProblem(name_cip)  # cip format
        return

    def write_problem_over_convex_hull(self, point, Psi, y):
        return write_problem_over_convex_hull_general(point, Psi, y)

    def return_radius(self):
        return self.lambda_value

    def has_constraints(self):
        return True


class l1_ball:
    def __init__(self, dim, lambdaVal):
        self.dim = dim
        self.lambdaVal = lambdaVal

    def linear_programming_oracle(self, x):
        v = np.zeros(len(x), dtype=float)
        maxInd = np.argmax(np.abs(x))
        v[maxInd] = -1.0 * np.sign(x[maxInd])
        return self.lambdaVal * v

    # Input is the xor over which we calculate the inner product.
    def away_oracle(self, grad, active_set, x):
        return max_vertex(grad, active_set)

    def initial_point(self):
        v = np.zeros(self.dim)
        v[0] = 1.0
        return self.lambdaVal * v

    # Project into the L1Ball.
    def project(self, v):
        u = np.abs(v)
        if u.sum() <= self.lambdaVal:
            return v
        w = projection_simplex_sort(u, s=self.lambdaVal)
        w *= np.sign(v)
        return w

    # Sort projection for the simplex.
    def new_vertex(self, vertex, active_set):
        for i in range(len(active_set)):
            if active_set[i].T.dot(vertex) > 0.5:
                return False, i
        return True, np.nan

    def change_radius(self, new_radius):
        self.lambdaVal = new_radius
        return

    def has_constraints(self):
        return False


class probability_simplex:
    def __init__(self, dim, alpha = 1.0):
        self.dim = dim
        self.alpha = alpha

    def linear_programming_oracle(self, x):
        v = np.zeros(len(x), dtype=float)
        v[np.argmin(x)] = self.alpha
        return v

    # Sort projection for the simplex.
    def project(self, x, s=1):
        return projection_simplex_sort(x, s = self.alpha)

    def away_oracle(self, grad, active_set, x):
        aux = np.multiply(grad, np.sign(x))
        indices = np.where(x > 0.0)[0]
        v = np.zeros(len(x), dtype=float)
        indexMax = indices[np.argmax(aux[indices])]
        v[indexMax] = self.alpha
        return v, indexMax

    # Sort projection for the simplex.
    def new_vertex(self, vertex, active_set):
        for i in range(len(active_set)):
            if active_set[i].T.dot(vertex) > 0.5:
                return False, i
        return True, np.nan

    def initial_point(self):
        v = np.zeros(self.dim)
        v[0] = self.alpha
        return v
