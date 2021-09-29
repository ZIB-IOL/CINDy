# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:36:47 2020

@author: pccom
"""
import numpy as np

def kuramoto_time(frequencies, max_time, number_of_snapshots, number_of_experiments = 10, K = 2.0, forcing_param = 0.2, relative_tolerance = 1.0e-10, absolute_tolerance = 1.0e-13):
    from scipy.integrate import solve_ivp

    number_of_oscillators = len(frequencies)

    def kuramoto_ode(_, theta):
        [theta_i, theta_j] = np.meshgrid(theta, theta)
        return frequencies + K / number_of_oscillators * np.sin(theta_j - theta_i).sum(0) + forcing_param * np.sin(theta)
    
    list_snapshots = []
    list_derivatives = []
    list_times = []
    for i in range(number_of_experiments):
        theta_init = 2 * np.pi * np.random.rand(number_of_oscillators)
        sol = solve_ivp(kuramoto_ode, [0, max_time], theta_init, method='DOP853', t_eval=np.linspace(0, max_time, number_of_snapshots), rtol = relative_tolerance, atol = absolute_tolerance)
        snapshots = sol.y
        derivatives = np.zeros([number_of_oscillators, number_of_snapshots])
        for i in range(number_of_snapshots):
            derivatives[:, i] = kuramoto_ode(0, snapshots[:, i])
            
        list_derivatives.append(derivatives)
        list_times.append(sol['t'])
        list_snapshots.append(snapshots)
    return list_snapshots, list_derivatives, list_times

def kuramoto_time_individual(frequencies, initial_position, start_time, end_time, number_of_samples, K = 2.0, forcing_param = 0.2, relative_tolerance = 1.0e-10, absolute_tolerance = 1.0e-13):
    from scipy.integrate import solve_ivp

    number_of_oscillators = len(frequencies)

    def kuramoto_ode(_, theta):
        [theta_i, theta_j] = np.meshgrid(theta, theta)
        return frequencies + K / number_of_oscillators * np.sin(theta_j - theta_i).sum(0) + forcing_param * np.sin(theta)
    
    sol = solve_ivp(kuramoto_ode, [start_time, end_time], initial_position, method='DOP853', t_eval=np.linspace(start_time, end_time, number_of_samples), rtol = relative_tolerance, atol = absolute_tolerance)
    return sol.y, sol['t']

#Convert angles from a
def angle_conversion(angles):
    return angles - np.floor(angles/(2*np.pi))*2.0*np.pi

def kuramoto_time_backup(theta_init, frequencies, max_time, number_of_snapshots, K = 2.0, forcing_param = 0.2):
    from scipy.integrate import solve_ivp

    number_of_oscillators = len(theta_init)

    def kuramoto_ode(_, theta):
        [theta_i, theta_j] = np.meshgrid(theta, theta)
        return frequencies + K / number_of_oscillators * np.sin(theta_j - theta_i).sum(0) + forcing_param * np.sin(theta)

    sol = solve_ivp(kuramoto_ode, [0, max_time], theta_init, method='BDF',
                          t_eval=np.linspace(0, max_time, number_of_snapshots))
    snapshots = sol.y
    derivatives = np.zeros([number_of_oscillators, number_of_snapshots])
    for i in range(number_of_snapshots):
        derivatives[:, i] = kuramoto_ode(0, snapshots[:, i])
    return snapshots, derivatives, sol['t']

def kuramoto_random(num_oscillators, number_of_snapshots, frequencies, K = 2.0, forcing_param = 0.2):
    """Kuramoto model

    Generate data for the Kuramoto model represented by the differential equation

        d/dt x_i = w_i + (2/d) * (sin(x_1 - x_i) + ... + sin(x_d - x_i)) + 0.2 * sin(x_i).

    See [1]_ and [2]_ for details.

    Parameters
    ----------
    theta_init: ndarray
        initial distribution of the oscillators
    frequencies: ndarray
        natural frequencies of the oscillators
    time: float
        integration time for BDF method
    number_of_snapshots: int
        number of snapshots

    Returns
    -------
    snapshots: ndarray(number_of_oscillators, number_of_snapshots)
        snapshot matrix containing random displacements of the oscillators in [-0.1,0.1]
    derivatives: ndarray(number_of_oscillators, number_of_snapshots)
        matrix containing the corresponding derivatives

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    .. [2] J. A. Acebrón, L. L. Bonilla, C. J. Pérez Vicente, F. Ritort, R. Spigler, "The Kuramoto model: A simple
           paradigm for synchronization phenomena", Rev. Mod. Phys. 77, pp. 137-185 , 2005
    """
#    snapshots = 2 * np.pi * np.random.rand(num_oscillators, number_of_snapshots) - np.pi
    snapshots = 2 * np.pi * np.random.rand(num_oscillators, number_of_snapshots)
    derivatives = np.zeros(snapshots.shape)
    for i in range(number_of_snapshots):
        for j in range(num_oscillators):
            derivatives[j,i] = frequencies[j] + K / num_oscillators * np.sin(snapshots[:,i] - snapshots[j,i]).sum() + forcing_param * np.sin(snapshots[j,i])
    return snapshots, derivatives


# Build the exact dynamics of the fermi_pasta_ulam if we have polynomials of order up to three.
# In this case the basis functions are sines and cosines.
# The first number_of_oscillators x are cosines, and the rest are sines.
def exact_solution_kuramoto(number_of_oscillators, frequencies, polinomial, K = 2.0, forcing_param = 0.2):
    reference_polynomial = polinomial.get_feature_names()
    num_basis_functions = len(reference_polynomial)
    # Build the exact solution at the boundary
    exact_solution = np.zeros((number_of_oscillators, num_basis_functions))
    # Build the exact solution.
    for i in range(0, number_of_oscillators):
        # First order terms.
        exact_solution[i, reference_polynomial.index("1")] = frequencies[i]
        exact_solution[i, reference_polynomial.index("x" + str(number_of_oscillators + i))] = forcing_param
        for j in range(number_of_oscillators):
            exact_solution[i, reference_polynomial.index("x" + str(i) + ' x' + str(number_of_oscillators + j))] += K/number_of_oscillators
            exact_solution[i, reference_polynomial.index("x" + str(j) + ' x' + str(number_of_oscillators + i))] += -K/number_of_oscillators
    return exact_solution

def add_constraints_symmetry_kuramoto(polinomial, number_of_oscillators, normalization_factors):
    feature_names = polinomial.get_feature_names()
    num_basis_functions = len(feature_names)
    pair_symmetries = []
    #Constraint on the cos^2 terms:
    for i in range(number_of_oscillators):
        for j in range(number_of_oscillators):
            index_first = feature_names.index("x" + str(j) + '^2')
            index_first_transformed = index_first + int(num_basis_functions*i)
            pair_symmetries.append({ "x" + str(index_first_transformed) : 1, 'constant': 0})
            
    #Constraint on the sin^2 terms:
    for i in range(number_of_oscillators):
        for j in range(number_of_oscillators):
            index_first = feature_names.index("x" + str(number_of_oscillators + j) + '^2')
            index_first_transformed = index_first + int(num_basis_functions*i)
            pair_symmetries.append({ "x" + str(index_first_transformed) : 1, 'constant': 0})
    
    type_symmetries = [('sin', 'cos'), ('cos'), ('sin')]
    for symmetry in type_symmetries:
        if(symmetry == ('sin', 'cos') or symmetry == ('cos', 'sin')):
            #Symmetry cos-sin
            #For i and j---> \xi_i(cos(x_i) sin(x_j)) = \xi_j(cos(x_j) sin(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(i) + ' x' + str(number_of_oscillators + j))
                        index_second = feature_names.index("x" + str(j) + ' x' + str(number_of_oscillators + i))
                        index_first_transformed = index_first + int(num_basis_functions*i)  
                        index_second_transformed = index_second + int(num_basis_functions*j)    
                        pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                        
            #Symmetry sin-cos
            #For i and j---> \xi_i(sin(x_i) cos(x_j)) = \xi_j(sin(x_j) cos(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(j) + ' x' + str(i + number_of_oscillators))
                        index_second = feature_names.index("x" + str(i) + ' x' + str(j + number_of_oscillators))
                        index_first_transformed = index_first + + int(num_basis_functions*i)  
                        index_second_transformed = index_second + int(num_basis_functions*j)
                        pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                        
            #Symmetry cos-cos
            #For i and j---> \xi_i(cos(x_i) cos(x_j)) = \xi_j(cos(x_j) cos(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        if(i <= j):
                            index = feature_names.index("x" + str(i) + ' x' + str(j))
                        else:
                            index = feature_names.index("x" + str(j) + ' x' + str(i))
                        index_first_transformed = index + int(num_basis_functions*i)  
                        index_second_transformed = index + int(num_basis_functions*j)
                        pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})

            #Symmetry sin-sin
            #For i and j---> \xi_i(sin(x_i) sin(x_j)) = \xi_j(sin(x_j) sin(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        if(i <= j):
                            index = feature_names.index("x" + str(i + number_of_oscillators) + ' x' + str(j + number_of_oscillators))
                        else:
                            index = feature_names.index("x" + str(j + number_of_oscillators) + ' x' + str(i + number_of_oscillators))
                        index_first_transformed = index + int(num_basis_functions*i)  
                        index_second_transformed = index + int(num_basis_functions*j)
                        pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                                      
    for symmetry in type_symmetries:
        if(symmetry == ('cos')):
            #Symmetry cos
            #For i and j---> \xi_i(cos(x_i)) = \xi_j(cos(x_j))
            for i in range(1, number_of_oscillators):
                index_first = feature_names.index("x" + str(0))
                index_second = feature_names.index("x" + str(i))
                index_first_transformed = index_first 
                index_second_transformed = index_second + int(num_basis_functions*i)    
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})

    for symmetry in type_symmetries:
        if(symmetry == ('cos')):
            #Symmetry cos
            #For i and j---> \xi_i(cos(x_j)) = \xi_j(cos(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(j))
                        index_second = feature_names.index("x" + str(i))
                        index_first_transformed = index_first + + int(num_basis_functions*i)  
                        index_second_transformed = index_second + int(num_basis_functions*j)
                        pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                  
                
    for symmetry in type_symmetries:
        if(symmetry == ('sin')):
            #Symmetry sin
            #For i and j---> \xi_i(sin(x_i)) = \xi_j(sin(x_j))
            for i in range(1, number_of_oscillators):
                index_first = feature_names.index("x" + str(number_of_oscillators + 0))
                index_second = feature_names.index("x" + str(number_of_oscillators + i))
                index_first_transformed = index_first 
                index_second_transformed = index_second + int(num_basis_functions*i)    
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})

    for symmetry in type_symmetries:
        if(symmetry == ('sin')):
            #Symmetry sin
            #For i and j---> \xi_i(sin(x_j)) = \xi_j(sin(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(j + number_of_oscillators))
                        index_second = feature_names.index("x" + str(i + number_of_oscillators))
                        index_first_transformed = index_first + + int(num_basis_functions*i)  
                        index_second_transformed = index_second + int(num_basis_functions*j)
                        pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                  
                
    # Remove duplicates before we output.
    # pair_symmetries = [dict(t) for t in {tuple(d.keys()) for d in pair_symmetries}]
    pair_symmetries =delete_dictionaries_with_duplicate_keys(pair_symmetries)
    return pair_symmetries

def add_constraints_symmetry_kuramoto_backup(polinomial, number_of_oscillators, normalization_factors):
    feature_names = polinomial.get_feature_names()
    num_basis_functions = len(feature_names)
    pair_symmetries = []
    #Constraint on the cos^2 terms:
    for i in range(number_of_oscillators):
        for j in range(number_of_oscillators):
            index_first = feature_names.index("x" + str(j) + '^2')
            index_first_transformed = index_first + int(num_basis_functions*i)
            pair_symmetries.append({ "x" + str(index_first_transformed) : 1, 'constant': 0})
            
    #Constraint on the sin^2 terms:
    for i in range(number_of_oscillators):
        for j in range(number_of_oscillators):
            index_first = feature_names.index("x" + str(number_of_oscillators + j) + '^2')
            index_first_transformed = index_first + int(num_basis_functions*i)
            pair_symmetries.append({ "x" + str(index_first_transformed) : 1, 'constant': 0})
    
    type_symmetries = [('sin', 'cos'), ('cos'), ('sin')]
    for symmetry in type_symmetries:
        if(symmetry == ('sin', 'cos') or symmetry == ('cos', 'sin')):
            #Symmetry cos-sin
            #For i and j---> \xi_i(cos(x_i) sin(x_j)) = \xi_j(cos(x_j) sin(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(i) + ' x' + str(number_of_oscillators + j))
                        index_second = feature_names.index("x" + str(j) + ' x' + str(number_of_oscillators + i))
                        index_first_transformed = index_first + int(num_basis_functions*i)  
                        index_second_transformed = index_second + int(num_basis_functions*j)    
                        symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                        pair_symmetries.append({ "x" + str(symmetry_pair[0]) : normalization_factors[index_second % num_basis_functions][0], "x" + str(symmetry_pair[1]) : -normalization_factors[index_first % num_basis_functions][0], 'constant': 0})
                        
            #Symmetry sin-cos
            #For i and j---> \xi_i(sin(x_i) cos(x_j)) = \xi_j(sin(x_j) cos(x_i))
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(j) + ' x' + str(i + number_of_oscillators))
                        index_second = feature_names.index("x" + str(i) + ' x' + str(j + number_of_oscillators))
                        index_first_transformed = index_first + + int(num_basis_functions*i)  
                        index_second_transformed = index_second + int(num_basis_functions*j)
                        symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                        pair_symmetries.append({ "x" + str(symmetry_pair[0]) : normalization_factors[index_second % num_basis_functions][0], "x" + str(symmetry_pair[1]) : -normalization_factors[index_first % num_basis_functions][0], 'constant': 0})
                        
                        
    for symmetry in type_symmetries:
        if(symmetry == ('cos')):
            #Symmetry cos
            #For i and j---> \xi_i(sin(x_i) cos(x_j)) = \xi_j(sin(x_j) cos(x_i))
            for i in range(1, number_of_oscillators):
                index_first = feature_names.index("x" + str(0))
                index_second = feature_names.index("x" + str(i))
                index_first_transformed = index_first 
                index_second_transformed = index_second + int(num_basis_functions*i)    
                symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                pair_symmetries.append({ "x" + str(symmetry_pair[0]) : normalization_factors[index_second % num_basis_functions][0], "x" + str(symmetry_pair[1]) : -normalization_factors[index_first % num_basis_functions][0], 'constant': 0})    

    # for symmetry in type_symmetries:
    #     if(symmetry == ('cos')):
    #         #Symmetry cos
    #         #For i and j---> \xi_i(cos(x_j)) = \xi_j(cos(x_i))
    #         for i in range(number_of_oscillators):
    #             for j in range(number_of_oscillators):
    #                 if(i != j):
    #                     index_first = feature_names.index("x" + str(j))
    #                     index_second = feature_names.index("x" + str(i))
    #                     index_first_transformed = index_first + + int(num_basis_functions*i)  
    #                     index_second_transformed = index_second + int(num_basis_functions*j)
    #                     symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
    #                     pair_symmetries.append({ "x" + str(symmetry_pair[0]) : normalization_factors[index_second % num_basis_functions][0], "x" + str(symmetry_pair[1]) : -normalization_factors[index_first % num_basis_functions][0], 'constant': 0})
                        
                
    for symmetry in type_symmetries:
        if(symmetry == ('sin')):
            #Symmetry sin
            for i in range(1, number_of_oscillators):
                index_first = feature_names.index("x" + str(number_of_oscillators + 0))
                index_second = feature_names.index("x" + str(number_of_oscillators + i))
                index_first_transformed = index_first 
                index_second_transformed = index_second + int(num_basis_functions*i)    
                symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                pair_symmetries.append({ "x" + str(symmetry_pair[0]) : normalization_factors[index_second % num_basis_functions][0], "x" + str(symmetry_pair[1]) : -normalization_factors[index_first % num_basis_functions][0], 'constant': 0})

    # for symmetry in type_symmetries:
    #     if(symmetry == ('sin')):
    #         #Symmetry cos
    #         #For i and j---> \xi_i(sin(x_j)) = \xi_j(sin(x_i))
    #         for i in range(number_of_oscillators):
    #             for j in range(number_of_oscillators):
    #                 if(i != j):
    #                     index_first = feature_names.index("x" + str(j + number_of_oscillators))
    #                     index_second = feature_names.index("x" + str(i + number_of_oscillators))
    #                     index_first_transformed = index_first + + int(num_basis_functions*i)  
    #                     index_second_transformed = index_second + int(num_basis_functions*j)
    #                     symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
    #                     pair_symmetries.append({ "x" + str(symmetry_pair[0]) : normalization_factors[index_second % num_basis_functions][0], "x" + str(symmetry_pair[1]) : -normalization_factors[index_first % num_basis_functions][0], 'constant': 0})
                        
                
    # Remove duplicates before we output.
    # pair_symmetries = [dict(t) for t in {tuple(d.keys()) for d in pair_symmetries}]
    pair_symmetries =delete_dictionaries_with_duplicate_keys_backup(pair_symmetries)
    return pair_symmetries

def add_constraints_symmetry_kuramoto_simple(polinomial, number_of_oscillators, normalization_factors):
    feature_names = polinomial.get_feature_names()
    num_basis_functions = len(feature_names)
    pair_symmetries = []
    #Constraint on the cos^2 terms:
    for i in range(number_of_oscillators):
        for j in range(number_of_oscillators):
            index_first = feature_names.index("x" + str(j) + '^2')
            index_first_transformed = index_first + int(num_basis_functions*i)
            pair_symmetries.append({ "x" + str(index_first_transformed) : 1, 'constant': 0})
        
    #Constraint on the sin^2 terms:
    for i in range(number_of_oscillators):
        for j in range(number_of_oscillators):
            index_first = feature_names.index("x" + str(number_of_oscillators + j) + '^2')
            index_first_transformed = index_first + int(num_basis_functions*i)
            pair_symmetries.append({ "x" + str(index_first_transformed) : 1, 'constant': 0})
            
    pair_symmetries = delete_dictionaries_with_duplicate_keys(pair_symmetries)
    return pair_symmetries

def add_constraints_symmetry_fermi_pasta_ulam_tsingou(polinomial, number_of_oscillators, normalization_factors):
    feature_names = polinomial.get_feature_names()
    num_basis_functions = len(feature_names)
    pair_symmetries = []
    
    #First order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_i) = \xi_{i+-1}(x_{i+-1})
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                index_first = feature_names.index("x" + str(i))
                index_first_transformed = index_first + int(num_basis_functions*i)
                index_second = feature_names.index("x" + str(j))
                index_second_transformed = index_second + int(num_basis_functions*j)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})

    #First order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_{i+-1}) = \xi_{i+-1}(x_{i})
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                index_first = feature_names.index("x" + str(i))
                index_first_transformed = index_first + int(num_basis_functions*j)
                index_second = feature_names.index("x" + str(j))
                index_second_transformed = index_second + int(num_basis_functions*i)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})

    #Second order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_i x_{i+-1}) = \xi_{i+-1}(x_i x_{i+-1})
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                if(i <= j):
                    index = feature_names.index("x" + str(i) + " x" + str(j))
                else:
                    index = feature_names.index("x" + str(j) + " x" + str(i))
                index_first_transformed = index + int(num_basis_functions*i)
                index_second_transformed = index + int(num_basis_functions*j)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                
    #Second order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_i^2) = \xi_{i+-1}(x_{i+-1}^2)
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                index_first = feature_names.index("x" + str(i) + "^2")
                index_second = feature_names.index("x" + str(j) + "^2")
                index_first_transformed = index_first + int(num_basis_functions*i)
                index_second_transformed = index_second + int(num_basis_functions*j)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                
    #Second order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_{i+-1}^2) = \xi_{i+-1}(x_i^2)
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                index_first = feature_names.index("x" + str(i) + "^2")
                index_second = feature_names.index("x" + str(j) + "^2")
                index_first_transformed = index_first + int(num_basis_functions*j)
                index_second_transformed = index_second + int(num_basis_functions*i)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                
    #Third order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_i^2 x_{i+-1}) = \xi_{i+-1}(x_{i+-1}^2 x_i)
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                if(i <= j):
                    index_first = feature_names.index("x" + str(i)  + "^2 x" + str(j))
                    index_second = feature_names.index("x" + str(i)  + " x" + str(j) + "^2")
                else:
                    index_first = feature_names.index("x" + str(j) + " x" + str(i) + "^2")
                    index_second = feature_names.index("x" + str(j) + "^2 x" + str(i))
                index_first_transformed = index_first + int(num_basis_functions*i)
                index_second_transformed = index_second + int(num_basis_functions*j)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})

    #Third order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_i x_{i+-1}^2) = \xi_{i+-1}(x_{i+-1} x_i^2)
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                if(i <= j):
                    index_first = feature_names.index("x" + str(i)  + " x" + str(j) + "^2")
                    index_second = feature_names.index("x" + str(i)  + "^2 x" + str(j))
                else:
                    index_first = feature_names.index("x" + str(j) + "^2 x" + str(i))
                    index_second = feature_names.index("x" + str(j) + " x" + str(i) + "^2")
                index_first_transformed = index_first + int(num_basis_functions*i)
                index_second_transformed = index_second + int(num_basis_functions*j)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})

    #Third order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_i^3) = \xi_{i+-1}(x_{i+-1}^3)
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                index_first = feature_names.index("x" + str(i)  + "^3")
                index_second = feature_names.index("x" + str(j)  + "^3")
                index_first_transformed = index_first + int(num_basis_functions*i)
                index_second_transformed = index_second + int(num_basis_functions*j)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                
    #Third order monomials
    #For x_i and x_{i+-1} ---> \xi_i(x_{i+-1}^3) = \xi_{i+-1}(x_{i}^3)
    for i in range(number_of_oscillators):
        for j in [i-1, i+1]:
            if j >= 0 and j < number_of_oscillators:
                index_first = feature_names.index("x" + str(i)  + "^3")
                index_second = feature_names.index("x" + str(j)  + "^3")
                index_first_transformed = index_first + int(num_basis_functions*j)
                index_second_transformed = index_second + int(num_basis_functions*i)
                pair_symmetries.append({ "x" + str(index_first_transformed) : 1.0/normalization_factors[index_first % num_basis_functions][0], "x" + str(index_second_transformed) : -1.0/normalization_factors[index_second % num_basis_functions][0], 'constant': 0})
                    
    pair_symmetries = delete_dictionaries_with_duplicate_keys(pair_symmetries)
    return pair_symmetries




def delete_dictionaries_with_duplicate_keys(list_dictionaries):
    seen = set()
    new_dictionary = []
    for d in list_dictionaries:
        t = tuple(sorted(d.keys()))
        if t not in seen:
            seen.add(t)
            new_dictionary.append(d)
    return new_dictionary

def delete_dictionaries_with_duplicate_keys_backup(list_dictionaries):
    seen = set()
    new_dictionary = []
    for d in list_dictionaries:
        t = tuple(d.keys())
        if t not in seen:
            seen.add(t)
            new_dictionary.append(d)
    return new_dictionary

def add_constraints_symmetry_kuramoto_backup_v2(polinomial, number_of_oscillators):
    type_symmetries = [('sin', 'cos'), ('cos'), ('sin')]
    feature_names = polinomial.get_feature_names()
    pair_symmetries = []
    for symmetry in type_symmetries:
        if(symmetry == ('sin', 'cos') or symmetry == ('cos', 'sin')):
            #Symmetry cos-sin
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(i) + ' x' + str(number_of_oscillators + j))
                        index_second = feature_names.index("x" + str(j) + ' x' + str(number_of_oscillators + i))
                        index_first_transformed = index_first + int(len(feature_names)*i)  
                        index_second_transformed = index_second + int(len(feature_names)*j)    
                        symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                        pair_symmetries.append(symmetry_pair)
            #Symmetry sin-cos
            for i in range(number_of_oscillators):
                for j in range(number_of_oscillators):
                    if(i != j):
                        index_first = feature_names.index("x" + str(j) + ' x' + str(i + number_of_oscillators))
                        index_second = feature_names.index("x" + str(i) + ' x' + str(j + number_of_oscillators))
                        index_first_transformed = index_first + + int(len(feature_names)*i)  
                        index_second_transformed = index_second + int(len(feature_names)*j)
                        symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                        pair_symmetries.append(symmetry_pair)
    for symmetry in type_symmetries:
        if(symmetry == ('cos')):
            #Symmetry cos
            for i in range(1, number_of_oscillators):
                index_first = feature_names.index("x" + str(0))
                index_second = feature_names.index("x" + str(i))
                index_first_transformed = index_first 
                index_second_transformed = index_second + int(len(feature_names)*i)    
                symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                pair_symmetries.append(symmetry_pair)
    for symmetry in type_symmetries:
        if(symmetry == ('sin')):
            #Symmetry sin
            for i in range(1, number_of_oscillators):
                index_first = feature_names.index("x" + str(number_of_oscillators + 0))
                index_second = feature_names.index("x" + str(number_of_oscillators + i))
                index_first_transformed = index_first 
                index_second_transformed = index_second + int(len(feature_names)*i)    
                symmetry_pair = (min(index_first_transformed, index_second_transformed), max(index_first_transformed, index_second_transformed))
                pair_symmetries.append(symmetry_pair)
    #Remove duplicates before we output.
    pair_symmetries = list(set([i for i in pair_symmetries])) 
    return pair_symmetries

def fermi_pasta_ulam_random(number_of_oscillators, number_of_snapshots, beta=0.7, x_min = - 0.1, x_max = 0.1):
    """Fermi–Pasta–Ulam problem.

    Generate data for the Fermi–Pasta–Ulam problem represented by the differential equation

        d^2/dt^2 x_i = (x_i+1 - 2x_i + x_i-1) + beta((x_i+1 - x_i)^3 - (x_i-x_i-1)^3).

    See [1]_ for details.

    Parameters
    ----------
    number_of_oscillators: int
        number of oscillators
    number_of_snapshots: int
        number of snapshots

    Returns
    -------
    snapshots: ndarray(number_of_oscillators, number_of_snapshots)
        snapshot matrix containing random displacements of the oscillators in [-0.1,0.1]
    derivatives: ndarray(number_of_oscillators, number_of_snapshots)
        matrix containing the corresponding derivatives

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # define random snapshot matrix
    snapshots = (x_max - x_min) * (np.random.rand(number_of_oscillators, number_of_snapshots) - 0.5) + (x_max + x_min)/2.0

    # compute derivatives
    derivatives = np.zeros((number_of_oscillators, number_of_snapshots))
    for j in range(number_of_snapshots):
        derivatives[0, j] = (
            snapshots[1, j]
            - 2 * snapshots[0, j]
            + beta * ((snapshots[1, j] - snapshots[0, j]) ** 3 - snapshots[0, j] ** 3)
        )
        for i in range(1, number_of_oscillators - 1):
            derivatives[i, j] = (
                snapshots[i + 1, j]
                - 2 * snapshots[i, j]
                + snapshots[i - 1, j]
                + beta
                * (
                    (snapshots[i + 1, j] - snapshots[i, j]) ** 3
                    - (snapshots[i, j] - snapshots[i - 1, j]) ** 3
                )
            )
        derivatives[-1, j] = (
            -2 * snapshots[-1, j]
            + snapshots[-2, j]
            + beta
            * (-snapshots[-1, j] ** 3 - (snapshots[-1, j] - snapshots[-2, j]) ** 3)
        )

    return snapshots, derivatives




def fermi_pasta_ulam_time(number_of_oscillators, exact_solution, polinomial, max_time, number_of_snapshots, number_of_experiments, relative_tolerance = 1.0e-10, absolute_tolerance = 1.0e-13):
    from scipy.integrate import solve_ivp

    # Plot the exact trajectory.
    def fun_exact(t, y):
        return np.dot(
            exact_solution, polinomial.fit_transform(y.reshape(1, -1)).T
        ).squeeze()

    list_snapshots = []
    list_derivatives = []
    list_times = []
    for i in range(number_of_experiments):
        initial_position =  np.random.uniform(-0.1, 1.0, size=number_of_oscillators)
        sol = solve_ivp(fun_exact, [0, max_time], initial_position, method='DOP853', t_eval=np.linspace(0, max_time, number_of_snapshots), rtol = relative_tolerance, atol = absolute_tolerance)
        snapshots = sol.y
        derivatives = np.zeros([number_of_oscillators, number_of_snapshots])
        for i in range(number_of_snapshots):
            derivatives[:, i] = fun_exact(0, snapshots[:, i])
            
        list_derivatives.append(derivatives)
        list_times.append(sol['t'])
        list_snapshots.append(snapshots)
    return list_snapshots, list_derivatives, list_times

def fermi_pasta_ulam_time_correct(number_of_oscillators, exact_solution, polinomial, max_time, number_of_snapshots, number_of_experiments, relative_tolerance = 1.0e-10, absolute_tolerance = 1.0e-13):
    from scipy.integrate import solve_ivp


    # Plot the exact trajectory.
    # Assume that the 
    def fun_exact(t, y):
        position = y[:number_of_oscillators]
        velocity = y[number_of_oscillators:]
        return np.append(velocity, np.dot(
            exact_solution, polinomial.fit_transform(position.reshape(1, -1)).T
        ).squeeze())

    list_snapshots = []
    list_derivatives = []
    list_second_derivatives = []
    list_times = []
    for i in range(number_of_experiments):
        initial_position =  np.random.uniform(-0.1, 0.1, size=number_of_oscillators)
        initial_velocity =  np.zeros(number_of_oscillators)
        vector = np.append(initial_position, initial_velocity)
        sol = solve_ivp(fun_exact, [0, max_time], vector, method='DOP853', t_eval=np.linspace(0, max_time, number_of_snapshots), rtol = relative_tolerance, atol = absolute_tolerance)
        snapshots = sol.y
        second_derivatives = np.zeros([number_of_oscillators, number_of_snapshots])
        for i in range(number_of_snapshots):
            second_derivatives[:, i] = fun_exact(0, snapshots[:number_of_oscillators, i])
        list_second_derivatives.append(second_derivatives)
        list_derivatives.append(snapshots[number_of_oscillators:, :])
        list_snapshots.append(snapshots[:number_of_oscillators, :])
        list_times.append(sol['t'])
    return list_snapshots, list_derivatives, list_second_derivatives, list_times

# def kuramoto_time_individual(frequencies, initial_position, start_time, end_time, number_of_samples, K = 2.0, forcing_param = 0.2, relative_tolerance = 1.0e-10, absolute_tolerance = 1.0e-13):
#     from scipy.integrate import solve_ivp

#     number_of_oscillators = len(frequencies)

#     def kuramoto_ode(_, theta):
#         [theta_i, theta_j] = np.meshgrid(theta, theta)
#         return frequencies + K / number_of_oscillators * np.sin(theta_j - theta_i).sum(0) + forcing_param * np.sin(theta)
    
#     sol = solve_ivp(kuramoto_ode, [start_time, end_time], initial_position, method='DOP853', t_eval=np.linspace(start_time, end_time, number_of_samples), rtol = relative_tolerance, atol = absolute_tolerance)
#     return sol.y, sol['t']

# # Perform an experiment where we let the system develop from an initial position.
# def fermi_pasta_ulam_time_individual(
#     exact_solution, polinomial, initial_position, t_min, t_max, num_steps
# ):
#     # Plot the exact trajectory.
#     def fun_exact(t, y):
#         return np.dot(
#             exact_solution, polinomial.fit_transform(y.reshape(1, -1)).T
#         ).squeeze()

#     from scipy.integrate import solve_ivp

#     sol_true = solve_ivp(
#         fun_exact,
#         [t_min, t_max],
#         initial_position,
#         t_eval=np.linspace(t_min, t_max, num_steps),
#         vectorized=False,
#     )
#     assert (
#         sol_true["status"] == 0
#     ), "The integration of the initial value solver was not succesfull."
#     return sol_true.y, sol_true['t']

# Perform an experiment where we let the system develop from an initial position.
def fermi_pasta_ulam_time_individual(
    exact_solution, polinomial, initial_position, t_min, t_max, num_steps
):
    # Plot the exact trajectory.
    def fun_exact(t, y):
        return np.dot(
            exact_solution, polinomial.fit_transform(y.reshape(1, -1)).T
        ).squeeze()

    from scipy.integrate import solve_ivp

    sol_true = solve_ivp(
        fun_exact,
        [t_min, t_max],
        initial_position,
        t_eval=np.linspace(t_min, t_max, num_steps),
        vectorized=False,
    )
    assert (
        sol_true["status"] == 0
    ), "The integration of the initial value solver was not succesfull."
    return sol_true.y, sol_true['t']

# Perform an experiment where we let the system develop from an initial position.
def fermi_pasta_ulam_time_individual_correct(
    exact_solution, polinomial, initial_position, initial_velocity, t_min, t_max, num_steps
):
    number_of_oscillators = exact_solution.shape[0]
    # Plot the exact trajectory.
    # Assume that the 
    def fun_exact(t, y):
        position = y[:number_of_oscillators]
        velocity = y[number_of_oscillators:]
        return np.append(velocity, np.dot(
            exact_solution, polinomial.fit_transform(position.reshape(1, -1)).T
        ).squeeze())
    from scipy.integrate import solve_ivp
    position_val = np.append(initial_position, initial_velocity)
    sol_true = solve_ivp(
        fun_exact,
        [t_min, t_max],
        position_val,
        t_eval=np.linspace(t_min, t_max, num_steps),
        vectorized=False,
    )
    assert (
        sol_true["status"] == 0
    ), "The integration of the initial value solver was not succesfull."
    
    return sol_true.y[:number_of_oscillators], sol_true.y[number_of_oscillators:], sol_true['t']



# Build the exact dynamics of the fermi_pasta_ulam if we have polynomials of order up to three.
def exact_solution_fermi_pasta_ulam(number_of_oscillators, polinomial, beta):
    reference_polynomial = polinomial.get_feature_names()
    num_basis_functions = len(reference_polynomial)
    # Build the exact solution at the boundary
    exact_solution = np.zeros((number_of_oscillators, num_basis_functions))
    # First order terms.
    exact_solution[0, reference_polynomial.index("x0")] = -2
    exact_solution[0, reference_polynomial.index("x1")] = 1
    # Third order terms
    exact_solution[0, reference_polynomial.index("x0^3")] = -2 * beta
    exact_solution[0, reference_polynomial.index("x1^3")] = beta
    exact_solution[0, reference_polynomial.index("x0^2 x1")] = 3 * beta
    exact_solution[0, reference_polynomial.index("x0 x1^2")] = -3 * beta
    # Build the exact solution in the interior.
    for i in range(1, number_of_oscillators - 1):
        # First order terms.
        exact_solution[i, reference_polynomial.index("x" + str(i))] = -2
        exact_solution[i, reference_polynomial.index("x" + str(i - 1))] = 1
        exact_solution[i, reference_polynomial.index("x" + str(i + 1))] = 1
        # Third order terms
        exact_solution[i, reference_polynomial.index("x" + str(i) + "^3")] = -2 * beta
        exact_solution[i, reference_polynomial.index("x" + str(i - 1) + "^3")] = (
            1 * beta
        )
        exact_solution[i, reference_polynomial.index("x" + str(i + 1) + "^3")] = (
            1 * beta
        )
        exact_solution[
            i, reference_polynomial.index("x" + str(i) + "^2 x" + str(i + 1))
        ] = (3 * beta)
        exact_solution[
            i, reference_polynomial.index("x" + str(i) + " x" + str(i + 1) + "^2")
        ] = (-3 * beta)
        exact_solution[
            i, reference_polynomial.index("x" + str(i - 1) + " x" + str(i) + "^2")
        ] = (3 * beta)
        exact_solution[
            i, reference_polynomial.index("x" + str(i - 1) + "^2 x" + str(i))
        ] = (-3 * beta)
    # Equation for the end point.
    exact_solution[
        number_of_oscillators - 1,
        reference_polynomial.index("x" + str(number_of_oscillators - 1)),
    ] = -2
    exact_solution[
        number_of_oscillators - 1,
        reference_polynomial.index("x" + str(number_of_oscillators - 2)),
    ] = 1
    # Third order terms
    exact_solution[
        number_of_oscillators - 1,
        reference_polynomial.index("x" + str(number_of_oscillators - 1) + "^3"),
    ] = (-2 * beta)
    exact_solution[
        number_of_oscillators - 1,
        reference_polynomial.index("x" + str(number_of_oscillators - 2) + "^3"),
    ] = beta
    exact_solution[
        number_of_oscillators - 1,
        reference_polynomial.index(
            "x"
            + str(number_of_oscillators - 2)
            + "^2 x"
            + str(number_of_oscillators - 1)
        ),
    ] = (-3 * beta)
    exact_solution[
        number_of_oscillators - 1,
        reference_polynomial.index(
            "x"
            + str(number_of_oscillators - 2)
            + " x"
            + str(number_of_oscillators - 1)
            + "^2"
        ),
    ] = (3 * beta)
    return exact_solution

def brusselator_time(initial_position, t_min = 0.0, t_max = 10.0, num_steps = 1000, r_coefficients = [1, 3, 1, 1]):
#% rate constants:
#r1 = 1; % 0 -> A
#r2 = 3; % A -> B
#r3 = 1; % 2A + B -> 3A
#r4 = 1; % A -> 0
    r1, r2, r3, r4 = r_coefficients
    def fun_exact(t, y):
        return np.array([r1-r2*y[0]+r3*y[0]**2*y[1]-r4*y[0], r2*y[0]-r3*y[0]**2*y[1]])
    
    from scipy.integrate import solve_ivp

    sol_true = solve_ivp(
        fun_exact,
        [0, t_max],
        initial_position,
        t_eval=np.linspace(t_min, t_max, num_steps),
        vectorized=False,
    )
    assert (
        sol_true["status"] == 0
    ), "The integration of the initial value solver was not succesfull."
    return sol_true, fun_exact(sol_true['t'], sol_true['y'])

# Build the exact dynamics of the fermi_pasta_ulam if we have polynomials of order up to three.
def exact_solution_brusselator(dimension, polinomial, r_coefficients):
    r1, r2, r3, r4 = r_coefficients
    reference_polynomial = polinomial.get_feature_names()
    num_basis_functions = len(reference_polynomial)
    # Build the exact solution at the boundary
    exact_solution = np.zeros((dimension, num_basis_functions))
    # Solution for the first species
    exact_solution[0, reference_polynomial.index("1")] = r1
    exact_solution[0, reference_polynomial.index("x0")] = - r2 - r4
    exact_solution[0, reference_polynomial.index("x0")] = - r2 - r4
    exact_solution[0, reference_polynomial.index("x0^2 x1")] = r3
    # Solution for the first species
    exact_solution[1, reference_polynomial.index("x0")] = r2
    exact_solution[1, reference_polynomial.index("x0^2 x1")] = -r3
    return exact_solution

def lutka_volterra_time(initial_position, t_min = 0.0, t_max = 10.0, num_steps = 1000, r_coefficients = [1, 1, 1, 1]):
#% rate constants:
#r1 = 1; % reproduction of prey: A -> 2A
#r2 = 1; % death of predator: B -> 0
#r3 = 1; % consumption: A + B -> B
#r4 = 1; % reproduction of predator: A + B -> A + 2B
    r1, r2, r3, r4 = r_coefficients
    def fun_exact(t, y):
        return np.array([y[0]*(r1-r3*y[1]), -y[1]*(r2-r4*y[0])])
    from scipy.integrate import solve_ivp

    sol_true = solve_ivp(
        fun_exact,
        [0, t_max],
        initial_position,
        t_eval=np.linspace(t_min, t_max, num_steps),
        vectorized=False,
    )
    assert (
        sol_true["status"] == 0
    ), "The integration of the initial value solver was not succesfull."
    return sol_true, fun_exact(sol_true['t'], sol_true['y'])

# Build the exact dynamics of the fermi_pasta_ulam if we have polynomials of order up to three.
def exact_solution_lutka_volterra(dimension, polinomial, r_coefficients):
    r1, r2, r3, r4 = r_coefficients
    reference_polynomial = polinomial.get_feature_names()
    num_basis_functions = len(reference_polynomial)
    # Build the exact solution at the boundary
    exact_solution = np.zeros((dimension, num_basis_functions))
    # Solution for the first species
    exact_solution[0, reference_polynomial.index("x0")] = r1
    exact_solution[0, reference_polynomial.index("x0 x1")] = - r3
    # Solution for the first species
    exact_solution[1, reference_polynomial.index("x1")] = - r2
    exact_solution[1, reference_polynomial.index("x0 x1")] = r4
    return exact_solution

def michaelis_menten_time(dimension, number_of_snapshots, number_of_experiments = 10, t_min = 0.0, t_max = 10.0, coefficients = [0.01, 1, 1]):
#Expressions
#d/dt C_1 = - k_{1}*C_{1}*C_{2} + k_{-1}* C_{3}
#d/dt C_2 = - k_{1}*C_{1}*C_{2} + (k_{-1} + k_{2})* C_{3}
#d/dt C_3 = k_{1}*C_{1}*C_{2} - (k_{-1} + k_{2})* C_{3}
#d/dt C_4 =  k_{2}*C_{3}
    # initial_position = np.array([1.0, 0.7, 0.0, 0.0])
    k_1, k_2, k_minus1 = coefficients
    def fun_exact(t, y):
        return np.array([-k_1*y[0]*y[1] + k_minus1*y[2], -k_1*y[0]*y[1] + (k_minus1 + k_2)*y[2], k_1*y[0]*y[1] - (k_minus1 + k_2)*y[2], k_2*y[2]])
    from scipy.integrate import solve_ivp
    
    list_snapshots = []
    list_derivatives = []
    list_times = []
    for i in range(number_of_experiments):
        initial_position = np.random.rand(4)
        sol = solve_ivp(
            fun_exact,
            [0, t_max],
            initial_position,
            t_eval=np.linspace(t_min, t_max, number_of_snapshots),
            vectorized=False,
        )
        assert (
            sol["status"] == 0
        ), "The integration of the initial value solver was not succesfull."
        snapshots = sol.y
        derivatives = np.zeros([dimension, number_of_snapshots])
        for i in range(number_of_snapshots):
            derivatives[:, i] = fun_exact(0, snapshots[:, i])
            
        list_derivatives.append(derivatives)
        list_times.append(sol['t'])
        list_snapshots.append(snapshots)
    return list_snapshots, list_derivatives, list_times


def michaelis_menten_time_individual(initial_position, t_min = 0.0, t_max = 10.0, num_steps = 1000, coefficients = [0.01, 1, 1]):
#Expressions
#d/dt C_1 = - k_{1}*C_{1}*C_{2} + k_{-1}* C_{3}
#d/dt C_2 = - k_{1}*C_{1}*C_{2} + (k_{-1} + k_{2})* C_{3}
#d/dt C_3 = k_{1}*C_{1}*C_{2} - (k_{-1} + k_{2})* C_{3}
#d/dt C_4 =  k_{2}*C_{3}
    k_1, k_2, k_minus1 = coefficients
    def fun_exact(t, y):
        return np.array([-k_1*y[0]*y[1] + k_minus1*y[2], -k_1*y[0]*y[1] + (k_minus1 + k_2)*y[2], k_1*y[0]*y[1] - (k_minus1 + k_2)*y[2], k_2*y[2]])
    from scipy.integrate import solve_ivp

    sol_true = solve_ivp(
        fun_exact,
        [0, t_max],
        initial_position,
        t_eval=np.linspace(t_min, t_max, num_steps),
        vectorized=False,
    )
    assert (
        sol_true["status"] == 0
    ), "The integration of the initial value solver was not succesfull."
    return sol_true, fun_exact(sol_true['t'], sol_true['y'])

# Build the exact dynamics of the fermi_pasta_ulam if we have polynomials of order up to three.
def exact_solution_michaelis_menten(dimension, polinomial, coefficients):
    assert dimension == 4, "The dimension of the michaelis-menten dynamic should be 4."
    k_1, k_2, k_minus1 = coefficients
    reference_polynomial = polinomial.get_feature_names()
    num_basis_functions = len(reference_polynomial)
    exact_solution = np.zeros((dimension, num_basis_functions))
    #First species.
    exact_solution[0, reference_polynomial.index("x0 x1")] = -k_1
    exact_solution[0, reference_polynomial.index("x2")] = k_minus1
    #Second species
    exact_solution[1, reference_polynomial.index("x0 x1")] = -k_1
    exact_solution[1, reference_polynomial.index("x2")] = k_minus1 + k_2
    #Third species
    exact_solution[2, reference_polynomial.index("x0 x1")] = k_1
    exact_solution[2, reference_polynomial.index("x2")] = -(k_minus1 + k_2)
    #Fourth species.
    exact_solution[3, reference_polynomial.index("x2")] = k_2
    return exact_solution

# Build the exact dynamics of the fermi_pasta_ulam if we have polynomials of order up to three.
def exact_solution_michaelis_menten_1D(polinomial, coefficients, initial_position):
    k_1, k_2, k_minus1 = coefficients
    reference_polynomial = polinomial.get_feature_names()
    num_basis_functions = len(reference_polynomial)
    exact_solution = np.zeros((dimension, num_basis_functions))
    #First species.
    exact_solution[0, reference_polynomial.index("x0 x1")] = -k_1
    exact_solution[0, reference_polynomial.index("x2")] = k_minus1
    #Second species
    exact_solution[1, reference_polynomial.index("x0 x1")] = -k_1
    exact_solution[1, reference_polynomial.index("x2")] = k_minus1 + k_2
    #Third species
    exact_solution[2, reference_polynomial.index("x0 x1")] = k_1
    exact_solution[2, reference_polynomial.index("x2")] = -(k_minus1 + k_2)
    #Fourth species.
    exact_solution[3, reference_polynomial.index("x2")] = k_2
    return exact_solution

#Add the constraints given by:
#d/dt (C_{2} + C_{3}) = 0
#d/dt (C_{1} + C_{3} + C_{4}) + 0
def add_constraints_michaelis_menten_easy(polinomial):
    feature_names = polinomial.get_feature_names()
    list_constraints = []
    for i in range(len(feature_names)):
        list_constraints.append({"x" + str(int(len(feature_names) + i)): 1.0, "x" + str(int(2.0*len(feature_names) + i)): 1.0, "constant": 0.0})
        list_constraints.append({"x" + str(i): 1.0, "x" + str(int(2.0*len(feature_names) + i)): 1.0,"x" + str(int(3.0*len(feature_names) + i)): 1.0 , "constant": 0.0})
    return list_constraints

def add_constraints_michaelis_menten_hard(polinomial, data, normalization_factors, epsilon):
    feature_names = polinomial.get_feature_names()
    num_data_points = data.shape[1]
    list_constraints = []
    #Four constraints per datapoint
    for j in range(num_data_points):
        constraint_dictionary = {}
        constraint_dictionary2 = {}
        constraint_dictionary3 = {}
        constraint_dictionary4 = {}
        for i in range(len(feature_names)):
            #First symmetry. One side of abs val.
            constraint_dictionary["x" + str(int(len(feature_names) + i))] = data[i, j]/normalization_factors[i][0]
            constraint_dictionary["x" + str(int(2.0*len(feature_names) + i))] = data[i, j]/normalization_factors[i][0]
            constraint_dictionary["constant"] = epsilon
            #First symmetry. Other side of abs val.
            constraint_dictionary2["x" + str(int(len(feature_names) + i))] = -data[i, j]/normalization_factors[i][0]
            constraint_dictionary2["x" + str(int(2.0*len(feature_names) + i))] = -data[i, j]/normalization_factors[i][0]
            constraint_dictionary2["constant"] = epsilon
            #Second symmetry. One side of abs val.
            constraint_dictionary3["x" + str(i)] = data[i, j]/normalization_factors[i][0]
            constraint_dictionary3["x" + str(int(2.0*len(feature_names) + i))] = data[i, j]/normalization_factors[i][0]
            constraint_dictionary3["x" + str(int(3.0*len(feature_names) + i))] = data[i, j]/normalization_factors[i][0]
            constraint_dictionary3["constant"] = epsilon
            #Second symmetry. Other side of abs val.
            constraint_dictionary4["x" + str(i)] = -data[i, j]/normalization_factors[i][0]
            constraint_dictionary4["x" + str(int(2.0*len(feature_names) + i))] = -data[i, j]/normalization_factors[i][0]
            constraint_dictionary4["x" + str(int(3.0*len(feature_names) + i))] = -data[i, j]/normalization_factors[i][0]
            constraint_dictionary4["constant"] = epsilon
        list_constraints.append(constraint_dictionary)
        list_constraints.append(constraint_dictionary2)
        list_constraints.append(constraint_dictionary3)
        list_constraints.append(constraint_dictionary4)
    return list_constraints

def simulate_dynamics(basis, dynamic, initial_position, t_max, num_steps = 1000):
    # Plot the exact trajectory.
    def fun_dynamic(t, y):
        return np.dot(
            dynamic, basis.fit_transform(y.reshape(1, -1)).T
        ).squeeze()
    
    from scipy.integrate import solve_ivp
    t_val = np.linspace(0, t_max, num_steps)
    sol_true = solve_ivp(fun_dynamic,[0, t_max], initial_position, t_eval= np.linspace(0.0, t_max, num_steps), vectorized=False)
    return sol_true['y'], t_val
    
def simulate_dynamics_kuramoto(basis, dynamic, initial_position, t_max, num_steps = 1000):
    # Plot the exact trajectory.
    def fun_dynamic(t, y):
        y_transformed = np.vstack(
            (np.cos(y), np.sin(y))
            )
        return np.dot(
            dynamic, basis.fit_transform(y_transformed.reshape(1, -1)).T
        ).squeeze()
    
    from scipy.integrate import solve_ivp
    t_val = np.linspace(0, t_max, num_steps)
    sol_true = solve_ivp(fun_dynamic,[0, t_max], initial_position, t_eval= np.linspace(0.0, t_max, num_steps), vectorized=False)
    return sol_true['y'], t_val