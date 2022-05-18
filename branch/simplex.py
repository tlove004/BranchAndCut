# #####################################################################################################################
# ###################################################               ###################################################
# ################################################# The Simplex Method ################################################
# ###################################################               ###################################################
# Author: Tyson Loveless
# CS 220: Synthesis of Digital Systems, Fall 2017
# Date: October 9 2017
#
# The simplex method is a linear programming(LP) solver for maximizing a LP problem in standard form, that is:
# We wish to maximize some c^T*x subject to A*x = b, with x >= 0
# Note that c^T is a coefficient matrix on variables x, and A is the constraints matrix.
# This program implements the simplex method to solve a given LP problem.
# ###################################################               ###################################################
# #####################################################################################################################
import numpy as np
import helper as help

global inf, the_tableau
inf = (-1, -1)
# #####################################################################################################################
# ###################################################               ###################################################
# ##############################################       Is Optimal       ###############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Checks if bottom row of tableau has any negative values (not including z), if so, returns false, otherwise true
def is_optimal(tableau):
    n, m = tableau.shape
    check = tableau[n-1]
    for val in range(0, m-1):
        if check[val] < 0:
            return False
    return True


# #####################################################################################################################
# ###################################################               ###################################################
# ##############################################       Is Feasible       ##############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Given a solution to simplex method, checks if the solution meets all constraints
def is_feasible(tableau):
    n, m = tableau.shape
    b = tableau[:, m-1]
    for val in b:
        if val < 0:
            help.write_to_file("bounded-infeasible")
            exit(1)
            return False
    return True


# #####################################################################################################################
# ###################################################               ###################################################
# #################################################       Pivot       #################################################
# ###################################################               ###################################################
# #####################################################################################################################
# Pivots a basic variable into the the basis by finding a pivot col/row then pivots new var by Gauss-Jordan elimination
def do_pivot(tableau, pos):
    n, m = tableau.shape
    if pos == (-1, -1):

        # want to pivot at column with lowest value on bottom row
        check = tableau[n-1]
        _min = 0
        col = 0     # the column index
        for index in range(0, m):
            if check[index] < _min:
                col = index
                _min = check[index]

        # check ratios to find which row to use in pivot
        # if no valid ratios, then, we have an unbounded solution, as we are maximizing
        numerator = tableau[:, m-1]
        denominator = tableau[:, col]
        _min = float("inf")
        row = 0
        valid = False
        for index in range(0, n):
            if denominator[index] > 0:
                check = numerator[index] / denominator[index]
                if check >= 0:
                    valid = True
                    if check < _min:
                        _min = check
                        row = index
        if not valid:
            help.write_to_file("+inf\n")
            exit(2)

        # performs Gauss-Jordan elimination around the pivot row/column
        pivot = tableau[row][col]
    else:
        row = pos[0]
        col = pos[1]
        pivot = tableau[row][col]
        #col = np.where(tableau[row] == pivot)[0][0]

    newRow = []
    newTableau = tableau.copy()

    if pivot == 1:
        newRow = newTableau[row]
    else:
        for b in range(0, m):
            newRow.append(tableau[row][b] / pivot)

    newTableau[row] = newRow

    base = newTableau[row]
    for index in range(0, n):
        if index == row:
            continue
        replace = tableau[index][col]
        if replace == 0:
            continue
        coeff = replace / base[col]
        newRow = []
        for j in range (0, m):
            newRow.append(tableau[index][j] - coeff*base[j])
        newTableau[index] = newRow

    return newTableau


# #####################################################################################################################
# ###################################################               ###################################################
# #################################################       Solve       #################################################
# ###################################################               ###################################################
# #####################################################################################################################
# Solves a LP problem represented as a simplex tableau using the simplex method
def solve(tableau, pos, o_tableau):
    global the_tableau
    the_tableau = o_tableau
    if pos == (-1, -1):
        if is_optimal(tableau):
            if is_feasible(tableau):
                n, m = tableau.shape
                z = tableau[n-1][m-1]
                return z, tableau, solution(z, tableau)
            else:
                return two_phase(tableau, the_tableau[the_tableau.shape[0]-1])
        else:
            return solve(do_pivot(tableau, inf), inf, the_tableau)
    else:
        return solve(do_pivot(tableau, pos), inf, the_tableau)


# #####################################################################################################################
# ###################################################               ###################################################
# ###############################################       Solution       ################################################
# ###################################################               ###################################################
# #####################################################################################################################
# Returns a string containing solution vector for all x in basis
def solution(z, tableau):
    output = ""

    n, m = tableau.shape
    row = 0
    for j in range(0, m - 1):
        basic = False
        for i in range(0, n - 1):
            if tableau[i][j] != 0:
                if tableau[i][j] != 1:
                    basic = False
                    break
                elif basic is False:
                    row = i
                    basic = True
                else:
                    basic = False
                    break
        if basic:
            if help.isclose(tableau[row][m - 1], int(round(tableau[row][m - 1], 10))):
                output += str(int(round(tableau[row][m - 1], 10)))
            else:
                output += str(tableau[row][m - 1])
            output += "\n"
        else:
            output += "0\n"

    return output


# #####################################################################################################################
# ###################################################               ###################################################
# ###############################################       phase_one       ###############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Solves the first phase of two phase simplex method. used when there is negative solution in basis after cutting
def phase_one(tableau):
    new_tableau = tableau.copy()
    n = 0
    while np.any(new_tableau[:new_tableau.shape[0]-1, new_tableau.shape[1]-1] < 0):
        n += 1
        row_to_replace = np.where(new_tableau.T[new_tableau.shape[1]-1] < 0)[0][0]

        # invert row that has negative solution in basis
        for index in range(0, new_tableau.shape[1]):
            val = new_tableau[row_to_replace][index]
            if val != 0:
                new_tableau[row_to_replace][index] = -val

        # introduce new slack variable
        new_slack = np.array(np.zeros(new_tableau.shape[0]))
        new_slack[row_to_replace] = 1
        new_tableau = np.insert(new_tableau, new_tableau.shape[1]-1, new_slack, 1)

        # update bottom row
        for col_index in range(0, new_tableau.shape[1]):
            sum = 0
            for row_index in range(0, new_tableau.shape[0]-1):
                sum += new_tableau[row_index][col_index]
            new_tableau[new_tableau.shape[0]-1][col_index] = new_tableau[new_tableau.shape[0]-1][col_index] - sum

    return solve(new_tableau, inf, the_tableau)[1], n


# #####################################################################################################################
# ###################################################               ###################################################
# ###############################################       phase_two       ###############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Solves the second phase of two phase simplex method. removes excess slack variables and solves using simplex method
def phase_two(objective_function, p1_tableau, n):
    new_tableau = p1_tableau.copy()

    # delete rows corresponding to extra slack vars from phase one
    for i in range(0, n):
        new_tableau = np.delete(new_tableau, new_tableau.shape[1]-2, 1)

    objective_function = np.append(objective_function, np.zeros(new_tableau.shape[1]-objective_function.shape[0]))
    new_tableau[new_tableau.shape[0]-1] = objective_function

    return solve(new_tableau, inf, the_tableau)


# #####################################################################################################################
# ###################################################               ###################################################
# ###############################################       two_phase       ###############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Implements the two-phase simplex method to deal with negative solutions in a basic solution
def two_phase(tableau, objective_function):

    phase_one_tableau, n = phase_one(tableau)

    return phase_two(objective_function, phase_one_tableau, n)
