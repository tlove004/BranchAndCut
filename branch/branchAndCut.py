# #####################################################################################################################
# ###################################################               ###################################################
# ############################################ Branch and Cut ILP Solver ##############################################
# ###################################################               ###################################################
# Author: Tyson Loveless
# CS 220: Synthesis of Digital Systems, Fall 2017
# Date: October 16 2017
#
# Branch and cut is a method for solving integer linear programming(ILP) problems.  It maximizing an ILP problem in
# standard form, that is:
# We wish to maximize some c^T*x subject to A*x <= b, with x >= 0, b >= 0
# Note that c^T is a coefficient matrix on variables x, and A is the constraints matrix.
# The difference between an ILP and an LP is that the vector x for an ILP is restricted to only integer solutions
# This program implements the branch and cut method to solve a given ILP problem.  It calls the simplex method LP solver
#    as a subroutine
# The choice of when to cut or branch is based on the amount the object solution is changed at each iteration
#    if the heuristic is met, then the algorithm will choose to cut every time, if it is not making progress with cuts,
#    then it will branch and reset the heuristic
# I am using a combined heuristic of meeting a certain differential on the objective solution between cuts as well as a
#    counter to record the number of times an insufficient cut has been made in a row.
# ###################################################               ###################################################
# #####################################################################################################################
import helper as help
import numpy as np
import simplex
import math
from fractions import Fraction as f

global A, b, cT, o_tableau

global DEBUG
DEBUG = False


# #####################################################################################################################
# ###################################################               ###################################################
# ##############################################       Get Cut Row      ###############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Gets a row to cut (from a non-integral basic solution)
def get_cut_rows(tableau, solution):
    sol_array = []
    for line in solution.split('\n'):
        sol_array.append(line)
    sol_array = np.asarray(sol_array)

    rows = []

    for var in range(0, cT.shape[0]):
        if not help.RepresentsInt(sol_array[var]):
            # find which row this belongs to
            row = np.where(tableau[:, var]==1)
            n = row[0][0]
            row = tableau[n].copy()
            rows.append(row)

    return rows


# #####################################################################################################################
# ###################################################               ###################################################
# ##############################################       Get Cut Row      ###############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Gets a row to cut (from a non-integral basic solution)
def get_cut_row(tableau, solution):
    sol_array = []
    for line in solution.split('\n'):
        sol_array.append(line)
    sol_array = np.asarray(sol_array)

    for var in range(0, cT.shape[0]):
        if not help.RepresentsInt(sol_array[var]):
            # find which row this belongs to
            row = np.where(tableau[:, var]==1)
            n = row[0][0]
            m = var
            row = tableau[n].copy()
            break

    return n, m, row


# #####################################################################################################################
# ###################################################               ###################################################
# ##############################################       Gomory Cut       ###############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Modifies constraint on Ax = b to remove known infeasible solutions
# takes floor of all values in a non-integral solution row and subtracts this from the row, then updates constraint row
# to only constrain the object function
def gomoryCuts(table, relaxed_table, solution):

    rows = get_cut_rows(relaxed_table, solution) # n has row indices, m are columns, rows are the rows
    constraints = []

    while rows:
        row = rows.pop()
        rounded = row.copy()

        for coeff in range(0, relaxed_table.shape[1]):
            if row[coeff] >= 0:
                rounded[coeff] = math.floor(row[coeff])
            else:
                rounded[coeff] = math.floor(row[coeff])
            row[coeff] = row[coeff]-rounded[coeff]
        constraints.append(update_constraint(row, table))

    return constraints


# #####################################################################################################################
# ###################################################               ###################################################
# ###################################       Checks if solution is integral       ######################################
# ###################################################               ###################################################
# #####################################################################################################################
# Checks our solution to make sure values are integral
def isIntegral(solution, tableau):
    if not simplex.is_optimal(tableau):
        return False
    # if not simplex.is_feasible(tableau):
    #     return False
    sol_array = []
    for line in solution.split('\n'):
        sol_array.append(line)
    sol_array = np.asarray(sol_array)

    for res in range(0, cT.shape[0]):
        if not help.RepresentsInt(sol_array[res]):
            return False

    return True


# #####################################################################################################################
# ###################################################               ###################################################
# ##############################################    update_constraint     #############################################
# ###################################################               ###################################################
# #####################################################################################################################
# Performs pivots on slack variables to transform a constraint to only affect values in cT
def update_constraint(constraint, table):
    table = np.insert(table, table.shape[0] - 1, constraint, 0)

    #pivot to bring objective function variables into basis
    for index in range(cT.shape[0], constraint.shape[0] - 1):
        if constraint[index] != 0:
            table = simplex.do_pivot(table, (index - cT.shape[0], index))

    #extract constraint from pivoting table
    constraint = table[table.shape[0] - 2]

    # negate values to flip inequality
    for index in range(0, constraint.shape[0]):
        if constraint[index] != 0:
            constraint[index] = -constraint[index]

    #new slack variable added
    constraint = np.insert(constraint, constraint.shape[0] - 1, [1])

    # rounds floats to integers if they are close (within 10 digits)
    for index in range(0, constraint.shape[0]):
        if help.isclose(constraint[index], int(round(constraint[index], 10))) or round(constraint[index]) == 0:
            val = float(int(round(constraint[index], 10)))
            constraint[index] = val

    # checks if all constraints coefficients are integers, if so, we can take the floor of the b value
    allInt = True
    for index in range(0, A.shape[1]):
        if not constraint[index].is_integer():
            allInt = False
            break
    if allInt:
        constraint[constraint.shape[0] - 1] = math.floor(constraint[constraint.shape[0] - 1])

    return constraint


# #####################################################################################################################
# ###################################################               ###################################################
# ##################################################    good cut     ##################################################
# ###################################################               ###################################################
# #####################################################################################################################
# Checks if the cut is a feasible cut  (infeasible if b is negative, and requires dual simplex or two phase simplex to
#   solve)
def good_cut(constraint):

    if constraint[constraint.shape[0]-1] < 0:
        return False

    #checks if given constraint constrains objective function, or just slack variables (or nothing)
    for index in range(0, A.shape[1]):
        if constraint[index] != 0:
            return True

    return False


# #####################################################################################################################
# ###################################################               ###################################################
# ##################################################   preprocess     #################################################
# ###################################################               ###################################################
# #####################################################################################################################
# Takes in tableau and makes sure constraints are all integers by finding gcd of all values within rows and
# multiplying all values to transform them to whole numbers
def preprocess(table):
    global A
    for row in range(0, table.shape[0]-1):
        non_integers = []
        for col in range(0, table.shape[1]):
            if not help.isclose(float(table[row][col]), int(round(table[row][col], 10))):
                non_integers.append(table[row][col])
        while non_integers:
            if non_integers.__len__() > 1:
                a = f.from_float(non_integers.pop()).limit_denominator().denominator #first float denom
                b = f.from_float(non_integers.pop()).limit_denominator().denominator #second float denom
                divisor = 1.0/help.lcm(a, b)  #integer lcm
                non_integers.append(divisor)
            else:
                divisor = f.from_float(non_integers.pop()).limit_denominator().denominator
                for col in range(0, A.shape[1]):
                    table[row][col] = table[row][col]*divisor
                table[row][table.shape[1]-1] = table[row][table.shape[1]-1]*divisor

    m, n = A.shape
    A = np.matrix(table)
    A = A[0:A.shape[0]-1, 0:A.shape[1]-(n+1)]

    return table, A


# #####################################################################################################################
# ###################################################               ###################################################
# ##################################################     branch       #################################################
# ###################################################               ###################################################
# #####################################################################################################################
# branches an infeasible integer solution into constraints as an integer floor and ceiling on the basic variable
def branch(tableau, solution):
    #n is row, m is column
    n, m, row = get_cut_row(tableau, solution)

    low = math.floor(row[row.shape[0]-1])
    high = math.ceil(row[row.shape[0]-1])

    left = np.array(np.zeros(row.shape[0]))

    left[m] = 1

    left[left.shape[0]-1] = 1

    right = np.append(left, high)
    left = np.append(left, low)

    return left, right


def add_constraint(table, constraint):
    while constraint.shape[0] <= table.shape[1]:
        constraint = np.insert(constraint, constraint.shape[0]-2, np.zeros(1))
    table = np.insert(table, table.shape[1]-1, np.zeros(table.shape[0]), 1)
    table = np.insert(table, table.shape[0]-1, constraint, 0)
    return table


# #####################################################################################################################
# ###################################################               ###################################################
# #################################################   main function   #################################################
# ###################################################               ###################################################
# #####################################################################################################################
def main():
    global int1, int2, b, cT, A
    int1, int2, b, cT, A = help.read_input_file()

    global o_tableau
    o_tableau = help.make_tableau(cT, A, b)

    o_tableau, A = preprocess(o_tableau)

    # add table to list of unsolved tables:
    L = []
    L.append(o_tableau)
    # initialize solution vector
    x = []
    # initialize objective value
    z = float("-inf")
    v = i = improvement = 0
    # while L is not empty
    while L:
        i += 1
        change = v
        table = L.pop()
        relaxed_Table = simplex.solve(table, (-1, -1), o_tableau)[1]
        if not simplex.is_feasible(relaxed_Table):
            # should I put relaxed_Table into L?
            continue
        #store objective value
        v = relaxed_Table[relaxed_Table.shape[0] - 1][relaxed_Table.shape[1] - 1]

        #get solution vector
        solution = simplex.solution(v, relaxed_Table)
        sol_array = []
        for line in solution.split('\n'):
            sol_array.append(line)
        sol_array = np.asarray(sol_array)

        # if worst than best, continue
        if v <= z:
            continue

        # if valid and better than last, update solution vector and best so far
        if isIntegral(solution, relaxed_Table):
            x = sol_array
            z = v
            continue

        #if desired, search for cutting planes
        if abs(v - change) > 0.1:
            improvement = 0
            if DEBUG:
                print "cut:" + str(abs(v-change))
        else:
            improvement += 1
            if DEBUG:
                print "needs improvement: " + str(abs(v-change))
        cut = False
        if improvement < 2:
            constraints = gomoryCuts(table, relaxed_Table, solution)
            for constraint in constraints:
                if good_cut(constraint):
                    cut = True
                    break
        if cut:
            while constraints:
                constraint = constraints.pop()
                if DEBUG:
                    print constraint
                table = add_constraint(table, constraint)
            L.append(table)
            continue
        if DEBUG:
            print "branching now"
        improvement = 0
        # have not improved by cutting in a while, try branching
        # branch problem into new problems with constraints on basic variable, add both to L, continue
        left, right = branch(relaxed_Table, solution)
        table = np.insert(table, table.shape[1] - 1, np.zeros(table.shape[0]), 1)
        l_table = np.insert(table, table.shape[0] - 1, right, 0)
        r_table = np.insert(table, table.shape[0] - 1, right, 0)
        L.append(l_table)
        L.append(r_table)

    if DEBUG:
        print z
        print o_tableau
        print x

    if help.isclose(z, int(round(z, 10))):
        output = str(int(round(z, 10)))
    else:
        output = str(z)
    output += "\n"
    output += solution

    help.write_to_file(output)

    return 0

# run script
main()
