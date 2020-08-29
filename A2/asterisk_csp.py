# Look for #IMPLEMENT tags in this file.
'''
All models need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = asterisk_csp_model_1(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the asterisk puzzle.

1. asterisk_csp_model_1 (worth 20/100 marks)
    - A model of an Asterisk grid built using only 
      binary not-equal constraints

2. asterisk_csp_model_2 (worth 20/100 marks)
    - A model of an Asterisk grid built using only 9-ary 
      all-different constraints

'''
from cspbase import *
import itertools


def setup_model_1(ast_grid):

    '''

    set up model_1

    '''

    grid_size = len(ast_grid)
    csp = CSP("asterisk_csp_model_1")
    domain = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    variables = []
    for row in range(grid_size):
        variables.append([])
        for col in range(grid_size):
            if ast_grid[row][col] is not None:
                var = Variable('V{}{}'.format(row + 1, col + 1), [ast_grid[row][col]])
            else:
                var = Variable('V{}{}'.format(row + 1, col + 1), domain)
            csp.add_var(var)
            variables[row].append(var)
    return csp, variables


def setup_model_2(ast_grid):
    '''

    set up model_2
    '''

    grid_size = len(ast_grid)
    csp = CSP("asterisk_csp_model_2")
    domain = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    variables = []

    for row in range(grid_size):
        variables.append([])
        for col in range(grid_size):
            if ast_grid[row][col] is not None:
                var = Variable('V{}{}'.format(row + 1, col + 1), [ast_grid[row][col]])
            else:
                var = Variable('V{}{}'.format(row + 1, col + 1), domain)
            csp.add_var(var)
            variables[row].append(var)
    return csp, variables

def check_domain(ast_grid, variables):
    '''
    Check and update the domain based on given value
    '''
    grid_size = len(ast_grid)
    col_lst = []
    ten_houses = init_tenhouses()

    for row in range(grid_size):
        row_var = variables[row]
        col_var = []

        # check column
        for col in range(grid_size):
            cur_var = row_var[col]
            if cur_var.cur_domain_size() == 1:
                for var in row_var:
                    if var != cur_var and var.in_cur_domain(cur_var.cur_domain()[0]):
                        var.prune_value(cur_var.cur_domain()[0])
            col_var.append(variables[col][row])
        col_lst.append(col_var)

    # check row
    for cur_col in range(grid_size):
        col_var = col_lst[cur_col]

        for row in range(grid_size):
            cur_var = col_var[row]
            if cur_var.cur_domain_size() == 1:
                for var in col_var:
                    if var != cur_var and var.in_cur_domain(cur_var.cur_domain()[0]):
                        var.prune_value(cur_var.cur_domain()[0])

    # check house
    for house in ten_houses:
        for index in range(len(house)):
            cur_var = variables[house[index][0] - 1][house[index][1] - 1]
            if cur_var.cur_domain_size() == 1:
                for t in house:
                    var = variables[t[0] - 1][t[1] - 1]
                    if var != cur_var and var.in_cur_domain(cur_var.cur_domain()[0]):
                        var.prune_value(cur_var.cur_domain()[0])



def asterisk_check(dom1, dom2):
    '''
    check the first model constraints
    '''
    return dom1 != dom2


def asterisk_check2(tup, dom1):
    '''
    check the second model constraints
    '''
    for dom in tup:
        if dom == dom1:
            return False
    return True


def init_tenhouses():
    '''
    Initialize the tenth house
    '''
    ten_houses = []
    tenth_house = [(2, 5), (3, 3), (3, 7), (5, 2), (5, 5), (5, 8), (7, 3), (7, 7), (8, 5)]
    for i in range(1, 9, 3):
        for j in range(1, 9, 3):
            temp_lst = []
            for t in itertools.product(range(i, i + 3), range(j, j + 3)):
                temp_lst.append(t)
            ten_houses.append(temp_lst)
    ten_houses.append(tenth_house)

    return ten_houses



def asterisk_csp_model_1(ast_grid):
    ##IMPLEMENT

    # set up
    grid_size = len(ast_grid)
    ten_houses = init_tenhouses()
    csp, variables = setup_model_1(ast_grid)

    # update domain
    check_domain(ast_grid, variables)

    # check row constraints
    for row in range(grid_size):
        for col in range(grid_size):
            for row_gap in range(row + 1, grid_size):
                cur_var = variables[row][col]
                var_col = variables[row_gap][col]

                dom_cur = cur_var.cur_domain()
                dom_col = var_col.cur_domain()

                constraint = Constraint('C(V{}{}, V{}{})'.format(row + 1, col + 1,
                                                                 row_gap + 1, col + 1), [cur_var, var_col])
                sat_tuples = []
                for t in itertools.product(dom_cur, dom_col):
                    if asterisk_check(t[0], t[1]):
                        sat_tuples.append(t)
                constraint.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(constraint)

            # check column constraints
            for col_gap in range(col + 1, grid_size):
                cur_var = variables[row][col]
                var_row = variables[row][col_gap]

                dom_cur = cur_var.cur_domain()
                dom_row = var_row.cur_domain()

                constraint = Constraint('C(V{}{}, V{}{})'.format(row + 1, col + 1, row + 1,
                                                                 col_gap + 1), [cur_var, var_row])
                sat_tuples = []
                for t in itertools.product(dom_cur, dom_row):
                    if asterisk_check(t[0], t[1]):
                        sat_tuples.append(t)
                constraint.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(constraint)

            # check house constraints
            for house in ten_houses:
                if (row + 1, col + 1) in house:
                    index = 0
                    for ii in range(index, len(house)):
                        if (row + 1, col + 1) == house[ii]:
                            index = ii
                            break
                    for i in range(index + 1, len(house)):
                        (spec_row, spec_col) = house[i]
                        cur_var = variables[row][col]
                        var_spec = variables[spec_row - 1][spec_col - 1]

                        dom_cur = cur_var.cur_domain()
                        dom_spec = var_spec.cur_domain()

                        constraint = Constraint('C(V{}{}, V{}{})'.format(row + 1, col + 1, spec_row, spec_col),
                                                [cur_var, var_spec])
                        sat_tuples = []
                        for t in itertools.product(dom_cur, dom_spec):
                            if asterisk_check(t[0], t[1]):
                                sat_tuples.append(t)
                        constraint.add_satisfying_tuples(sat_tuples)
                        csp.add_constraint(constraint)

    return csp, variables


def asterisk_csp_model_2(ast_grid):
    ##IMPLEMENT

    # set up
    grid_size = len(ast_grid)
    csp, variables = setup_model_2(ast_grid)
    check_domain(ast_grid, variables)

    # filling in row and column
    for i in range(grid_size):
        row_var = variables[i]
        col_var = []
        row_dom = []
        col_dom = []

        for j in range(grid_size):
            row_dom.append(row_var[j].cur_domain())
            col_var.append(variables[j][i])
            col_dom.append(variables[j][i].cur_domain())

        con_row = Constraint("C(row{})".format(i), row_var)
        sat_tuples_row = []
        for t in itertools.product(*row_dom):
            test = True
            for index in range(9):
                if not asterisk_check2(t[0:index], t[index]):
                    test = False
                    break
            if test:
                sat_tuples_row.append(t)
        con_row.add_satisfying_tuples(sat_tuples_row)
        csp.add_constraint(con_row)

        con_col = Constraint("C(col{})".format(i), col_var)
        sat_tuples_col = []
        for t in itertools.product(*col_dom):
            test = True
            for index in range(9):
                if not asterisk_check2(t[0:index], t[index]):
                    test = False
                    break
            if test:
                sat_tuples_col.append(t)
        con_col.add_satisfying_tuples(sat_tuples_col)
        csp.add_constraint(con_col)

    # fill in houses
    ten_houses_variables = []
    ten_houses_domain = []
    tenth_house = [(2, 5), (3, 3), (3, 7), (5, 2), (5, 5), (5, 8), (7, 3), (7, 7), (8, 5)]
    for i in range(1, 9, 3):
        for j in range(1, 9, 3):
            temp_var = []
            temp_dom = []
            for t in itertools.product(range(i, i + 3), range(j, j + 3)):
                temp_var.append(variables[t[0] - 1][t[1] - 1])
                temp_dom.append(variables[t[0] - 1][t[1] - 1].cur_domain())
            ten_houses_variables.append(temp_var)
            ten_houses_domain.append(temp_dom)

    temp_var = []
    temp_dom = []
    for t in tenth_house:
        temp_var.append(variables[t[0] - 1][t[1] - 1])
        temp_dom.append(variables[t[0] - 1][t[1] - 1].cur_domain())
    ten_houses_variables.append(temp_var)
    ten_houses_domain.append(temp_dom)

    for i in range(len(ten_houses_variables)):
        con_spec = Constraint("C(house{})".format(i), ten_houses_variables[i])
        sat_tuples_spec = []
        for t in itertools.product(*ten_houses_domain[i]):
            test = True
            for index in range(9):
                if not asterisk_check2(t[0:index], t[index]):
                    test = False
                    break
            if test:
                sat_tuples_spec.append(t)
        con_spec.add_satisfying_tuples(sat_tuples_spec)
        csp.add_constraint(con_spec)

    return csp, variables


