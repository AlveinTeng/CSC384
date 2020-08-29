from cspbase import *
from asterisk_csp import *
from propagators import *


test_ord_mrv = True;
test_model = True;

grid_1 = [[None, 1, None, None, None, None, None, 6, None],
 [3, None, 9, None, None, None, 1, None, 5],
 [None, 8, None, 3, None, 5, None, 7, None],
 [None, None, 2, None, 7, None, 8, None, None],
 [None, None, None, 6, None, 8, None, None, None],
 [None, None, 8, None, 9, None, 2, None, None],
 [None, 2, None, 4, None, 1, None, 9, None],
 [9, None, 4, None, None, None, 6, None, 1],
 [None, 3, None, None, None, None, None, 8, None]]

answer_1 = [2,1,5,7,4,9,3,6,8,3,7,9,2,8,6,1,4,5,4,8,6,3,1,5,9,7,2,
1,9,2,5,7,4,8,3,6,5,4,3,6,2,8,7,1,9,7,6,8,1,9,3,2,5,4,
8,2,7,4,6,1,5,9,3,9,5,4,8,3,7,6,2,1,6,3,1,9,5,2,4,8,7]



grid_2 = [[None, None, None, 2, None, None, None, 1, None],
 [None, None, None, None, None, None, None, 7, 9],
 [None, None, None, 1, None, 7, None, None, 5],
 [None, 1, 4, 3, 6, None, None, None, None],
 [7, None, None, None, None, None, None, None, 2],
 [None, None, None, None, 1, 2, 4, 5, None],
 [3, None, None, 9, None, 4, None, None, None],
 [9, 7, None, None, None, None, None, None, None],
 [None, 6, None, None, None, 3, None, None, None]]

answer_2 = [5,4,7,2,9,6,3,1,8,1,2,3,4,5,8,6,7,9,6,8,9,1,3,7,2,4,5,
2,1,4,3,6,5,8,9,7,7,3,5,8,4,9,1,6,2,8,9,6,7,1,2,4,5,3,
3,5,1,9,2,4,7,8,6,9,7,2,6,8,1,5,3,4,4,6,8,5,7,3,9,2,1]


if __name__ == "__main__":

    if test_model:
        score = 1
        #1st model test
        # csp, var_array = asterisk_csp_model_1(grid_1)
        csp, var_array = asterisk_csp_model_2(grid_1)
        cons = csp.get_all_cons()
        bin_flag = True
        for c in cons:
            if (len(c.get_scope()) != 2):
                bin_flag = False
                print("Non binary constraint")
                break
        solver = BT(csp)
        # solver.trace_on()
        # solver.bt_search(prop_FC)
        solver.bt_search(prop_GAC)
        # solver.bt_search(prop_BT)
        sol = []
        for i in range(len(var_array)):
            for j in range(len(var_array)):
                sol.append(var_array[i][j].get_assigned_value())
        if (sol != answer_1):
            print("Wrong solution")
        if ((sol == answer_1) and not bin_flag):
            print("Passed first model test")
        else:
            print("Failed first model test")
        #2nd model test
        # csp, var_array = asterisk_csp_model_1(grid_2)
        csp, var_array = asterisk_csp_model_2(grid_2)
        cons = csp.get_all_cons()
        bin_flag = True
        for c in cons:
            if (len(c.get_scope()) != 2):
                bin_flag = False
                print("Non binary constraint")
                break
        solver = BT(csp)
        # solver.trace_on()
        solver.bt_search(prop_GAC)
        # solver.bt_search(prop_BT)
        sol = []
        for i in range(len(var_array)):
            for j in range(len(var_array)):
                sol.append(var_array[i][j].get_assigned_value())
        if (sol != answer_2):
            print("Wrong solution")
        if ((sol == answer_2) and not bin_flag):
            print("Passed second model test")
        else:
            print("Failed second model test")

    if test_ord_mrv:

        a = Variable('A', [1])
        b = Variable('B', [1])
        c = Variable('C', [1])
        d = Variable('D', [1])
        e = Variable('E', [1])

        simpleCSP = CSP("Simple", [a,b,c,d,e])

        count = 0
        for i in range(0,len(simpleCSP.vars)):
            simpleCSP.vars[count].add_domain_values(range(0, count))
            count += 1

        var = []
        var = ord_mrv(simpleCSP)

        if var:
            if((var.name) == simpleCSP.vars[0].name):
                print("Passed First Ord MRV Test")
            else:
                print("Failed First Ord MRV Test")
        else:
           print("No Variable Returned from Ord MRV")

        a = Variable('A', [1,2,3,4,5])
        b = Variable('B', [1,2,3,4])
        c = Variable('C', [1,2])
        d = Variable('D', [1,2,3])
        e = Variable('E', [1])

        simpleCSP = CSP("Simple", [a,b,c,d,e])

        var = []
        var = ord_mrv(simpleCSP)

        if var:
            if((var.name) == simpleCSP.vars[len(simpleCSP.vars)-1].name):
                print("Passed Second Ord MRV Test")
            else:
                print("Failed Second Ord MRV Test")
        else:
           print("No Variable Returned from Ord MRV")
