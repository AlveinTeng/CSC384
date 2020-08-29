import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.interpolate import PchipInterpolator, CubicSpline, lagrange
import time

years = np.linspace(1900,1980,9)
pops = np.array([76212168,
                92228496,
                106021537,
                123202624,
                132164569,
                151325798,
                179323175,
                203302031,
                226542199])

def vandermonde():
    """
    Heath Computer Problem 7.8 (a)

    find the condition number for each basis function and
    show the condition numbers in a table
    """
    # find the vandermond matrix of each basis function
    V1 = np.vander(years)
    V2 = np.vander(years - 1900)
    V3 = np.vander(years - 1940)
    V4 = np.vander((years - 1940) / 40)

    # compute the condition number with order infinity for each basis function
    cond1 = np.linalg.cond(V1, np.inf)
    cond2 = np.linalg.cond(V2, np.inf)
    cond3 = np.linalg.cond(V3, np.inf)
    cond4 = np.linalg.cond(V4, np.inf)

    # show the result in a table
    tb = PrettyTable()
    tb.title = "cond number comparison"
    tb.field_names = ["basis function", "conditioning"]
    tb.add_row(["basis func 1", "{:.3e}".format(cond1)])
    tb.add_row(["basis func 2", "{:.3e}".format(cond2)])
    tb.add_row(["basis func 3", "{:.3e}".format(cond3)])
    tb.add_row(["basis func 4", "{:.3e}".format(cond4)])

    # select the Vandermonde matrix to return
    # also generate the basis_func function on the data xs
    V, basis_func = None, None
    cond = min([cond1, cond2, cond3, cond4])
    if cond == cond1:
        V = V1
        basis_func = lambda x: x
    elif cond == cond2:
        V = V2
        basis_func = lambda x: x - 1900
    elif cond == cond3:
        V = V3
        basis_func = lambda x: x - 1940
    elif cond == cond4:
        V = V4
        basis_func = lambda x: (x - 1940) / 40

    return tb, V, basis_func

def fit_polynomial_model(V, basis_func):
    """
    fit a polynomial model from the sol of
    the linear system (Vandermonde matrix)
    """
    coeff = np.linalg.solve(V, pops)
    # use basis_func to transform the data to make correct predictions
    poly = lambda x: np.polyval(coeff, basis_func(x))
    return poly

def eval_polynomial(poly):
    """
    evaluate the polynomial at years in 1-year interval
    """
    # transform the data points to evaluate with the coeffs
    xs = np.arange(years[0], years[-1] + 1)
    # find the solution for fitting
    sol = poly(xs)
    return sol

def plot_fitted(sol, name, title):
    """
    plot the given interpolation
    """
    xs = np.arange(years[0], years[-1] + 1)
    plt.figure()
    plt.plot(xs, sol, "-b", label="polynomial")
    plt.plot(years, pops, "*k", label="data points")
    plt.title(title)
    plt.legend()
    plt.savefig(name + ".png")

def hermite_cubic_interp():
    """
    interpolate a hermite cubic interpolation spline with the
    pchip algorithm (Piecewise Cubic Hermite Interpolating Polynomial)
    which uses monotonic cubic splines to find the value of new points
    """
    hermite_cubic = PchipInterpolator(years, pops)
    return hermite_cubic

def eval_hermite_cubic(cubic_interpolate):
    """
    evaluate the hermite cubic interpolation spline
    at years in 1-year interval
    """
    xs = np.arange(years[0], years[-1] + 1)
    sol = cubic_interpolate(xs)
    return sol

def cubic_interp():
    """
    interpolate a cubic interpolation spline with the
    CubicSpline algorithm
    """
    cs = CubicSpline(years, pops)
    return cs

def eval_cubic(cs):
    """
    evaluate the cubic interpolation spline
    at years in 1-year interval
    """
    xs = np.arange(years[0], years[-1] + 1)
    sol = cs(xs)
    return sol

def extrapolate(poly, hermite_cubic, cs, year=1990, true=248709873):
    """
    extrapolate on the given year with each model and compute the error
    on the prediction, finally show the result on a table
    """
    # generate the table for the error term
    tb = PrettyTable()
    tb.title = "error of each prediction (true = {})".format(true)
    tb.field_names = ["method", "prediction", "error"]

    # make predictions by each model
    res1 = poly(year)
    res2 = hermite_cubic(year)
    res3 = cs(year)

    # show the result on the model
    tb.add_row(["polynomial", res1, "{:.3e}".format(np.abs(true - res1))])
    tb.add_row(["hermite cubic spline interpolation", res2, "{:.3e}".format(np.abs(true - res2))])
    tb.add_row(["cubic spline interpolation", res3, "{:.3e}".format(np.abs(true - res3))])

    return tb

def plot_together(poly, hermite_cubic, cs):
    """
    plot the predictions made by the polynomial, hermite cubic interpolation
    and cubic interpolation spline together to make comparison
    """
    xs = np.arange(years[0], years[-1] + 1)

    # make predictions by each model
    sol1 = poly(xs)
    sol2 = hermite_cubic(xs)
    sol3 = cs(xs)

    # put the predictions together on a graph
    plt.figure()
    plt.plot(xs, sol1, "-r", label="polynomial")
    plt.plot(xs, sol2, "-g", label="hermite cubic spline")
    plt.plot(xs, sol3, "-b", label="cubic spline")
    plt.plot(years, pops, "*k", label="data points")
    plt.title("predictions from different models")
    plt.legend()
    plt.savefig("comparison.png")

def fit_langrange_model(basis_func):
    """
    fit a polynomial model by lagrange
    """
    poly_lagrange = lagrange(basis_func(years), pops)
    model = lambda x: poly_lagrange(basis_func(x))
    return model

def eval_lagrange(poly_lagrange):
    """
    evaluate the polynomial langrange
    at years in 1-year interval
    """
    xs = np.arange(years[0], years[-1] + 1)
    sol = poly_lagrange(xs)
    return sol

def timing_experiment(poly, hermite_cubic, cs, poly_lagrange):
    """
    test the average time consumption for the polynomial, hermite
    cubic spline interpolation, cubic spline interpolation and the
    polynomial from langrange method over 100 runs, test on years from
    1900 to 1980 in 1-year interval
    """
    # create the table
    tb = PrettyTable()
    tb.title = "mean time on each method"
    tb.field_names = ["method", "run time"]

    # test the polynomial method
    start = time.perf_counter()
    for i in range(100):
        sol = eval_polynomial(poly)
    time_taken1 = time.perf_counter() - start
    time_taken1 /= 100

    # test the hermite cubic spline interpolation method
    start = time.perf_counter()
    for i in range(100):
        sol = eval_hermite_cubic(hermite_cubic)
    time_taken2 = time.perf_counter() - start
    time_taken2 /= 100

    # test the cubic spline interpolation method
    start = time.perf_counter()
    for i in range(100):
        sol = eval_cubic(cs)
    time_taken3 = time.perf_counter() - start
    time_taken3 /= 100

    # test the polynomial (derived from lagrange) method
    start = time.perf_counter()
    for i in range(100):
        sol = eval_hermite_cubic(poly_lagrange)
    time_taken4 = time.perf_counter() - start
    time_taken4 /= 100

    # add data to the table
    tb.add_row(["polynomial", time_taken1])
    tb.add_row(["hermite cubic", time_taken2])
    tb.add_row(["cubic interpolation spline", time_taken3])
    tb.add_row(["polynomial by lagrange", time_taken4])

    return tb

def fit_newton():
    """
    fit the polynomial model from newton's method, by the lower
    triangular matrix system (divided differences approach)
    """
    # generate the linear system to find coefficients
    ts = years
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, ts[1] - ts[0], 0, 0, 0, 0, 0, 0, 0],
        [1, ts[2] - ts[0], (ts[2] - ts[0])* (ts[2] - ts[1]), 0, 0, 0, 0, 0, 0],
        [1, ts[3] - ts[0], (ts[3] - ts[0])* (ts[3] - ts[1]), (ts[3] - ts[0])* (ts[3] - ts[1]) * (ts[3] - ts[2]), 0, 0, 0, 0, 0],
        [1, ts[4] - ts[0], (ts[4] - ts[0]) * (ts[4] - ts[1]), (ts[4] - ts[0]) * (ts[4] - ts[1]) * (ts[4] - ts[2]), (ts[4] - ts[0]) * (ts[4] - ts[1]) * (ts[4] - ts[2]) * (ts[4] - ts[3]), 0, 0, 0, 0],
        [1, ts[5] - ts[0], (ts[5] - ts[0]) * (ts[5] - ts[1]), (ts[5] - ts[0]) * (ts[5] - ts[1]) * (ts[5] - ts[2]), (ts[5] - ts[0]) * (ts[5] - ts[1]) * (ts[5] - ts[2]) * (ts[5] - ts[3]), (ts[5] - ts[0]) * (ts[5] - ts[1]) * (ts[5] - ts[2]) * (ts[5] - ts[3]) * (ts[5] - ts[4]), 0, 0, 0],
        [1, ts[6] - ts[0], (ts[6] - ts[0]) * (ts[6] - ts[1]), (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]), (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]) * (ts[6] - ts[3]), (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]) * (ts[6] - ts[3]) * (ts[6] - ts[4]), (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]) * (ts[6] - ts[3]) * (ts[6] - ts[4]) * (ts[6] - ts[5]), 0, 0],
        [1, ts[7] - ts[0], (ts[7] - ts[0]) * (ts[7] - ts[1]), (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]), (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (ts[7] - ts[3]), (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (ts[7] - ts[3]) * (ts[7] - ts[4]), (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (ts[7] - ts[3]) * (ts[7] - ts[4]) * (ts[7] - ts[5]), (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (ts[7] - ts[3]) * (ts[7] - ts[4]) * (ts[7] - ts[5]) * (ts[7] - ts[6]), 0],
        [1, ts[8] - ts[0], (ts[8] - ts[0]) * (ts[8] - ts[1]), (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]), (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (ts[8] - ts[3]), (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (ts[8] - ts[3]) * (ts[8] - ts[4]), (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (ts[8] - ts[3]) * (ts[8] - ts[4]) * (ts[8] - ts[5]), (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (ts[8] - ts[3]) * (ts[8] - ts[4]) * (ts[8] - ts[5]) * (ts[8] - ts[6]), (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (ts[8] - ts[3]) * (ts[8] - ts[4]) * (ts[8] - ts[5]) * (ts[8] - ts[6]) * (ts[8] - ts[7])]
    ])

    # solve for the coefficients
    x = np.linalg.solve(A, pops)

    # create the polynomial model from the newton's method
    model = lambda t: x[0] + x[1] * (t - years[0]) + x[2] * (t - years[0]) * (t - years[1]) + \
        x[3] * (t - years[0]) * (t - years[1]) * (t - years[2]) + x[4] * (t - years[0]) * (t - years[1]) * (t - years[2]) * (t - years[3]) + \
            x[5] * (t - years[0]) * (t - years[1]) * (t - years[2]) * (t - years[3]) * (t - years[4]) + \
                 x[6] * (t - years[0]) * (t - years[1]) * (t - years[2]) * (t - years[3]) * (t - years[4]) * (t - years[5]) + \
                     x[7] * (t - years[0]) * (t - years[1]) * (t - years[2]) * (t - years[3]) * (t - years[4]) * (t - years[5]) * (t - years[6]) + \
                        x[8] * (t - years[0]) * (t - years[1]) * (t - years[2]) * (t - years[3]) * (t - years[4]) * (t - years[5]) * (t - years[6]) * (t - years[7])
    return model

def newton_extra_point(newton_model, year=1990, true=248709873):
    """
    add a new data point (1990, 248709873) to the old newton polynomial
    model, to create a new model (by the incremental approach)
    """
    ts = years
    # use the incremental approach to add a new term (derived from the new data point)
    # to the previous model to create the new model
    new_coeff = (true - newton_model(year)) / ((year - ts[0]) * (year - ts[1]) * (year - ts[2]) * (year - ts[3]) * (year - ts[4]) * \
        (year - ts[5]) * (year - ts[6]) * (year - ts[7]) * (year - ts[8]))
    new_newton_model = lambda t: newton_model(t) + new_coeff * (t - years[0]) * (t - years[1]) * (t - years[2]) * \
        (t - years[3]) * (t - years[4]) * (t - years[5]) * (t - years[6]) * (t - years[7]) * (t -years[8])
    return new_newton_model

def eval_newton(newton_model):
    """
    evaluate the polynomial newton
    at years in 1-year interval
    """
    xs = np.arange(years[0], years[-1] + 1)
    sol = newton_model(xs)
    return sol

def plot_two_newton(newton_model, new_newton_model, year=1990, true=248709873):
    """
    plot the old and new newton model by adding the extra new data point
    to the original data points
    """
    xs = np.arange(years[0], years[-1] + 10 + 1)
    sol1 = newton_model(xs)
    sol2 = new_newton_model(xs)

    plt.figure()
    plt.plot(xs, sol1, "--b", label="newton without new point")
    plt.plot(xs, sol2, "-c", label="newton with new point")
    plt.plot(years.tolist() + [year], pops.tolist() + [true], "*k", label="data point")
    plt.title("comparison on two newton methods")
    plt.legend()
    plt.savefig("newton_comp.png")

def eval_poly_after_rounding(V, basis_func):
    """
    compare and contrast the polynomial models built
    by the original and rounded data points (creating a table
    and a chart on trend)
    """
    # round the data to the nearest million
    round_pops = np.round(pops, -6)
    # find coefficients with original and rounded data
    coeff = np.linalg.solve(V, pops)
    new_coeff = np.linalg.solve(V, round_pops)
    # build models on original and rounded data
    poly = lambda x: np.polyval(coeff, basis_func(x))
    new_poly = lambda x: np.polyval(new_coeff, basis_func(x))

    # find solution with two polynomials
    xs = np.arange(years[0], years[-1] + 1)
    sol = poly(xs)
    new_sol = new_poly(xs)

    # create the table to show data
    tb = PrettyTable()
    tb.title = "comparison on polynomial coefficients"
    tb.field_names = ["degree", "old polynomial coeff", "new polynomial coeff", "difference (abs val)"]
    for i in range(years.shape[0]):
        tb.add_row([years.shape[0] - 1 - i, coeff[i], new_coeff[i], np.abs(coeff[i] - new_coeff[i])])

    # plot the trend of two polynomials
    plt.figure()
    plt.plot(xs, sol, "--g", label="original polynomial model")
    plt.plot(xs, new_sol, "--r", label="round model")
    plt.plot(years, pops, "*k", label="data points")
    plt.title ("comparison on polynomials")
    plt.legend()
    plt.savefig("poly_comp.png")

    return tb

if __name__ == "__main__":
    # pass

    # Q1
    tb, V, basis_func = vandermonde()
    print(tb)
    # Q2
    poly = fit_polynomial_model(V, basis_func)
    sol = eval_polynomial(poly)
    plot_fitted(sol, "poly", "fitted polynomial by the Vandermonde matrix")
    # Q3
    hermite_cubic = hermite_cubic_interp()
    sol = eval_hermite_cubic(hermite_cubic)
    plot_fitted(sol, "cubic_hermite_spline", "cubic hermite spline interpolation")
    # Q4
    cs = cubic_interp()
    sol = eval_cubic(cs)
    plot_fitted(sol, "cubic_interpolation", "cubic spline interpolation")
    plot_together(poly, hermite_cubic, cs)
    # Q5
    tb = extrapolate(poly, hermite_cubic, cs)
    print(tb)
    # Q6
    poly_lagrange = fit_langrange_model(basis_func)
    sol = eval_lagrange(poly_lagrange)
    plot_fitted(sol, "lagrange", "polynomial by lagrange")
    tb = timing_experiment(poly, hermite_cubic, cs, poly_lagrange)
    print(tb)
    # Q7
    newton_model = fit_newton()
    sol = eval_newton(newton_model)
    plot_fitted(sol, "newton_model", "polynomial by newton")
    new_newton_model = newton_extra_point(newton_model)
    plot_two_newton(newton_model, new_newton_model)
    # Q8
    tb = eval_poly_after_rounding(V, basis_func)
    print(tb)
