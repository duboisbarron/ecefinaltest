#!/usr/bin/env python
import csv
from sklearn import linear_model
import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from math import sqrt


# MASTERFILES = [
#     'small.csv',
#     'train.csv',
#     'test.csv'
# ]

MASTERFILES = [
    '/classes/ece2720/pe4/small.csv',
    '/classes/ece2720/pe4/train.csv',
    '/classes/ece2720/pe4/test.csv'
]
# reg = linear_model.LinearRegression()
#
# reg.fit([[3, 4], [23, 233]], [3, 9])
#
# print(reg.coef_)
# print(reg.intercept_)

astar_coeffs_unregularized = []
astar_coeffs_ridge = []



def a_b_train(training_file):
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []

    y_vals = []

    csv_file = open(training_file, 'rU')
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row_num, row in enumerate(csv_reader):
        if row_num == 0:
            continue

        col1.append(float(row[0]))
        col2.append(float(row[1]))
        col3.append(float(row[2]))
        col4.append(float(row[3]))
        col5.append(float(row[4]))
        col6.append(float(row[5]))
        col7.append(float(row[6]))
        col8.append(float(row[7]))
        y_vals.append(float(row[8]))



    col1np = np.array(col1)
    col2np = np.array(col2)
    col3np = np.array(col3)
    col4np = np.array(col4)
    col5np = np.array(col5)
    col6np = np.array(col6)
    col7np = np.array(col7)
    col8np = np.array(col8)

    matrix = np.array([col1np, col2np, col3np, col4np, col5np, col6np, col7np, col8np])
    yvalsnp = np.array(y_vals)
    #
    reg = linear_model.LinearRegression()
    reg.fit(matrix.transpose(), yvalsnp)

    # print(reg.coef_)
    if training_file == MASTERFILES[1]:
        for coef in reg.coef_:
            astar_coeffs_unregularized.append(coef)
    # print(reg.intercept_)
    print('R^2 for regression on training data from ' + training_file + ' ' + str(reg.score(matrix.transpose(), yvalsnp)))

    col1test = []
    col2test = []
    col3test = []
    col4test = []
    col5test = []
    col6test = []
    col7test = []
    col8test = []
    y_valstest = []
    csv_test = open(MASTERFILES[2], 'rU')
    csv_reader_test = csv.reader(csv_test, delimiter=',')
    for index, row in enumerate(csv_reader_test):
        if index == 0:
            continue
        else:
            col1test.append(float(row[0]))
            col2test.append(float(row[1]))
            col3test.append(float(row[2]))
            col4test.append(float(row[3]))
            col5test.append(float(row[4]))
            col6test.append(float(row[5]))
            col7test.append(float(row[6]))
            col8test.append(float(row[7]))
            y_valstest.append(float(row[8]))

    col1nptest = np.array(col1test)
    col2nptest = np.array(col2test)
    col3nptest = np.array(col3test)
    col4nptest = np.array(col4test)
    col5nptest = np.array(col5test)
    col6nptest = np.array(col6test)
    col7nptest = np.array(col7test)
    col8nptest = np.array(col8test)

    yvalsnp_test = np.array(y_valstest)

    matrix_test = np.array([col1nptest, col2nptest, col3nptest, col4nptest, col5nptest,
                       col6nptest, col7nptest, col8nptest]).transpose()

    # print(reg.predict(matrix_test))
    print('R^2 values for test data: ' + str(reg.score(matrix_test, yvalsnp_test)))

def get_data(filename):
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []

    y_vals = []

    csv_file = open(filename, 'rU')
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row_num, row in enumerate(csv_reader):
        if row_num == 0:
            continue

        col1.append(float(row[0]))
        col2.append(float(row[1]))
        col3.append(float(row[2]))
        col4.append(float(row[3]))
        col5.append(float(row[4]))
        col6.append(float(row[5]))
        col7.append(float(row[6]))
        col8.append(float(row[7]))
        y_vals.append(float(row[8]))

    col1np = np.array(col1)
    col2np = np.array(col2)
    col3np = np.array(col3)
    col4np = np.array(col4)
    col5np = np.array(col5)
    col6np = np.array(col6)
    col7np = np.array(col7)
    col8np = np.array(col8)

    matrix = np.array([col1np, col2np, col3np, col4np, col5np, col6np, col7np, col8np]).transpose()
    yvalsnp = np.array(y_vals)


    return matrix, yvalsnp

def run_ridge():
    matrix, yvalsnp = get_data(MASTERFILES[0])


    matrix_test, yvalsnp_test = get_data(MASTERFILES[1])

    # print('num of data points in small is: ' + str(len(matrix[0])))

    # print(range(100))
    n = 50


    r2_small_arr = []
    r2_test_arr = []
    lambda_vals = []
    for m in range(100):
        lambda_val = n*(1.2)**float(m)
        lambda_vals.append(lambda_val)
        ridge_reg = linear_model.Ridge(alpha=lambda_val)
        ridge_reg.fit(matrix, yvalsnp)
        # print(ridge_reg.intercept_)
        # print(ridge_reg.coef_)
        r2_small_value = ridge_reg.score(matrix, yvalsnp)
        # print(r2_small_value)
        r2_small_arr.append(r2_small_value)

        r2_test_value = ridge_reg.score(matrix_test, yvalsnp_test)

        r2_test_arr.append(r2_test_value)
        # print(r2_test_value)

        # print(r2_small_value, r2_test_value)


    # print(len(r2_test_arr), len(r2_small_arr), len(lambda_vals))
    # clf = linear_model.Ridge(alpha=1.0)
    '''
    Plot R^2 for both data sets as a function of lambda on a single graph using a log-scale for lambda
    use the "semilogx()" function in PyPlot
    '''
    # print(str(max(r2_test_arr)))
    index_best_lambda = r2_test_arr.index(max(r2_test_arr))
    # print(index_best_lambda)
    # print(lambda_vals[index_best_lambda])


    # print(str(lambda_vals[index_best_lambda])  + 'is the best of the 100 lambda values!!')
    ridge_reg = linear_model.Ridge(alpha=lambda_vals[index_best_lambda])
    ridge_reg.fit(matrix, yvalsnp)
    # print(len(ridge_reg.coef_))
    # print(ridge_reg.coef_)
    for coef in ridge_reg.coef_:
        astar_coeffs_ridge.append(coef)


    '''
    for this value of lambda, is the corresponding a* vector shorter than the a* vector in the regularized
    '''


    #
    # print(min(lambda_vals))
    # print(max(lambda_vals))
    # print(max(lambda_vals))
    plt.ylabel('R^2 Values')
    plt.xlabel('Lambda Values')
    plt.title('Lambda Values vs R^2 Values for Ridge Regression trained on small.csv')
    plt.semilogx(lambda_vals, r2_small_arr, label='small.csv')
    plt.semilogx(lambda_vals, r2_test_arr, label='test.csv')
    plt.legend()

    plt.savefig('figure1.pdf')
    plt.clf()
    # plt.show()





def run_lasso():
    matrix, yvalsnp = get_data(MASTERFILES[0])

    matrix_test, yvalsnp_test = get_data(MASTERFILES[2])


    # print(matrix.shape[0])
    n = matrix.shape[0]
    lambda_values = []
    r2_small_values = []
    r2_test_values = []
    for v in range(151):
        if v == 0:
            continue
        lambda_value = 2 * n * v
        lambda_values.append(v)
        # TODO: need to scale the alpha value somehow!!
        lasso_reg = linear_model.Lasso(alpha=v)

        lasso_reg.fit(matrix, yvalsnp)
        # print(ridge_reg.intercept_)
        # print(ridge_reg.coef_)
        r2_small_value = lasso_reg.score(matrix, yvalsnp)
        # print(r2_small_value)
        r2_small_values.append(r2_small_value)

        r2_test_value = lasso_reg.score(matrix_test, yvalsnp_test)

        r2_test_values.append(r2_test_value)


    # print(len(r2_small_values), len(r2_test_values))


    max_r2test = max(r2_test_values)

    # print(max_r2test in r2_test_values)



    best_index = r2_test_values.index(max_r2test)
    # print('THIS IS THE BEST LAMBDA VALUE: ' + str(lambda_values[best_index]))

    lassreg = linear_model.Lasso(alpha=lambda_values[best_index])
    lassreg.fit(matrix, yvalsnp)
    # print(lassreg.coef_)





    plt.title('Lambda Values vs R^2 Values based on LASSO Regression trained on small.csv')
    plt.xlabel('Lambda Values')
    plt.ylabel('R^2 Values')
    plt.plot(lambda_values, r2_small_values, label='small.csv')
    plt.plot(lambda_values, r2_test_values, label='test.csv')
    plt.legend()

    plt.savefig('figure2.pdf')
    plt.clf()
    # plt.show()


def adding_features():
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []

    y_vals = []

    csv_file = open(MASTERFILES[1], 'rU')
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row_num, row in enumerate(csv_reader):
        if row_num == 0:
            continue

        col1.append(float(row[0]))
        col2.append(float(row[1]))
        col3.append(float(row[2]))
        col4.append(float(row[3]))
        col5.append(float(row[4]))
        col6.append(float(row[5]))
        col7.append(float(row[6]))
        col8.append(float(row[7]))
        y_vals.append(float(row[8]))

    col1np = np.array(col1)
    col1sq = np.array([x ** 2.0 for x in col1])

    col2np = np.array(col2)
    col2sq = np.array([x ** 2.0 for x in col2])

    col3np = np.array(col3)
    col3sq = np.array([x ** 2.0 for x in col3])

    col4np = np.array(col4)
    col4sq = np.array([x ** 2.0 for x in col4])

    col5np = np.array(col5)
    col5sq = np.array([x ** 2.0 for x in col5])

    col6np = np.array(col6)
    col6sq = np.array([x ** 2.0 for x in col6])

    col7np = np.array(col7)
    col7sq = np.array([x ** 2.0 for x in col7])

    col8np = np.array(col8)
    col8sq = np.array([x ** 2.0 for x in col8])

    matrix = np.array([col1np, col1sq,
                       col2np, col2sq,
                       col3np, col3sq,
                       col4np, col4sq,
                       col5np, col5sq,
                       col6np, col6sq,
                       col7np, col7sq,
                       col8np, col8sq
                       ])
    yvalsnp = np.array(y_vals)

    reg = linear_model.LinearRegression()
    reg.fit(matrix.transpose(), yvalsnp)
    # print(reg.coef_)
    # print(reg.intercept_)
    print('R^2 for regression on quadratic data from train.csv: ' + str(reg.score(matrix.transpose(), yvalsnp)))




    col1test = []
    col2test = []
    col3test = []
    col4test = []
    col5test = []
    col6test = []
    col7test = []
    col8test = []
    y_valstest = []
    csv_test = open(MASTERFILES[2], 'rU')
    csv_reader_test = csv.reader(csv_test, delimiter=',')
    for index, row in enumerate(csv_reader_test):
        if index == 0:
            continue
        else:
            col1test.append(float(row[0]))
            col2test.append(float(row[1]))
            col3test.append(float(row[2]))
            col4test.append(float(row[3]))
            col5test.append(float(row[4]))
            col6test.append(float(row[5]))
            col7test.append(float(row[6]))
            col8test.append(float(row[7]))
            y_valstest.append(float(row[8]))

    col1nptest = np.array(col1test)
    col1testsq = np.array([x ** 2.0 for x in col1nptest])

    col2nptest = np.array(col2test)
    col2testsq = np.array([x ** 2.0 for x in col2nptest])

    col3nptest = np.array(col3test)
    col3testsq = np.array([x ** 2.0 for x in col3nptest])

    col4nptest = np.array(col4test)
    col4testsq = np.array([x ** 2.0 for x in col4nptest])

    col5nptest = np.array(col5test)
    col5testsq = np.array([x ** 2.0 for x in col5nptest])

    col6nptest = np.array(col6test)
    col6testsq = np.array([x ** 2.0 for x in col6nptest])

    col7nptest = np.array(col7test)
    col7testsq = np.array([x ** 2.0 for x in col7nptest])

    col8nptest = np.array(col8test)
    col8testsq = np.array([x ** 2.0 for x in col8nptest])

    yvalsnp_test = np.array(y_valstest)

    matrix_test = np.array([
                            col1nptest, col1testsq,
                            col2nptest, col2testsq,
                            col3nptest, col3testsq,
                            col4nptest, col4testsq,
                            col5nptest, col5testsq,
                            col6nptest, col6testsq,
                            col7nptest, col7testsq,
                            col8nptest, col8testsq
                            ]).transpose()

    print('R^2 for test data is: ' + str(reg.score(matrix_test, yvalsnp_test)))

def magnitude_list(arr):
    sqlist = [x**2.0 for x in arr]
    summ = sum(sqlist)
    return sqrt(summ)

if __name__ == "__main__":
    a_b_train(MASTERFILES[1])
    # print('running same stuff but with small.csv as training set ')
    a_b_train(MASTERFILES[0])
    run_ridge()


    # print(astar_coeffs_unregularized)
    # print(astar_coeffs_ridge)
    #
    # print(magnitude_list(astar_coeffs_ridge))
    #
    #
    # print(magnitude_list(astar_coeffs_unregularized))

    run_lasso()
    adding_features()

