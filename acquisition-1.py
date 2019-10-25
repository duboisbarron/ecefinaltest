#!/usr/bin/env python
import csv
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import codecs
import os
import math

# PART 2
# PART 2
def get_data(filename):
    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = []
    npData = np.array([])
    for row in csv_reader:
        data.append(row)
    for value in data[0]:
        npData = np.append(npData, float(value))
    return npData


def make_gaussian_pdf(data, figname):
    mean = np.mean(data)
    std = np.std(data)
    data.sort()

    pdf_values = [scipy.stats.norm.pdf(value, mean, std) for value in data]
    plt.plot(data, pdf_values)
    plt.title('Gaussian PDF of "synthetic.csv" Values')
    plt.xlabel('Actual Values')
    plt.ylabel('PDF Values')
    plt.savefig(figname)
    plt.clf()

def make_histogram():
    data = get_data('/classes/ece2720/pe3/synthetic.csv')

    plt.hist(data, bins=100, density=True)

    mean = np.mean(data)
    std = np.std(data)
    data.sort()
    pdf_values = [scipy.stats.norm.pdf(value, mean, std) for value in data]
    plt.plot(data, pdf_values)

    plt.title('Histogram for "synthetic.csv"')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    # plt.show()
    plt.savefig('figure1')
    plt.clf()

# TODO: FIGURE OUT WHICH PLOT IS CORRECT
def use_scipy_n_prob_plot():
    data = get_data('/classes/ece2720/pe3/synthetic.csv')
    scipy.stats.probplot(data, plot=plt.subplot())
    plt.title('Probability Plot for "synthetic.csv"')
    plt.savefig('figure2')
    plt.clf()
    # plt.show()

def detect_chauvenet_criteria(np_array, dmax):

    two_n = 2.0*float(len(np_array))
    # print("criterion > 1/2")
    # print(two_n*(1.0 - scipy.stats.norm.cdf(dmax)))
    #
    # print('THE VALUE THAT IS THE OUTLIER IS: ' + str(min(np_array)))

    return two_n*(1.0 - scipy.stats.norm.cdf(dmax))


def calc_d_max(np_array):
    mu = maximum_likelihood_estimator_mu(np_array)
    sig = maximum_likelihood_estimator_sig_sq(np_array)

    stdeviation = math.sqrt(sig)

    # print(mu, stdeviation)
    #
    #
    # print("variance is: " + str(maximum_likelihood_estimator_sig_sq(np_array)))
    # print("std is: " + str(stdeviation))
    # print("mu is: " + str(mu))



    arr = [(value-mu)/stdeviation for value in np_array]

    # print(max(arr))
    return max(arr)
    # arr = []
    # for value in np_array:
    #     arr.append((value-mu)/sig)
    # print(max(arr))
    # print(max(arr))
    # print(max(arr))
    # print(max(arr))
    # print(max(arr))
    # print(max(arr))
    # print(max(arr))
    # print(max(arr))
    # return max(arr)


def maximum_likelihood_estimator_mu(np_array):
    data_sum = np.sum(np_array)
    return (1.0/float(len(np_array))) * data_sum

def maximum_likelihood_estimator_sig_sq(np_array):

    # print(ml_mu)
    # print(np_array)
    ml_mu = maximum_likelihood_estimator_mu(np_array)

    squared_difference = np.array([])

    for value in np_array:
        x = value - ml_mu

        # print(x**2)
        # print(value - ml_mu)
        squared_difference = np.append(squared_difference, x**2)

        # squared_difference = np.append(squared_difference, (float(x_i) - float(ml_mu)) ** 2)

    sum_squared_diff = np.sum(squared_difference)
    return 1.0/float(len(np_array))*sum_squared_diff


def get_ml_sig():

    data = get_data('/classes/ece2720/pe3/synthetic.csv')

    mu = maximum_likelihood_estimator_mu(data)
    print('maximum likelihood estimation of mu is: ' + str(mu))

    print('maximum likelihood estimation of variance is: ' + str(maximum_likelihood_estimator_sig_sq(data, mu)))

def max_like_mu():
    data = get_data('/classes/ece2720/pe3/synthetic.csv')

    print('maximum likelihood estimation of mu is: ' + str(maximum_likelihood_estimator_mu(data)))

def make_n_prob_plot():
    data = get_data('/classes/ece2720/pe3/synthetic.csv')
    data.sort()
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = np.array([])
    for n in data:
        normalized_data = np.append(normalized_data, (n-mean)/std)

    plt.plot(normalized_data, data, 'bo')
    plt.title('Normal Probablity Plot for "synthetic.csv"')
    plt.ylabel("Z-Scores")
    plt.xlabel("Measured Data")
    plt.savefig('figure2')
    plt.clf()
    # plt.show()

# END PART 2
# END PART 2
# END PART 2

def age_histogram():
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)
    valid_ages = np.array([])
    for row in csv_reader:

        try:
            valid_ages = np.append(valid_ages, int(row[5]))
        except Exception as e:
            continue

    plt.hist(valid_ages, bins=100, density=True)

    mean = np.mean(valid_ages)
    std = np.std(valid_ages)
    valid_ages.sort()
    pdf_values = [scipy.stats.norm.pdf(value, mean, std) for value in valid_ages]
    plt.plot(valid_ages, pdf_values)
    plt.ylabel('Frequency')
    plt.xlabel('Value Observed')
    plt.title('Histogram of Ages of Titanic Passengers n=' + str(len(valid_ages)))
    plt.savefig('figure3')
    plt.clf()



    #
    # plt.title('Gaussian PDF of Titanic Passenger Age Values')
    # plt.xlabel('Actual Values')
    # plt.ylabel('PDF Values')
    # plt.savefig('age_histo_PDF_values')
    #
    # plt.clf()



    # plt.show()

def age_prob_plot():
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)
    row_num = 0
    column_headers = []
    valid_ages = np.array([])
    for row in csv_reader:

        try:
            # print(type(row[5]))
            valid_ages = np.append(valid_ages, int(row[5]))
            # valid_ages.append(int(row[5]))
        except Exception as e:
            continue
    #
    # print(maximum_likelihood_estimator_mu(valid_ages))
    # print(maximum_likelihood_estimator_sig_sq(valid_ages))
    scipy.stats.probplot(valid_ages, plot=plt.subplot())
    plt.title('Probability Plot for Ages of Titanic Passengers')
    plt.savefig('figure5')
    plt.clf()

# FARE HISTOGRAM
# FARE HISTOGRAM
# FARE HISTOGRAM
# FARE HISTOGRAM
# FARE HISTOGRAM
# FARE HISTOGRAM
def fare_histogram():
    # print("CALKSJFLKA")
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)
    row_num = 0
    column_headers = []
    valid_fares = []
    for row in csv_reader:

        try:
            valid_fares = np.append(valid_fares, float(row[9]))
        except Exception as e:
            continue

    plt.hist(valid_fares, bins=50, density=True)
    mean = np.mean(valid_fares)
    std = np.std(valid_fares)
    valid_fares.sort()
    pdf_values = [scipy.stats.norm.pdf(value, mean, std) for value in valid_fares]
    plt.plot(valid_fares, pdf_values)

    plt.ylabel('Frequency')
    plt.xlabel('Value Observed')
    plt.title('Histogram of Fares of Titanic Passengers n=' + str(len(valid_fares)))
    plt.savefig('figure4')
    plt.clf()
    # plt.show()


    #
    # plt.title('Gaussian PDF of Titanic Passenger Fare Values')
    # plt.xlabel('Actual Values')
    # plt.ylabel('PDF Values')
    # plt.savefig('fare_histo_PDF_values')
    #
    # plt.clf()


def fare_prob_plot():
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)
    valid_fares = np.array([])
    for row in csv_reader:

        try:
            valid_fares = np.append(valid_fares, float(row[9]))
            # valid_ages.append(int(row[5]))
        except:
            continue
    #
    # print(maximum_likelihood_estimator_mu(valid_fares))
    # print(maximum_likelihood_estimator_sig_sq(valid_fares))

    scipy.stats.probplot(valid_fares, plot=plt.subplot())
    plt.title('Probability Plot for Passenger Fares of the Titanic')
    plt.savefig('figure6')
    plt.clf()




# determine the overall survival rate of passengers
# need total number of survivors and total number of deaths

# return the number of people in the group, the total number of people in the file
def get_overall_survival_rate():
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)
    row_num = 0
    column_headers = []
    valid_fares = np.array([])

    num_survived = 0
    num_dead = 0

    for row in csv_reader:

        try:
            if int(row[1]) == 0:
                num_dead += 1
            elif int(row[1]) == 1:
                num_survived +=1
            # valid_ages.append(int(row[5]))
        except:
            continue

    total = float(num_survived) + float(num_dead)

    print(int(total), float(num_survived)/float(total))

    # print('total number of passengers is: ' + str(total))
    # print('total number of survivors is: ' + str(float(num_survived)))
    # print('total number of dead is: ' + str(float(num_dead)))
    # print('overall survival rate is: ' + str(float(num_survived)/float(total)))
    #
    # return total, float(num_survived), float(num_dead), float(num_survived)/(float(num_survived) + float(num_dead))

# determine the overall survival rate of passengers
# need total number of survivors and total number of deaths
# return the number of people in the group, the total number of people in the file
def get_survival_rate_men_and_women():
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)

    num_women = 0
    dead_women = 0
    alive_women = 0

    num_men = 0
    dead_men = 0
    alive_men = 0


    for row in csv_reader:

        try:
            if(row[4] == 'female'):
                num_women += 1
                if int(row[1]) == 0:
                    dead_women += 1
                elif int(row[1]) == 1:
                    alive_women += 1
            elif(row[4] == 'male'):
                # print('found male')
                num_men += 1
                if int(row[1]) == 0:
                    dead_men += 1
                elif int(row[1]) == 1:
                    alive_men += 1
            # valid_ages.append(int(row[5]))
        except:
            continue

    total_women = num_women
    total_men = num_men

    print(total_women, float(alive_women)/float(total_women))
    print(total_men, float(alive_men)/float(total_men))



    arr = []
    for x in range(10):
        arr.append(x)


    #
    # print("total number of women is: " + str(total_women))
    # print("number of female survivors is: " + str(alive_women))
    # print("survival rate of women is: " + str(
    #     float(alive_women) / float(total_women)
    # ))
    #
    # print("total number of men is: " + str(total_men))
    # print("number of female survivors is: " + str(alive_men))
    #
    # print("survival rate of men is: " + str(
    #     float(alive_men) / float(total_men)
    # ))
    # print(total_women + total_men)

    # return total, float(num_survived), float(num_dead), float(num_survived)/(float(num_survived) + float(num_dead))
def get_survival_rate_first_class(passenger_class):
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)

    num_class = 0
    num_alive_class = 0
    num_dead_class = 0


    for row in csv_reader:

        try:
            if row[2] == passenger_class:
                # print("found first class ")
                num_class += 1
                if int(row[1]) == 0:
                    num_dead_class += 1
                elif int(row[1]) == 1:
                    num_alive_class += 1

            # valid_ages.append(int(row[5]))
        except:
            continue
    print(num_class, float(num_alive_class) / float(num_class))
    # print('total number of  class ' + passenger_class + ' passengers is: ' + str(num_class))
    # print('number of survivors of class ' + passenger_class + ' passengers is ' + str(num_alive_class))
    # print('survival rate of class ' + passenger_class + ' passengers is: ' + str(
    #     float(num_alive_class) / float(num_class)
    # ))

# TODO: make sure this shit works
def get_first_class_sex_survival_rate(passenger_class, passenger_sex):
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)

    num_in_group = 0
    num_alive = 0
    num_dead = 0


    for row in csv_reader:

        try:
            if row[2] == passenger_class:
                # print("found first class ")

            #     see if they are the right gender
                if row[4] == passenger_sex:
                    num_in_group+=1
                    if int(row[1]) == 0:
                        num_dead += 1
                    elif int(row[1]) == 1:
                        num_alive += 1

            # valid_ages.append(int(row[5]))
        except:
            continue

    print(num_in_group,  float(num_alive) / float(num_in_group))
    # print('total number of  class ' + passenger_class + ' ' + passenger_sex + ' passengers is: ' + str(num_in_group))
    # print('total number of  survivors of class ' + passenger_class + ' ' + passenger_sex + ' passengers is: ' + str(num_alive))
    # print('survival rate of class ' + passenger_class + ' passengers is: ' + str(
    #     float(num_alive) / float(num_in_group)
    # ))

def get_fare_survived_greater(fare):
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)

    num_in_group = 0
    num_alive = 0
    num_dead = 0


    for row in csv_reader:

        try:
            if float(row[9]) >= fare:
                num_in_group += 1
                if int(row[1]) == 0:
                    num_dead += 1
                elif int(row[1]) == 1:
                    num_alive += 1


            # valid_ages.append(int(row[5]))
        except:
            continue
    print(num_in_group, float(num_alive) / float(num_in_group))
    # print('total number of  passenger\'s who\'s fares exceeded ' + str(fare) + ' passengers is: ' + str(num_in_group))
    # print('total number of SURVIVORS passenger\'s who\'s fares exceeded ' + str(fare) + ' passengers is: ' + str(num_alive))
    # print('survival rate of passenger\'s who\'s fares exceeded ' + str(fare) + ' passengers is: ' + str(
    #     float(num_alive) / float(num_in_group)
    # ))

def get_fare_survived_less(fare):
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)

    num_in_group = 0
    num_alive = 0
    num_dead = 0


    for row in csv_reader:

        try:
            if float(row[9]) <= fare:
                num_in_group += 1
                if int(row[1]) == 0:
                    num_dead += 1
                elif int(row[1]) == 1:
                    num_alive += 1


            # valid_ages.append(int(row[5]))
        except:
            continue
    print(num_in_group, float(num_alive) / float(num_in_group))
    # print('total number of  passenger\'s who\'s fare was less than ' + str(fare) + ' passengers is: ' + str(num_in_group))
    # print('total number of  SURVIVORS passenger\'s who\'s fare was less than ' + str(fare) + ' passengers is: ' + str(num_alive))
    # print('survival rate of passenger\'s who\'s fares was less than ' + str(fare) + ' passengers is: ' + str(
    #     float(num_alive) / float(num_in_group)
    # ))


def get_family_survival_rate():
    csv_file = open("/classes/ece2720/pe3/titanic.csv")
    csv_reader = csv.reader(csv_file)

    num_in_group = 0
    num_alive = 0
    num_dead = 0

    for row in csv_reader:

        try:
            # print(int(row[7]))
            if int(row[7]) != 0:
                num_in_group += 1
                if int(row[1]) == 0:
                    num_dead += 1
                elif int(row[1]) == 1:
                    num_alive += 1

            # valid_ages.append(int(row[5]))
        except:
            continue
    print(num_in_group, float(num_alive) / float(num_in_group))
    # print('total number of  passengers traveling with family: ' + str(num_in_group))
    # print('total number of  SURVIVORS traveling with family: ' + str(num_alive))
    # print('survival rate of passengers traveling with a family : ' + str(
    #     float(num_alive) / float(num_in_group)
    # ))

def print_values_for_tables():
    get_overall_survival_rate()
    get_survival_rate_men_and_women()
    get_survival_rate_first_class('1')
    get_survival_rate_first_class('3')
    get_first_class_sex_survival_rate('1', 'male')
    get_first_class_sex_survival_rate('3', 'female')
    get_fare_survived_greater(100.0)
    get_fare_survived_less(50.0)
    get_family_survival_rate()



# f = codecs.open('unicode1.dat', 'r', 'UTF-8')
# print(f.readline(), os.path.getsize('unicode1.dat'))
#
# f = codecs.open('unicode2.dat', 'r', 'UTF-32-le')
# print(f.readline(), os.path.getsize('unicode2.dat'))
#
# f = codecs.open('unicode3.dat', 'r', 'UTF-8')
# print(f.readline(), os.path.getsize('unicode3.dat'))
#
# f = codecs.open('unicode4.dat', 'r', 'UTF-16')
# print(f.readline(), os.path.getsize('unicode4.dat'))
#
# f = codecs.open('unicode5.dat', 'r', 'UTF-32-be')
# print(f.readline(), os.path.getsize('unicode5.dat'))

#
# make_histogram()
# use_scipy_n_prob_plot()
# age_histogram()
# fare_histogram()
# age_prob_plot()
# fare_prob_plot()

#
# print("doing women")
# values = get_survival_rate_women()
# print("total number of people: " + str(values[0]))
# print("total number of survivors: " + str(values[1]))
# print("total number of dead: " + str(values[2]))
# print("survival rate for this group: " + str(values[3]))


# print_values_for_tables()



# get_ml_sig()
# d_max = calc_d_max(get_data('synthetic.csv'))
# print(d_max)
#
# print(len(get_data('synthetic.csv')))
#
# two_n = 2.0*len(get_data('synthetic.csv'))
#
# one_minus = (1.0 - scipy.stats.norm.cdf(d_max))
#
# print(two_n * one_minus)
# print(two_n * one_minus)
# print(two_n * one_minus)
# print(two_n * one_minus)
#
#
# print(calc_d_max(get_data('synthetic.csv')))
# print(calc_d_max(get_data('synthetic.csv')))
# print(calc_d_max(get_data('synthetic.csv')))
# print(calc_d_max(get_data('synthetic.csv')))
# print(calc_d_max(get_data('synthetic.csv')))
# d_max = calc_d_max(get_data('synthetic.csv'))
#
# detect_chauvenet_criteria(get_data('synthetic.csv'), d_max)
# detect_chauvenet_criteria(get_data('synthetic.csv'), d_max)
# detect_chauvenet_criteria(get_data('synthetic.csv'), d_max)


print(codecs.open('/classes/ece2720/pe3/unicode1.dat', 'rb', 'UTF-8').readline())
print(os.path.getsize('/classes/ece2720/pe3/unicode1.dat'))


print(codecs.open('/classes/ece2720/pe3/unicode2.dat', 'rb', 'UTF-32-le').readline())
print(os.path.getsize('/classes/ece2720/pe3/unicode2.dat'))

print(codecs.open('/classes/ece2720/pe3/unicode3.dat', 'rb', 'UTF-8').readline())
print(os.path.getsize('/classes/ece2720/pe3/unicode3.dat'))

print(codecs.open('/classes/ece2720/pe3/unicode4.dat', 'rb', 'UTF-16').readline())
print(os.path.getsize('/classes/ece2720/pe3/unicode4.dat'))

print(codecs.open('/classes/ece2720/pe3/unicode5.dat', 'rb', 'UTF-32-be').readline())
print(os.path.getsize('/classes/ece2720/pe3/unicode5.dat'))

# print('File Size is: ' + str(os.path.getsize('unicode4.dat')))

print_values_for_tables()


make_histogram()
use_scipy_n_prob_plot()
age_histogram()
fare_histogram()
age_prob_plot()
fare_prob_plot()



