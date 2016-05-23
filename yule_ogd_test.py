from arma import ARMA
#from e_arma_predicter import *
#from v2_e_arma_predicter import *
#from arma_predicter import *
#from numpy_arma_predicter import *
#from s_arma_predicter import *
#from dpark_arma_predicter import *
from yule_walker_ar_predicter import *
#from ogd_impute_predicter import *
import random
import pickle
import sys
from ar import AR
from matplotlib import pyplot



missing_percent = 0.2
#a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [-0.2, 0.3], 0.1)
a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
#a = ARMA([], [0.7], 0.5)
time_series = [a.generater.next() for i in range(10000)]
p = ArPredicter(5)

def run_test():
    for index, x in enumerate(time_series):
        print index
        if index < 6:
            p.predict_and_fit(x)
        elif random.random() > missing_percent:
            rec_x = p.predict_and_fit(x)
        else:
            p.predict_and_fit('*')
    plot(p)
    '''
    with open("p", "wb") as f:
        pickle.dump(p, f)
    '''

def plot(p):
    serrors = [error*error for error in p.errors]
    xs = [index for index, e in enumerate(serrors)]
    ys = [sum(serrors[:index+1])/(index+1) for index, e in enumerate(serrors)]
    ys1 = [0 if x == '*' else x for x in time_series]
    print ys[-1]
    #pyplot.plot(xs, ys,'g^',xs, ys1)
    pyplot.plot(xs, ys)
    pyplot.show()


if __name__ == '__main__':
    run_test()
