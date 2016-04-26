from arma import ARMA
from e_arma_predicter import *
import random
import pickle
import sys
from matplotlib import pyplot



missing_percent = 0.1
a = ARMA([0.3, -0.4], [0.1], 0.3)
time_series = [a.generater.next() for i in range(4000)]
p = ArmaPredicter(5, a.max_noise)

def run_test():
    for index, x in enumerate(time_series):
        print index
        if random.random() > missing_percent:
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
    pyplot.plot(xs, ys)
    pyplot.show()


if __name__ == '__main__':
    run_test()
