from arma import ARMA
from arma_predicter import *
import random
import pickle
import sys
import cProfile


missing_percent = 0.05
#a = ARMA([0.6, -0.5, 0.4, -0.4, 0.3], [0.3 -0.2], 0.3)
a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [-0.2, 0.3], 0.3)
time_series = [a.generater.next() for i in range(4000)]
p = ArmaPredicter(10, a.max_noise)

def run_test():
    rec_xs = []
    for index, x in enumerate(time_series):
        print index
        if random.random() > missing_percent:
            rec_x = p.predict_and_fit(x)
            rec_xs.append(rec_x)
        else:
            p.predict_and_fit('*')
    with open("p", "wb") as f:
        pickle.dump(p, f)
    with open("rec_xs", "wb") as f:
        pickle.dump(rec_xs, f)

if __name__ == '__main__':
    cProfile.run("run_test()","result")
