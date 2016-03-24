from arma import ARMA
from e_arma_predicter import *
#from arma_predicter import *
import random
import pickle
import sys
import cProfile


missing_percent = 0.1
#a = ARMA([0.6, -0.5, 0.4, -0.4, 0.3], [0.3 -0.2], 0.3)
a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [-0.2, 0.3], 0.3)
time_series = [a.generater.next() for i in range(10000)]
p = ArmaPredicter(10, a.max_noise)

def run_test():
    for index, x in enumerate(time_series):
        print index
        if random.random() > missing_percent:
            rec_x = p.predict_and_fit(x)
        else:
            p.predict_and_fit('*')
    with open("p", "wb") as f:
        pickle.dump(p, f)
    with open("noises", "wb") as f:
        pickle.dump(a.noises, f)

if __name__ == '__main__':
    cProfile.run("run_test()","result")
