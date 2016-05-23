from arma import ARMA
from arma_predicter import *
import random
import pickle
import sys
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression as LR



missing_percent = 0.3
#a = ARMA([], [0.3], 0.6)
a = ARMA([], [0.1], 0.6)
time_series = [a.generater.next() for i in range(1000)]
p = ArmaPredicter(3, a.max_noise)

def run_test():
    ts = []
    for time in time_series:
        if random.random() > missing_percent:
            ts.append(time)
        else:
            ts.append('*')
    noises = [] 
    for index, x in enumerate(time_series):
        print index
        rec_x = p.predict_and_fit(x)
        if x == '*':
            noises.append(0)
        else:
            noises.append(x - rec_x)
    #plot(p)
    #ts = ts[1000:2000]
    #noises = noises[1000:2000]
    return learn_arma(ts, noises, p)

def plot(p):
    serrors = [error*error for error in p.errors]
    xs = [index for index, e in enumerate(serrors)]
    ys = [sum(serrors[:index+1])/(index+1) for index, e in enumerate(serrors)]
    pyplot.plot(xs, ys, ys1)
    pyplot.show()

def learn_arma(ts, noises, p):
    serrors = [error*error for error in p.errors]
    variance =  sum(serrors) / len(serrors)
    print variance
    xs = []
    ys = []
    for index, (t, noise) in enumerate(zip(ts, noises)):
        if t != '*':
            if index + 1 >= len(ts):
                continue
            if ts[index+1] != '*':
                xs.append([t])
                ys.append(ts[index+1] - noises[index+1])
            
    regressioner = LR()
    regressioner.fit(xs, ys)
    print regressioner.coef_
    return variance, regressioner.coef_
        
    


if __name__ == '__main__':
    time = 20
    v_rs = [run_test() for i in range(time)]
    vs = [v for v, r in v_rs]
    rs = [r for v, r in v_rs]
    rs =  [r for r, in rs]
    print sum(vs)/time
    print sum(rs)/time
