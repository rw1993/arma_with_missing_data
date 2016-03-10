from arma import *
from arma_predicter import *
import random

p = ArmaPredicter(10, a.max_noise)

for index, x in enumerate(time_series):
    if random.random() > 0.2:
        p.predict_and_fit(x)
    else:
        p.predict_and_fit('*')
    print index
    print p.average_error, a.average_error(p.xs, p.F)


