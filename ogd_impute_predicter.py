# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy

class ArPredicter(object, ):

    def __init__(self, p, learning_rate=0.003, w_range=1):
        self.p = p
        self.ws = [0 for i in range(self.p)]
        self.xs = []
        self.errors = []
        self.w_range = w_range

    def predict_and_fit(self, obersation_x):
        if len(self.xs) < self.p:
            self.xs.append(obersation_x)
            self.errors.append(obersation_x if obersation_x != '*' else 0)
            if obersation_x == '*':
                print 'error!'
                return -1
            return 0
        else:
            if obersation_x == '*':
                predict_x = self.predict_x()
                self.xs.append(predict_x)
                self.errors.append(0)
                return predict_x
            else:
                predict_x = self.predict_x()
                self.errors.append(predict_x - obersation_x)
                self.fit(predict_x, obersation_x)
                self.xs.append(obersation_x)
                return predict_x

    def predict_x(self):
        past_p_xs = np.array(self.xs[-self.p:])
        ws = np.array(self.ws)
        return ws.dot(past_p_xs)

    def fit(self, predict_x, obersation_x):
        past_p_xs = self.xs[-self.p:]
        deltas = map(lambda x:x*(predict_x-obersation_x), past_p_xs)
        ws_deltas = zip(self.ws, deltas)
        def update(w_d):
            result = w_d[0] - w_d[1]
            if abs(result) > self.w_range:
                return self.w_range if result > 0 else - self.w_range
            return result
        self.ws = map(update, ws_deltas)

if __name__ == "__main__":
    p = ArPredicter(1)
    p.xs.append(0.3)
    p.xs.append(-0.3)
    p.xs.append(0.3)
    p.xs.append(-0.3)
    print p.ws
