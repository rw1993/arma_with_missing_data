# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np


class ArmaPredicter(object, ):

    def __init__(self, p, missing_ability, max_x, learning_rate=0.003):
        self.p = p
        self.d = int(p * missing_ability)
        self.w_len = (1 + self.d) * self.d / 2
        self.D = max_x * (self.w_len ** 0.5)
        self.G = self.D
        self.ws = np.array([0 for i in range(self.w_len)])
        self.xs = []
        self.errors = []
        self.learning_rate = learning_rate
        self.last = np.array([0 for i in range(self.w_len)])

    def predict_and_fit(self, obersation_x):
        if len(self.xs) < self.d:
            self.xs.append(obersation_x)
            self.errors.append(obersation_x if obersation_x != '*' else 0)
        else:
            if obersation_x == '*':
                self.xs.append('*')
                self.errors.append(0)
            else:
                predict_x, expand_xs = self.predict_x()
                self.errors.append(predict_x - obersation_x)
                self.fit(predict_x, obersation_x, expand_xs)
                self.xs.append(obersation_x)
                return predict_x

    def fit(self, predict_x, obersation_x, expand_xs):
        self.last = self.last + (predict_x - obersation_x) * expand_xs
        self.ws = -self.learning_rate * self.last
        norm = np.linalg.norm(self.last)
        self.ws = self.ws / max(1.0, self.learning_rate / self.D * norm)
        
    def predict_x(self):
        past_d_xs = self.xs[-self.d:]
        expand_xs = self.expand_xs(past_d_xs)
        return self.ws.dot(expand_xs), expand_xs
        

    def expand_xs(self, past_d_xs):
        expand_xs = []
        missing_count = 0
        for i in range(len(past_d_xs)):
            this_x = past_d_xs[i]
            if this_x == '*':
                xs_for_expand = [0 for j in past_d_xs[i:]]
                expand_xs += xs_for_expand
                continue
            sub_sequece = past_d_xs[i:]
            xs_for_expand = [0 if x != '*' else this_x for x in sub_sequece]
            xs_for_expand[0] = this_x 
            expand_xs += xs_for_expand
        return np.array(expand_xs)


if __name__ == "__main__":
    p = ArmaPredicter(1, 4, 1)
    print 5,'*',3,2,1
    print  p.expand_xs(['5','*',3,2,1])
