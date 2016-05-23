# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np


class ArmaPredicter(object, ):

    def __init__(self, p, missing_ability, max_x, learning_rate=0.003):
        self.p = p
        self.d = int(p * missing_ability)
        self.D = max_x * (2 ** (self.d / 2))
        self.G = self.D
        self.ws = np.array([0 for i in range(2 ** self.d -1)])
        self.xs = []
        self.errors = []
        self.learning_rate = learning_rate
        self.last = np.array([0 for i in range(2 ** self.d -1)])

    def predict_and_fit(self, obersation_x):
        if len(self.xs) < self.d:
            self.xs.append(obersation_x)
            self.errors.append(obersation_x)
        else:
            if obersation_x == '*':
                self.xs.append('*')
                self.errors.append(0)
            else:
                predict_x, expand_xs = self.predict_x()
                self.errors.append(predict_x - obersation_x)
                self.fit(predict_x, obersation_x, expand_xs)
                self.xs.append('*')
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

        def bin_represent(past_d_xs):
            return [1 if x != '*' else 0 for x in past_d_xs]

        b_xs = bin_represent(past_d_xs)

        for i in range(1, 2 ** self.d):
            bi = [int(num) for num in str(bin(i)[2:])]
            locate = len(bi)
            compare = b_xs[-len(bi):]
            if compare[0] == 0:
                expand_xs.append(0)
                continue
            if_add = True
            for index, (num1, num2) in enumerate(zip(compare, bi)):
                if index == 0:
                    pass
                else:
                    if num1 > num2:
                        if_add = False
                        continue
            if if_add:
                expand_xs.append(past_d_xs[-locate])
            else:
                expand_xs.append(0)
        return np.array(expand_xs)


if __name__ == "__main__":
    p = ArmaPredicter(1, 4, 1)
    expand_xs = p.expand_xs(['5','*',3,2,1])
