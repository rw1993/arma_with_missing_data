# -*- coding:utf8 -*-
import numpy

class AR(object, ):
    
    @property
    def max_noise(self):
        return max(map(abs, self.noises))

    def __init__(self, alphas, sigma):
        self.alphas = alphas
        self.sigma = sigma
        self.init_first_few_xs()
        self.generater = self.generate_data()

    def init_first_few_xs(self):
        self.xs = [0 for alpha in self.alphas]
        self.noises = [self.noise() for alpha in self.alphas]

    @property
    def current_xs(self):
        return self.xs[-len(self.alphas):]

    def noise(self):
        if self.sigma == 0:
            return 0
        return numpy.random.normal(0, self.sigma, 1)[0]

    def generate_data(self):
        while True:
            # AR part
            sum_ar = 0.0
            for alpha, x in zip(self.alphas, self.current_xs):
                sum_ar += alpha * x

            n = self.noise()
            x = sum_ar + n

            self.xs.append(x)
            if abs(x) > 1:
                if x > 0:
                    x = 1
                else:
                    x = -1
            yield x
