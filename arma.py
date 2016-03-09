# -*- coding:utf8 -*-
import numpy


class ARMA(object, ):


    @property
    def average_error(self):
        return sum(self.noises) / float(len(self.xs))


    def __init__(self, alphas, betas, sigma):
        self.alphas = alphas
        self.betas = betas
        self.sigma = sigma
        self.init_first_few_ys()
        self.generater = self.generate_data()

    def init_first_few_ys(self):
        self.xs = [self.noise() for alpha in self.alphas]
        self.noises = self.xs

    @property
    def current_xs(self):
        return self.xs[-len(self.alphas):]

    @property
    def current_noises(self):
        return self.noises[-len(self.betas):]
        
    def noise(self):
        return numpy.random.normal(0, self.sigma, 1)[0]

    def generate_data(self):
        while True:
            # AR part
            sum_ar = 0.0
            for alpha, x in zip(self.alphas, self.current_xs):
                sum_ar += alpha * x

            # MA part
            sum_ma = 0.0
            for beta, noise in zip(self.betas, self.current_noises):
                sum_ma += beta * noise

            self.xs.append(sum_ar+sum_ma)
            self.noises.append(sum_ma)
            yield sum_ar+sum_ma



if __name__ == '__main__':
    a = ARMA([0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 0.3)
    print a.xs
    print a.current_xs
    print a.current_noises
    print [a.generater.next() for i in range(10)] 
    print a.average_error

