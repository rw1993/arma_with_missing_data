# -*- coding:utf8 -*-
import numpy

class ARMA(object, ):
    

    @property
    def max_noise(self):
        return max(map(abs, self.noises))

    def average_error(self, other_xs, count):
        sum_noise = 0.0
        for x1, noise in zip(other_xs, self.noises):
            if x1 != '*':
                sum_noise += abs(noise)

        return sum_noise / float(count)

    def __init__(self, alphas, betas, sigma):
        self.alphas = alphas
        self.betas = betas
        self.sigma = sigma
        self.init_first_few_xs()
        self.generater = self.generate_data()

    def init_first_few_xs(self):
        self.xs = [1 for alpha in self.alphas]
        self.noises = [self.noise() for alpha in self.alphas]

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

            n = self.noise()
            x = sum_ar + sum_ma +n

            self.xs.append(x)
            self.noises.append(n)
            yield x



#a = ARMA([0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 0.3)
a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [-0.2, -0.3], 0.3)
time_series = [a.generater.next() for i in range(2000)]
print time_series[-1]
