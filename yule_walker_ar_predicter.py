# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy

class ArPredicter(object, ):

    def __init__(self, p, learning_rate=0.003):
        self.p = p
        self.xs = []
        self.errors = []

    def predict_and_fit(self, obersation_x):
        if len(self.xs) <= self.p:
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
                self.xs.append(obersation_x)
                return predict_x

    def predict_x(self):
        past_p_xs = self.xs[-self.p:][::-1]
        return self.ws.dot(past_p_xs)

    @property
    def ws(self):
        c0 = sum([x*x for x in self.xs]) / len(self.xs)
        c0 = float(c0)
        #print 'c0',c0
        xs = self.xs[::-1]

        def r(i):
            x_xis = [x * xs[index+i] for index, x in enumerate(xs) if index+i<len(xs)]
            #if len(x_xis) == 0:
                #print i,xs
            ci = sum(x_xis) / len(x_xis)

            return ci/c0

        rs_left = [r(i+1) for i in range(self.p)]
        rs_right = [r(i) for i in range(self.p)]
        r_matrix = [rs_right,]
        for i in range(self.p - 1):
            rs_right = [rs_right[-1]] + rs_right[:-1]
            r_matrix.append(rs_right)
        r_matrix =np.array(r_matrix)
        rs_left = np.array(rs_left)
        '''
        print r_matrix
        print '-------'
        print rs_left
        print '-------'
        '''
        return np.linalg.inv(r_matrix).dot(rs_left)

        
if __name__ == "__main__":
    p = ArPredicter(1)
    p.xs.append(0.3)
    p.xs.append(-0.3)
    p.xs.append(0.3)
    p.xs.append(-0.3)
    print p.ws
