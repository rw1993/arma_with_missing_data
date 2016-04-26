# -*- coding:utf8 -*-
import dpark


class ArmaPredicter(object,):

    @property
    def average_error(self):
        return sum(self.errors)/float(len(self.errors))

    def __init__(self, p, max_noise, missing_ability=3):
        #p = m + k p = 10
        self.p = p
        self.d = int(missing_ability * p) # more we choose, righter we get but slower
        self.ws = [0 for i in range(self.d**2-1)]
        self.xs = []
        self.errors = []
        self.max_noise = max_noise
        self.del_ = [0 for i in range(self.d**2-1)]
     
    def predict_and_fit(self, x):
        if len(self.xs) < self.d:
            # 对于前几个数据,不加预测 
            self.xs.append(x)
            self.errors.append(self.max_noise)
        else:
            self._expand_xs()
            if x == '*':
                return
            else:
                # predict x and append noise
                rec_x = 0.0
                for w, past_x in zip(self.ws, self.expand_xs):
                    rec_x += w * past_x
                self.errors.append(abs(rec_x-x))
            self.fit(rec_x, x)
            self.xs.append(x)
            return rec_x

    @property
    def F(self):
        return len([x for x in self.xs if x != '*'])


    def fit(self, y, x):
        del_this_turn = [(y - x)*past_x for past_x in self.expand_xs]
        self.learning_rate = 1.0 / (float(self.F)**0.5)
        def new_del_(x):
            return x[0]+x[1]
        self.del_ = map(new_del_, zip(self.del_, del_this_turn))
        del_sum = reduce(lambda x,y:x+y**2, self.del_)**0.5
        divide = max(1, self.learning_rate*del_sum*(2**(-self.d/2)))
        self.ws = map(lambda x:-self.learning_rate*x/divide, self.del_)

    def _expand_xs(self):
        past_xs = self.xs[-self.d:][::-1]

        def n_to_b(n):
            string = bin(n)[2:]
            b_s = [int(s) for s in string][::-1]
            return b_s

        def add(n):
            b_s = n_to_b(n)
            xs = [past_x for past_x, i in zip(past_xs, b_s)]
            if xs[-1] == '*':
                return 0
            else:
                for i, x in zip(b_s, xs)[:-1]:
                    if i < (x != '*'):
                        return 0
            return xs[-1]

        rdd = dpark.parallelize([i for i in range(2**self.d-1)], 5)
        rdd = rdd.map(add)
        self.expand_xs = rdd.collect()
