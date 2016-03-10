# -*- coding:utf8 -*-


class ArmaPredicter(object,):
    

    @property
    def average_error(self):
        return sum(self.errors)/float(len(self.errors))

    def __init__(self, p, max_noise, learning_rate=1.0, missing_ability=1.5):
        #p = m + k p = 10
        self.p = p
        self.d = int(missing_ability * p) # more we choose, righter we get but slower
        self.ws = [0 for i in range(self.d**2-1)]
        self.xs = []
        self.errors = []
        self.max_noise = max_noise
        self.learning_rate = learning_rate
        self.del_ = [0 for i in range(self.d**2-1)]

    def predict_and_fit(self, x):
        if len(self.xs) < self.d:
            # 对于前几个数据,不加预测 
            self.xs.append(x)
            self.errors.append(self.max_noise)
        else:
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

    @property
    def F(self):
        return len([x for x in self.xs if x != '*'])

    def fit(self, y, x):
        del_this_turn = [(y - x)*past_x for past_x in self.expand_xs]
        self.learning_rate = 1 / (float(self.F)**0.5)
        def new_del_(x):
            return x[0]+x[1]

        self.del_ = map(new_del_, zip(self.del_, del_this_turn))
        del_sum = reduce(lambda x,y:x+y**2, self.del_)**0.5
        divide = max(1, self.learning_rate*del_sum*(2**(-self.d/2)))
        self.ws = map(lambda x:-self.learning_rate*x/divide, self.del_)

    @property
    def expand_xs(self):
        past_d_xs = self.xs[-self.d:]

        def n_to_b(n):
            b = [0 for i in range(self.d)]
            string = bin(n)[2:]
            b_s = [int(s) for s in string]
            for index, s in enumerate(b_s):
                b[index] = s
            return b

        def add(n):
            b = n_to_b(n)
            xs = reversed(past_d_xs)
            rt = 0
            for i, x in zip(b, xs):
                if i <= (x != '*'):
                    return 0
                else:
                    if i == 1:
                        rt = x
            return rt
        return [add(i) for i in range(1, 2**(self.d)-1)]
