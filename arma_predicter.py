# -*- coding:utf8 -*-


class ArmaPredicter(object,):


    def __init__(self, p, max_noise, learning_rate):
        #p = m + k p = 10
        self.p = p
        self.d = 2 * p # more we choose, righter we get but slower
        self.ws = [0 for i in range(self.d)]
        self.xs = []
        self.errors = []
        self.max_noise = max_noise
        self.learning_rate = learning_rate

    def predict_and_fit(self, x):
        if len(self.xs) < self.d:
            # 对于前几个数据,不加预测 
            self.xs.append(x)
            self.errors.append(self.max_noise)
        else:
            if x == '*':
                return 0
            else:
                # predict x and append noise
                rec_x = 0.0
                for w, past_x in zip(self.ws, self.expand_xs):
                    rec_x += w * past_x
                self.errors.append(abs(rec_x-x))
            self.xs.append(x)
            self.fit()

    def fit(self):
        pass 


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
                        
        return [add(i) for i in range(1, 2**(self.d)-1)    


        


        


