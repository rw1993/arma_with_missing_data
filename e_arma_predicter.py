# -*- coding:utf8 -*-


class ArmaPredicter(object,):

    def __init__(self, p, max_noise, missing_ability=3,
                 learning_rate=0.003):
        #p = m + k p = 10
        self.p = p
        self.d = int(missing_ability * p) # more we choose, righter we get but slower
        self.xs = []
        self.errors = []
        self.max_noise = max_noise
        self.predict_xs =[0.0]
        self.learning_rate = learning_rate

    def predict_and_fit(self, x):
        if len(self.xs) <= self.d:
            self.xs.append(x)
            self.predict_xs.append(0.0)
            self.errors.append(self.max_noise)
            return
        else:
            self.xs.append(x)
            if x == '*':
                return
            error = abs(self.predict_xs[-1] - x)
            self.errors.append(error)
            self.predict()

    def predict(self):
        old_x = self.predict_xs[-1]
        new_x = old_x - self.learning_rate * self.Err * self.K
        self.predict_xs.append(new_x)

    @property
    def Err(self):
        y = self.xs[-1]
        wx = self.predict_xs[-1]
        return wx - y
    
    @property
    def K(self):
        xs1 = self.xs[-self.d:][::-1]
        xs2 = self.xs[:-1][-self.d:][::-1]
        rt = 0.0

        def C(k):
            count = 0
            for i in range(k-1):
                if xs1[i] == '*' and xs2[i] == '*':
                    count += 1
            return count

        for i in range(self.d):
            if xs1[i] == '*':
                continue
            if xs2[i] == '*':
                continue
            rt += 2 ** C(i) * xs1[i] * xs2[i]

        return rt
