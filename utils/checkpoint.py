import math

class CkptIntervalManagerConst:
    def __init__(self, interval):
        self.interval = interval
    

    def next(self, cur_step):
        return (cur_step + self.interval - 1) // self.interval * self.interval
    

    def is_ckpt_step(self, step):
        return cur_step % self.interval == 0


    def n_th(self):
        return n * interval


class CkptIntervalManagerExponential:
    """
    c[n] = floor(c[1] * (r^n - 1) / (r - 1))
    """
    def __init__(self, c1, r):
        assert r > 1
        self.c1 = c1
        self.r = r
    

    def next(self, cur_step):
        r, c1 = self.r, self.c1
        a = r
        while math.floor(c1 * (a - 1) / (r - 1)) <= cur_step:
            a *= r
        return math.floor(c1 * (a - 1) / (r - 1))


    def is_ckpt_step(self, step):
        r, c1 = self.r, self.c1
        a = r
        while math.floor(c1 * (a - 1) / (r - 1)) < cur_step:
            a *= r
        return math.floor(c1 * (a - 1) / (r - 1)) == step

    
    def n_th(self, n):
        return math.floor(self.c1 * (self.r ** n - 1) / (self.r - 1))