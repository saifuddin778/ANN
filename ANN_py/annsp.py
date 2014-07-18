from __future__ import division
import random
import math
import sys
sys.dont_write_bytecode = True

class annsp(object):
    def __init__(self, data, labels):
        if self.verify_dimensions(data):
            self.x = data
            self.y = labels
            self.weights = [random.uniform(1,len(data)) for _ in range(0, len(data[0])+1)]
            self.eta = 0.01
            self.train()

    def verify_dimensions(self, x):
        random_pick = len(x[random.randrange(1, len(x))])
        if sum(map(len, x))/len(x) == random_pick:
            return True
        else:
            return False

    def adder(self, w, v):
        return sum([i*j for i,j in zip(w,v)])

    def signum(self, val):
        if val >= 0:
            return 1
        else:
            return -1

    def scalar_prod(self, v, s):
        r = []
        for i in range(0, len(v)):
            r.append(s*v[i])
        return r
    
    def vector_adapt(self, w, x, ex, d):
        vector_sum = lambda x,y: [a+b for a,b in zip(x,y)]
        diff = self.scalar_prod(x, self.eta*(ex-d))
        updated_weights = vector_sum(w, diff)
        return updated_weights

    def train(self):
        for i in range(0, len(self.x)):
            self.x[i].insert(0,self.y[i])
        
        for i in range(0, 20):
            for j in range(0, len(self.x)):
                expected_decision = self.y[j]
                decision = self.signum(self.adder(self.weights, self.x[j]))
                if decision != expected_decision:
                    self.weights = self.vector_adapt(self.weights, self.x[j], expected_decision, decision)
                
    def classify(self, v):
        v.insert(0, 1)
        decision = self.signum(self.adder(self.weights, v))
        return decision

