# def check(kwargs={}):
#     for kw in kwargs:
#         print(kw,kwargs[kw])

# def test(bill, kwargs1={}, kwargs2={}):
#     print(bill)
#     for kw in kwargs1:
#         print(kw,kwargs1[kw])
#     check(kwargs=kwargs2)

# d1 = {'thom':1,'david':2}
# d2 = {'luke':7,'joel':8}
# test(0,d1,d2)

class KWAs:
    def __init__(self, algo):
        self.algo = algo

    def set_kwas_array(self,kwas)

def tester(**kwas):
    print(kwas)

tester(thom=1,david=2)