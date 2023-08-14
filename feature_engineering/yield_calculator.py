"""
yield calculator class
convert bond price to a yield in a generic way
"""

class YldCalc:
    yld = None
    def __int__(self, price, maturity, cpn, freq):
        self.price = price
        self.maturity = maturity
        self.cpn = cpn
        self.freq = freq

    def calc_price_to_yield(self):

        return YldCalc.yld

    def calc_yield_to_spread(self, benchmark):
        if YldCalc.yld:
            return YldCalc.yld - benchmark
        else:
            return YldCalc.calc_price_to_yield(self) - benchmark