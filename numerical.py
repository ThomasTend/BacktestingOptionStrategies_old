import math

class numerical():
    def __init__(self):
        pass

    def bisection(self, f, a, b, error=1e-9):
        """
        Implements the bisection algorithm for root finding. 
        f is assumed to be lambda expressions. 
        """
        while (b-a)/2 > error:
            c = (a+b)/2
            if f(c) == 0:
                return c
            if (f(c) > 0 and f(a) > 0) or (f(c)<0 and f(a)<0):
                a = c
            else:
                b = c
        return b

    def newton_Raphson(self, f, f_prime, root = 1, error=1e-9):
        """
        Implements the Newton Raphson algorithm. 
        f and f_prime are assumed to be lambda expressions. 
        """
        print(f(root))
        try:
            if abs(f(root)) < error:
                return root
            else:
                root = root - f(root) / f_prime(root)
                return self.newton_Raphson(f, f_prime, root=root)
        except RecursionError as e:
            print("RecursionError occured, returning the last value of the root.")
            return root
