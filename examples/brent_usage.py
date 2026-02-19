import math
from methods.brent import brent_method

def main():
    f = lambda x: x**2 - 2
    result = brent_method(f, a=1.0, b=2.0)

    print("Root:", result.root)
    print("Iterations:", result.iterations)
    print("Converged:", result.converged)

if __name__ == "__main__":
    main()