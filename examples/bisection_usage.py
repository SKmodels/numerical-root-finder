import math 
from methods.bisection import bisection_method

def main():
    f = lambda x: x**2 - 2
    result = bisection_method(f, 1.0, 2.0)

    print("Bisection root:", result.root)
    print("Iterations:", result.iterations)
    print("Converged:", result.converged)
    print("Check f(root):", f(result.root))
    print("Expected sqrt(2):", math.sqrt(2))

if __name__ == "__main__":
    main()