import math
from methods.secant import secant_method

def main():
    f = lambda x: x**2 - 2
    result = secant_method(f, x0=1.0, x1=2.0)

    print("\nFinal Result")
    print("Root:", result.root)
    print("Iterations:", result.iterations)
    print("Converged:", result.converged)
    print("Verification f(root):", f(result.root))

if __name__ == "__main__":
    main()