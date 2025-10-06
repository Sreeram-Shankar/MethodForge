import mpmath as mp

def legendre_roots_and_weights(n):
    #initial guess for legendre polynomial roots using cos formula
    xs = [mp.cos(mp.pi * (4*k - 1) / (4*n + 2)) for k in range(1, n+1)]
    #define legendre polynomial of degree n
    Pn = lambda x: mp.legendre(n, x)

    def Pn_prime(x):
        #compute derivative of legendre polynomial
        if n == 0:
            return mp.mpf('0')
        elif n == 1:
            return mp.mpf('1')
        else:
            if abs(x*x - 1) < mp.eps:
                return mp.mpf('0')
            return n * (x * mp.legendre(n, x) - mp.legendre(n-1, x)) / (x*x - 1)

    #newton-raphson iteration to find exact roots
    for k in range(n):
        x = xs[k]
        for _ in range(100):
            fx = Pn(x)
            dfx = Pn_prime(x)
            if abs(dfx) < mp.eps:
                break
            step = fx / dfx
            x_new = x - step
            if mp.almosteq(x_new, x, rel_eps=mp.eps*100, abs_eps=mp.eps*100):
                break
            x = x_new
        xs[k] = x

    #compute quadrature weights using derivative formula
    ws = []
    for x in xs:
        dPn = Pn_prime(x)
        if abs(dPn) < mp.eps or abs(1 - x**2) < mp.eps:
            w = mp.mpf('0')
        else:
            w = mp.mpf(2) / ((1 - x**2) * (dPn**2))
        ws.append(w)
    return xs, ws

def power_vandermonde(c):
    #build vandermonde matrix with powers of c
    s = len(c)
    V = mp.matrix(s, s)
    for i in range(s):
        t = mp.mpf('1')
        for k in range(s):
            V[i, k] = t
            t *= c[i]
    return V

def invert(M):
    #matrix inversion with fallback methods
    s = M.rows
    I = mp.eye(s)
    return M**-1 if hasattr(M, '__pow__') else mp.lu_solve(M, I)  

def build_gauss_legendre_irk(s):
    #get legendre quadrature points and weights
    x, w = legendre_roots_and_weights(s)

    #transform from [-1,1] to [0,1] interval
    c = [ (xi + 1)/2 for xi in x ]
    b = [ wi/2 for wi in w ]

    #build and invert vandermonde matrix
    V = power_vandermonde(c)
    Vinv = invert(V)

    #compute monomial integration matrix
    sN = s
    monoint = mp.matrix(sN, sN)  # (i,k)
    for i in range(sN):
        for k in range(sN):
            monoint[i, k] = c[i]**(k+1) / mp.mpf(k+1)

    #compute butcher tableau matrix A
    A = monoint * Vinv
    A_list = [[A[i, j] for j in range(sN)] for i in range(sN)]
    
    #apply correction to ensure A·1 = c condition
    print(f"  Applying correction to ensure A·1 = c...")
    ones = [mp.mpf('1')]*sN
    A1 = [sum(A_list[i][j]*ones[j] for j in range(sN)) for i in range(sN)]
    
    for i in range(sN):
        correction = c[i] - A1[i]
        for j in range(sN):
            A_list[i][j] += correction / sN
    
    #verify correction worked
    print(f"  Final verification...")
    A1_corrected = [sum(A_list[i][j]*ones[j] for j in range(sN)) for i in range(sN)]
    err_final = mp.sqrt(sum((A1_corrected[i]-c[i])**2 for i in range(sN)))
    print(f"  Corrected error: {err_final}")
    
    return A_list, b, c

def nstr_fixed(x, digits=80):
    #format number to fixed precision string
    return mp.nstr(x, n=digits)

def write_A_b_c_triplets(A, b, c, basename, digits=80):
    #write butcher tableau coefficients to separate files
    s = len(b)

    #write A matrix coefficients
    with open(f"{basename}_A.txt", "w") as fa:
        for i in range(s):
            for j in range(s):
                fa.write(f"{i+1} {j+1} {nstr_fixed(A[i][j], digits)}\n")

    #write b vector coefficients
    with open(f"{basename}_b.txt", "w") as fb:
        for j in range(s):
            fb.write(f"{j+1} {nstr_fixed(b[j], digits)}\n")

    #write c vector coefficients
    with open(f"{basename}_c.txt", "w") as fc:
        for i in range(s):
            fc.write(f"{i+1} {nstr_fixed(c[i], digits)}\n")

    #write combined triplets format
    with open(f"{basename}_triplets.txt", "w") as ft:
        # A
        for i in range(s):
            for j in range(s):
                ft.write(f"{i+1} {j+1} {nstr_fixed(A[i][j], digits)}\n")
        # c
        for i in range(s):
            ft.write(f"{i+1} 0 {nstr_fixed(c[i], digits)}\n")
        # b
        for j in range(s):
            ft.write(f"0 {j+1} {nstr_fixed(b[j], digits)}\n")

def sanity_checks(A, b, c):
    #verify butcher tableau conditions
    s = len(b)
    ones = [mp.mpf('1')]*s
    A1 = [sum(A[i][j]*ones[j] for j in range(s)) for i in range(s)]
    err1 = mp.sqrt(sum((A1[i]-c[i])**2 for i in range(s)))
    sb = sum(b)
    return err1, sb



if __name__ == "__main__":
    #main execution for gauss-legendre quadrature
    mp.mp.dps = 200  

    for s in (6, 7):
        #build butcher tableau for different stage counts
        A, b, c = build_gauss_legendre_irk(s)
        err1, sb = sanity_checks(A, b, c)
        print(f"\nGauss–Legendre IRK: stages={s}, order={2*s}")
        print("||A·1 - c||_2 =", nstr_fixed(err1, 30))
        print("sum(b)        =", nstr_fixed(sb, 30))
        print("c in (0,1)?   =", all((ci > 0) and (ci < 1) for ci in c))

        #save coefficients to files
        base = f"gauss_legendre_s{s}"
        write_A_b_c_triplets(A, b, c, base, digits=80)
        print(f"Written: {base}_A.txt, {base}_b.txt, {base}_c.txt, {base}_triplets.txt")
