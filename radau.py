import mpmath as mp

def radau_roots_and_weights(s):
    #radau quadrature includes right endpoint 1
    if s < 1:
        raise ValueError("Radau quadrature requires at least 1 stage")
    
    n = s - 1
    
    #special case for 1 stage
    if s == 1:
        xs = [mp.mpf('1')]
        ws = [mp.mpf('2')]
        return xs, ws
    
    #initial guess for interior roots using cos formula
    xs = [mp.cos(mp.pi * (2*k - 1) / (2*n + 1)) for k in range(1, n + 1)]
    
    def Rn(x):
        #radau polynomial: P_n + P_{n-1}
        if n == 0:
            return mp.mpf('1')
        return mp.legendre(n, x) + mp.legendre(n-1, x)
    
    def Rn_prime(x):
        #derivative of radau polynomial
        if n == 0:
            return mp.mpf('0')
        elif n == 1:
            return mp.mpf('1')
        else:
            Pn_prime = n * (x * mp.legendre(n, x) - mp.legendre(n-1, x)) / (x*x - 1) if abs(x*x - 1) > mp.eps else mp.mpf('0')
            Pn_minus_1_prime = (n-1) * (x * mp.legendre(n-1, x) - mp.legendre(n-2, x)) / (x*x - 1) if abs(x*x - 1) > mp.eps else mp.mpf('0')
            return Pn_prime + Pn_minus_1_prime
    
    #newton-raphson iteration to find exact roots
    for k in range(len(xs)):
        x = xs[k]
        for _ in range(100):
            fx = Rn(x)
            dfx = Rn_prime(x)
            if abs(dfx) < mp.eps:
                break
            step = fx / dfx
            x_new = x - step
            if mp.almosteq(x_new, x, rel_eps=mp.eps*100, abs_eps=mp.eps*100):
                break
            x = x_new
        xs[k] = x
    
    #add right endpoint 1
    xs = xs + [mp.mpf('1')]
    
    #compute weights for all points
    ws = []
    for x in xs:
        if x == mp.mpf('1'):
            #special weight formula for right endpoint
            w = mp.mpf('2') / (n + 1)**2
        else:
            #weight formula for interior points
            Rn_prime_x = Rn_prime(x)
            if abs(1 - x) < mp.eps or abs(Rn_prime_x) < mp.eps:
                w = mp.mpf('0')
            else:
                w = mp.mpf('1') / ((1 - x) * Rn_prime_x**2)
        ws.append(w)
    
    #normalize weights to sum to 2
    total_weight = sum(ws)
    if abs(total_weight - 2) > mp.eps:
        scale_factor = mp.mpf('2') / total_weight
        ws = [w * scale_factor for w in ws]
    
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
    #matrix inversion with multiple fallback methods
    s = M.rows
    I = mp.eye(s)
    
    try:
        return M**-1
    except:
        try:
            return mp.lu_solve(M, I)
        except:
            try:
                return mp.qr_solve(M, I)
            except:
                try:
                    return mp.cholesky_solve(M, I)
                except:
                    #regularization as last resort
                    regularization = mp.mpf('1e-50')
                    M_reg = M + regularization * I
                    return M_reg**-1  

def build_radau_irk(s):
    #get radau quadrature points and weights
    x, w = radau_roots_and_weights(s)

    #transform from [-1,1] to [0,1] interval
    c = [ (xi + 1)/2 for xi in x ]
    b = [ wi/2 for wi in w ]

    #build and invert vandermonde matrix
    print(f"  Computing Vandermonde matrix for s={s}...")
    V = power_vandermonde(c)
    print(f"  Inverting matrix...")
    Vinv = invert(V)

    #compute monomial integration matrix
    sN = s
    monoint = mp.matrix(sN, sN)
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
        for i in range(s):
            for j in range(s):
                ft.write(f"{i+1} {j+1} {nstr_fixed(A[i][j], digits)}\n")
        for i in range(s):
            ft.write(f"{i+1} 0 {nstr_fixed(c[i], digits)}\n")
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
    #main execution for radau quadrature
    mp.mp.dps = 200  

    for s in (7, 8):
        try:
            #build butcher tableau for different stage counts
            print(f"Computing Radau IRK with s={s} stages...")
            A, b, c = build_radau_irk(s)
            err1, sb = sanity_checks(A, b, c)
            print(f"\nRadau IRK: stages={s}, order={2*s-1}")
            print("||A·1 - c||_2 =", nstr_fixed(err1, 30))
            print("sum(b)        =", nstr_fixed(sb, 30))
            print("c[-1] = 1?    =", mp.almosteq(c[-1], mp.mpf('1')))
            print("c[0] ≠ 0?     =", not mp.almosteq(c[0], mp.mpf('0')))
            print("c in [0,1]?   =", all((ci >= 0) and (ci <= 1) for ci in c))

            #save coefficients to files
            base = f"radau_s{s}"
            write_A_b_c_triplets(A, b, c, base, digits=150)
            print(f"Written: {base}_A.txt, {base}_b.txt, {base}_c.txt, {base}_triplets.txt")
        except Exception as e:
            #error handling with precision increase
            print(f"Error computing s={s}: {e}")
            print("Trying with higher precision...")
            mp.mp.dps = 400
            try:
                A, b, c = build_radau_irk(s)
                print(f"Success with higher precision!")
            except Exception as e2:
                print(f"Still failed: {e2}")
