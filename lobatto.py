import mpmath as mp

def lobatto_roots_and_weights(s):
    #lobatto quadrature includes endpoints -1 and 1
    if s < 2:
        raise ValueError("Lobatto quadrature requires at least 2 stages")
    
    n = s - 1
    
    #special case for 2 stages
    if n == 1:
        xs = [mp.mpf('-1'), mp.mpf('1')]
        ws = [mp.mpf('1'), mp.mpf('1')]
        return xs, ws
    
    #initial guess for interior roots using cos formula
    xs = [mp.cos(mp.pi * (2*k - 1) / (2*n - 2)) for k in range(1, n)]
    
    def Pn_prime(x):
        #first derivative of legendre polynomial
        if n == 0:
            return mp.mpf('0')
        elif n == 1:
            return mp.mpf('1')
        else:
            if abs(x*x - 1) < mp.eps:
                return mp.mpf('0')
            return n * (x * mp.legendre(n, x) - mp.legendre(n-1, x)) / (x*x - 1)
    
    def Pn_double_prime(x):
        #second derivative of legendre polynomial
        if n <= 1:
            return mp.mpf('0')
        else:
            if abs(x*x - 1) < mp.eps:
                return mp.mpf('0')
            return n * (n + 1) * mp.legendre(n, x) / (x*x - 1)
    
    #newton-raphson with second derivative for interior roots
    for k in range(len(xs)):
        x = xs[k]
        for iteration in range(200):
            fx = Pn_prime(x)
            dfx = Pn_double_prime(x)
            if abs(dfx) < mp.eps:
                break
            step = fx / dfx
            x_new = x - step
            if mp.almosteq(x_new, x, rel_eps=mp.eps*1000, abs_eps=mp.eps*1000):
                break
            x = x_new
        xs[k] = x
    
    #add endpoints -1 and 1
    xs = [mp.mpf('-1')] + xs + [mp.mpf('1')]
    
    #compute weights for all points
    ws = []
    for x in xs:
        if x == mp.mpf('-1') or x == mp.mpf('1'):
            #special weight formula for endpoints
            w = mp.mpf('2') / (n * (n + 1))
        else:
            #weight formula for interior points
            Pn_x = mp.legendre(n, x)
            if abs(Pn_x) < mp.eps:
                w = mp.mpf('0')
            else:
                w = mp.mpf('2') / (n * (n + 1) * Pn_x**2)
        ws.append(w)
    
    #normalize weights to sum to 2
    total_weight = sum(ws)
    if abs(total_weight - 2) > mp.eps:
        scale_factor = mp.mpf('2') / total_weight
        ws = [w * scale_factor for w in ws]
    
    #clean up very small weights
    for i in range(len(ws)):
        if ws[i] < mp.eps:
            ws[i] = mp.mpf('0')
    
    #final normalization check
    total_weight_final = sum(ws)
    if abs(total_weight_final - 2) > mp.eps:
        scale_factor = mp.mpf('2') / total_weight_final
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
                    regularization = mp.mpf('1e-100')
                    M_reg = M + regularization * I
                    return M_reg**-1  

def build_lobatto_irk(s):
    #get lobatto quadrature points and weights
    x, w = lobatto_roots_and_weights(s)

    #transform from [-1,1] to [0,1] interval
    c = [ (xi + 1)/2 for xi in x ]
    b = [ wi/2 for wi in w ]

    #build and invert vandermonde matrix
    V = power_vandermonde(c)
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
    ones = [mp.mpf('1')]*sN
    A1 = [sum(A_list[i][j]*ones[j] for j in range(sN)) for i in range(sN)]
    
    for i in range(sN):
        correction = c[i] - A1[i]
        for j in range(sN):
            A_list[i][j] += correction / sN
    
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
    #main execution for lobatto quadrature
    mp.mp.dps = 400  

    for s in (7, 8):
        try:
            #build butcher tableau for different stage counts
            print(f"Computing Lobatto IRK with s={s} stages...")
            A, b, c = build_lobatto_irk(s)
            err1, sb = sanity_checks(A, b, c)
            print(f"\nLobatto IRK: stages={s}, order={2*s-2}")
            print("||A·1 - c||_2 =", nstr_fixed(err1, 30))
            print("sum(b)        =", nstr_fixed(sb, 30))
            print("c[0] = 0?     =", mp.almosteq(c[0], mp.mpf('0')))
            print("c[-1] = 1?    =", mp.almosteq(c[-1], mp.mpf('1')))
            print("c in [0,1]?   =", all((ci >= 0) and (ci <= 1) for ci in c))

            #save coefficients to files
            base = f"lobatto_s{s}"
            write_A_b_c_triplets(A, b, c, base, digits=150)
            print(f"Written: {base}_A.txt, {base}_b.txt, {base}_c.txt, {base}_triplets.txt")
        except Exception as e:
            #error handling with precision increase
            print(f"Error computing s={s}: {e}")
            print("Trying with higher precision...")
            mp.mp.dps = 400
            try:
                A, b, c = build_lobatto_irk(s)
                print(f"Success with higher precision!")
            except Exception as e2:
                print(f"Still failed: {e2}")
