import mpmath as mp
mp.dps = 100

#jacobi polynomial function
def jacobi_P(n, alpha, beta, x):
    x = mp.mpf(x)
    return mp.binomial(n + alpha, n) * mp.hyp2f1(-n, n + alpha + beta + 1, alpha + 1, (1 - x) / 2)

#find roots of jacobi polynomial
def jacobi_roots_radau_interior(n, dps_scan=None):
    alpha, beta = mp.mpf(1), mp.mpf(0)
    f = lambda x: jacobi_P(n, alpha, beta, x)

    if dps_scan is None:
        dps_scan = max(2000, 400 * n)
    xs = [mp.mpf(-1) + 2*i/(dps_scan-1) for i in range(dps_scan)]
    fs = [f(x) for x in xs]

    roots = []
    for i in range(dps_scan-1):
        a, b = xs[i], xs[i+1]
        fa, fb = fs[i], fs[i+1]
        if fa == 0:
            roots.append(a)
            continue
        if fa*fb < 0:
            for _ in range(200):
                m = (a + b) / 2
                fm = f(m)
                if abs(fm) < mp.mpf('1e-70') or abs(b - a) < mp.mpf('1e-50'):
                    roots.append(m)
                    break
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm

    roots = [r for r in roots if -1 < r < 1]
    roots = sorted(set([mp.nstr(r, 60) for r in roots]))
    roots = [mp.mpf(r) for r in roots]

    if len(roots) != n:
        if dps_scan < 20000:
            return jacobi_roots_radau_interior(n, dps_scan=20000)
        raise RuntimeError(f"Expected {n} interior roots, got {len(roots)}")

    return roots

#generate radau nodes
def radau_right_nodes_on_01(s):
    if s < 1:
        raise ValueError("Radau quadrature requires s >= 1.")
    if s == 1:
        return [mp.mpf('1')]
    interior = jacobi_roots_radau_interior(s - 1)
    x_all = interior + [mp.mpf(1)]
    c = [ (x + 1) / 2 for x in x_all ]
    return c

#lagrange basis polynomial
def lagrange_basis(c, j):
    xj = c[j]
    others = [c[k] for k in range(len(c)) if k != j]
    denom = mp.mpf(1)
    for xk in others:
        denom *= (xj - xk)
    def Lj(x):
        num = mp.mpf(1)
        for xk in others:
            num *= (x - xk)
        return num / denom
    return Lj

#build tableau matrices
def build_A_b(c):
    s = len(c)
    A = [[mp.mpf(0) for _ in range(s)] for _ in range(s)]
    b = [mp.mpf(0)] * s
    for j in range(s):
        Lj = lagrange_basis(c, j)
        b[j] = mp.quad(Lj, [0, 1])
        for i in range(s):
            A[i][j] = mp.quad(Lj, [0, c[i]])
    return A, b

#format number string
def nstr_fixed(x, digits=80):
    return mp.nstr(x, n=digits)

#write tableau to files
def write_A_b_c_triplets(A, b, c, basename, digits=80):
    s = len(b)
    with open(f"{basename}_A.txt", "w") as fa:
        for i in range(s):
            for j in range(s):
                fa.write(f"{i+1} {j+1} {nstr_fixed(A[i][j], digits)}\n")
    with open(f"{basename}_b.txt", "w") as fb:
        for j in range(s):
            fb.write(f"{j+1} {nstr_fixed(b[j], digits)}\n")
    with open(f"{basename}_c.txt", "w") as fc:
        for i in range(s):
            fc.write(f"{i+1} {nstr_fixed(c[i], digits)}\n")
    with open(f"{basename}_triplets.txt", "w") as ft:
        for i in range(s):
            for j in range(s):
                ft.write(f"{i+1} {j+1} {nstr_fixed(A[i][j], digits)}\n")
        for i in range(s):
            ft.write(f"{i+1} 0 {nstr_fixed(c[i], digits)}\n")
        for j in range(s):
            ft.write(f"0 {j+1} {nstr_fixed(b[j], digits)}\n")

#verify tableau correctness
def check_tableau(A, b, c):
    print("\nRow-sum checks (ΣA[i,:] ≈ c[i]):")
    for i in range(len(c)):
        ssum = mp.fsum(A[i])
        diff = ssum - c[i]
        print("  i={}: ΣA={}  c={}  diff={}".format(
              i+1, mp.nstr(ssum, 25), mp.nstr(c[i], 25), mp.nstr(diff, 5)))
    print("\nMoment checks (b·c^k ≈ 1/(k+1)):")
    for k in range(min(10, 2*len(c))):
        lhs = mp.fsum([b[j] * c[j]**k for j in range(len(c))])
        rhs = mp.mpf(1)/(k+1)
        err = lhs - rhs
        print("  k={:2d}: {}  target={}  err={}".format(
              k, mp.nstr(lhs, 25), mp.nstr(rhs, 25), mp.nstr(err, 5)))

#main radau builder function
def build_radau_irk(s):
    print(f"  Computing Radau–IIA nodes for s={s} …")
    c = radau_right_nodes_on_01(s)
    print(f"  Building A and b by integrating Lagrange basis …")
    A, b = build_A_b(c)
    print(f"  Verifying tableau …")
    check_tableau(A, b, c)
    return A, b, c

#main execution block
if __name__ == "__main__":
    mp.mp.dps = 100  
    for s in (7, 8):
        print(f"\nComputing Radau–IIA IRK with s={s} stages (order {2*s-1}) …")
        try:
            A, b, c = build_radau_irk(s)
            err_rows = mp.sqrt(mp.fsum([(mp.fsum(A[i]) - c[i])**2 for i in range(len(c))]))
            sb = mp.fsum(b)
            print("||A·1 - c||_2 =", nstr_fixed(err_rows, 30))
            print("sum(b)        =", nstr_fixed(sb, 30))
            print("c[-1] = 1?    =", mp.almosteq(c[-1], mp.mpf('1')))
            print("c[0]  > 0?    =", c[0] > 0)

            base = f"radau_s{s}"
            write_A_b_c_triplets(A, b, c, base, digits=70)
            print(f"Written: {base}_A.txt, {base}_b.txt, {base}_c.txt, {base}_triplets.txt")
        except Exception as e:
            print(f"Failed for s={s}: {e}")
