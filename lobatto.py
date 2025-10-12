import mpmath as mp
mp.dps = 100

#lobatto roots and weights generator
def lobatto_roots_and_weights(s):
    if s < 2:
        raise ValueError("Lobatto quadrature requires at least 2 stages")
    n = s - 1

    xs = [mp.mpf(-1)]
    Pn = lambda x: mp.legendre(n, x)
    Pn_prime = lambda x: n*(x*Pn(x) - mp.legendre(n-1, x)) / (x**2 - 1)

    for k in range(1, n):
        x0 = mp.cos(mp.pi * (k) / (n))
        x = x0
        for _ in range(100):
            fx = Pn_prime(x)
            dfx = mp.diff(Pn_prime, x)
            if dfx == 0:
                break
            x_new = x - fx/dfx
            if mp.almosteq(x_new, x, rel_eps=mp.eps*100, abs_eps=mp.eps*100):
                break
            x = x_new
        xs.append(x)

    xs.append(mp.mpf(1))

    ws = []
    for i, x in enumerate(xs):
        if i == 0 or i == s-1:
            w = mp.mpf(2) / (n*(n+1))
        else:
            w = mp.mpf(2) / (n*(n+1) * (Pn(x)**2))
        ws.append(w)

    return xs, ws

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

#build lobatto iic irk tableau
def build_lobatto_IIIC_irk(s):
    print(f"Computing Lobatto–IIIC IRK (s={s}) …")
    xs, ws = lobatto_roots_and_weights(s)
    c = [(x + 1)/2 for x in xs]
    b = [w/2 for w in ws]

    A = [[mp.mpf(0) for _ in range(s)] for _ in range(s)]
    for j in range(s):
        Lj = lagrange_basis(c, j)
        for i in range(s):
            A[i][j] = mp.quad(Lj, [0, c[i]])

    print("Row-sum check (ΣA[i,:] ≈ c[i]):")
    for i in range(s):
        rs = mp.fsum(A[i])
        diff = rs - c[i]
        print(f"  i={i+1}: ΣA={mp.nstr(rs,25)}  c={mp.nstr(c[i],25)}  diff={mp.nstr(diff,5)}")

    print("Moment check (b·c^k ≈ 1/(k+1)):")
    for k in range(min(10, 2*s-2)):
        lhs = mp.fsum([b[j]*c[j]**k for j in range(s)])
        rhs = mp.mpf(1)/(k+1)
        err = lhs - rhs
        print(f"  k={k:2d}: {mp.nstr(lhs,25)}  target={mp.nstr(rhs,25)}  err={mp.nstr(err,5)}")

    return A, b, c

#format number string
def nstr_fixed(x, digits=80): return mp.nstr(x, n=digits)

#write tableau to files
def write_A_b_c_triplets(A, b, c, basename, digits=80):
    s = len(b)
    with open(f"{basename}_A.txt", "w") as fa:
        for i in range(s):
            for j in range(s):
                fa.write(f"{i+1} {j+1} {nstr_fixed(A[i][j],digits)}\n")
    with open(f"{basename}_b.txt", "w") as fb:
        for j in range(s):
            fb.write(f"{j+1} {nstr_fixed(b[j],digits)}\n")
    with open(f"{basename}_c.txt", "w") as fc:
        for i in range(s):
            fc.write(f"{i+1} {nstr_fixed(c[i],digits)}\n")
    with open(f"{basename}_triplets.txt", "w") as ft:
        for i in range(s):
            for j in range(s):
                ft.write(f"{i+1} {j+1} {nstr_fixed(A[i][j],digits)}\n")
        for i in range(s):
            ft.write(f"{i+1} 0 {nstr_fixed(c[i],digits)}\n")
        for j in range(s):
            ft.write(f"0 {j+1} {nstr_fixed(b[j],digits)}\n")

#main execution block
if __name__ == "__main__":
    mp.mp.dps = 100
    for s in (7, 8):
        A, b, c = build_lobatto_IIIC_irk(s)
        err_rows = mp.sqrt(mp.fsum([(mp.fsum(A[i]) - c[i])**2 for i in range(s)]))
        err_mom = max(abs(mp.fsum([b[j]*c[j]**k for j in range(s)]) - mp.mpf(1)/(k+1)) for k in range(2*s-2))
        print(f"||A·1−c||₂ = {mp.nstr(err_rows,40)}")
        print(f"max moment error = {mp.nstr(err_mom,40)}")
        print(f"sum(b) = {mp.nstr(mp.fsum(b),40)}")
        print(f"c[0]=0? {mp.almosteq(c[0],mp.mpf('0'))}, c[-1]=1? {mp.almosteq(c[-1],mp.mpf('1'))}\n")
        base = f"lobatto_s{s}"
        write_A_b_c_triplets(A,b,c,base,digits=80)
        print(f"Written: {base}_A.txt, _b.txt, _c.txt, _triplets.txt\n")
