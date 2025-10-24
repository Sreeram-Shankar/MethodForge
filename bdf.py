import mpmath as mp

#compute bdf coefficients
def bdf_coeffs(k, prec=300):
    mp.mp.dps = prec
    h = mp.mpf('1')
    A = mp.matrix([[(-j*h)**m for j in range(k+1)] for m in range(k+1)])
    b = mp.matrix([mp.mpf('0') if m == 0 else h*m*(0**(m-1)) for m in range(k+1)])
    alpha = mp.lu_solve(A, b)
    alpha = [a/alpha[0] for a in alpha]
    return alpha

#verify bdf order accuracy
def check_order_bdf(alpha):
    k = len(alpha) - 1
    errs = []
    h = mp.mpf('1')
    for p in range(k+1):
        lhs = mp.fsum(alpha[j]*(-j*h)**p for j in range(k+1))
        rhs = h * p * (0**(p-1)) if p > 0 else 0
        errs.append(abs(lhs - rhs))
    return max(errs)

#adaptive precision bdf computation
def adaptive_bdf(k, max_prec=2000, target_error=1e-80):
    prec = 200
    best_coeffs = None
    best_err = mp.inf
    while prec <= max_prec:
        mp.mp.dps = prec
        coeffs = bdf_coeffs(k, prec)
        err = check_order_bdf(coeffs)
        if err < best_err:
            best_err, best_coeffs = err, coeffs
        if err < target_error:
            return best_coeffs, prec, err
        prec = int(prec * 1.5)
    return best_coeffs, prec, best_err

#save bdf coefficients to file
def save_bdf_to_file(coeffs, k, prec, filename=None):
    if filename is None:
        filename = f"bdf{k}_high_precision.txt"
    with open(filename, "w") as f:
        f.write(f"# BDF coefficients for order k={k}\n")
        f.write(f"# Precision: {prec} digits\n\n")
        for i, c in enumerate(coeffs):
            f.write(f"{i} {mp.nstr(c, 80)}\n")

#main execution block
if __name__ == "__main__":
    test_orders = [2, 20]
    for k in test_orders:
        coeffs, used_prec, err = adaptive_bdf(k)
        print(f"\nBDF order {k}: (precision {used_prec})")
        for i, c in enumerate(coeffs):
            print(f"a[{i}] =", mp.nstr(c, 50))
        print("max error =", err)
        save_bdf_to_file(coeffs, k, used_prec)
