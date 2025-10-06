import mpmath as mp
import numpy as np

def adams_bashforth_coeffs(k, prec=500):
    #compute adams-bashforth coefficients for order k
    if k >= 12:
        prec = max(prec, 1000)
    
    mp.mp.dps = prec
    #lagrange interpolation points for ab
    xs = [-j for j in range(k)]
    coeffs = []
    
    for j in range(k):
        def ell(x):
            #lagrange basis polynomial for point j
            num, den = mp.mpf('1'), mp.mpf('1')
            for m in range(k):
                if m != j:
                    diff_num = x - xs[m]
                    diff_den = xs[j] - xs[m]
                    num *= diff_num
                    den *= diff_den
            return num/den
        
        #integrate basis polynomial over [0,1]
        coeffs.append(mp.quad(ell, [0, 1], method='gauss-legendre', maxdegree=100))
    
    return coeffs

def adams_moulton_coeffs(k, prec=500):
    #compute adams-moulton coefficients for order k
    if k >= 12:
        prec = max(prec, 1000)
    
    mp.mp.dps = prec
    #lagrange interpolation points for am (includes current point)
    xs = [1] + [-j for j in range(k-1)]
    coeffs = []
    
    for j in range(k):
        def ell(x):
            #lagrange basis polynomial for point j
            num, den = mp.mpf('1'), mp.mpf('1')
            for m in range(k):
                if m != j:
                    diff_num = x - xs[m]
                    diff_den = xs[j] - xs[m]
                    num *= diff_num
                    den *= diff_den
            return num/den
        
        #integrate basis polynomial over [0,1]
        coeffs.append(mp.quad(ell, [0, 1], method='gauss-legendre', maxdegree=100))
    
    return coeffs

def check_order(coeffs, kind):
    #verify order conditions for multistep coefficients
    k = len(coeffs)
    xs = [-j for j in range(k)] if kind=="AB" else [1]+[-j for j in range(k-1)]
    errs = []
    
    #check order conditions for powers 0 to k-1
    for p in range(k):
        lhs = mp.fsum(coeffs[j]*xs[j]**p for j in range(k))
        rhs = 1/(p+1)
        error = abs(lhs-rhs)
        errs.append(error)
    
    max_error = max(errs)
    return max_error

def adaptive_precision_coeffs(k, kind="AB", max_prec=2000, target_error=1e-100):
    #adaptive precision to achieve target accuracy
    prec = 200
    best_coeffs = None
    best_error = float('inf')
    
    while prec <= max_prec:
        try:
            #compute coefficients with current precision
            if kind == "AB":
                coeffs = adams_bashforth_coeffs(k, prec)
            else:
                coeffs = adams_moulton_coeffs(k, prec)
            
            #check accuracy of computed coefficients
            current_error = check_order(coeffs, kind)
            
            if current_error < best_error:
                best_coeffs = coeffs
                best_error = current_error
            
            if current_error < target_error:
                return coeffs, prec, current_error
                
            #increase precision for next iteration
            prec = int(prec * 1.5)
            
        except Exception as e:
            break
    
    return best_coeffs, prec//2, best_error

def save_coefficients_to_file(coeffs, filename, kind, k, precision_used):
    #save computed coefficients to text file
    with open(filename, 'w') as f:
        f.write(f"# {kind} coefficients for order k={k}\n")
        f.write(f"# Precision used: {precision_used} decimal places\n")
        f.write(f"# Generated with enhanced precision multistep calculator\n")
        f.write(f"# Format: coefficient_index value\n\n")
        
        for i, c in enumerate(coeffs):
            f.write(f"{i} {mp.nstr(c, 100)}\n")

if __name__ == "__main__":
    #main execution for multistep coefficient computation
    test_orders = [12, 14]
    
    for k in test_orders:
        #compute coefficients with adaptive precision for high orders
        if k >= 12:
            ab_coeffs, ab_prec, ab_error = adaptive_precision_coeffs(k, "AB")
            am_coeffs, am_prec, am_error = adaptive_precision_coeffs(k, "AM")
        else:
            ab_coeffs = adams_bashforth_coeffs(k, prec=500)
            am_coeffs = adams_moulton_coeffs(k, prec=500)
            ab_error = check_order(ab_coeffs, "AB")
            am_error = check_order(am_coeffs, "AM")
            ab_prec = 500
            am_prec = 500
        
        #display adams-bashforth results
        print(f"Adams–Bashforth order {k} coefficients:")
        for c in ab_coeffs:
            print(mp.nstr(c, 50))
        print("max error =", mp.nstr(ab_error, 10))
        print("epsilon ratio =", mp.nstr(ab_error/mp.eps, 10), "\n")
        
        #display adams-moulton results
        print(f"Adams–Moulton order {k} coefficients:")
        for c in am_coeffs:
            print(mp.nstr(c, 50))
        print("max error =", mp.nstr(am_error, 10))
        print("epsilon ratio =", mp.nstr(am_error/mp.eps, 10), "\n")
        
        #save coefficients to files
        save_coefficients_to_file(ab_coeffs, f"ab{k}_high_precision.txt", "AB", k, ab_prec)
        save_coefficients_to_file(am_coeffs, f"am{k}_high_precision.txt", "AM", k, am_prec)
