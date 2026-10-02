#!/usr/bin/env python3
"""Independent exact classification of five identical planar-vortex equilibria.

Normalization: sum_{j != i} 1/(z_i-z_j) = 2*conjugate(z_i).
The proof has two parts:
  (1) every physical collision-free equilibrium has reflection symmetry;
  (2) the reflection-symmetric equations have exactly five physical shapes,
      with the discriminants computed here.

This is NOT a transcription of Faugere--Svartz and does NOT use their
classification or their intermediate 17-solution count. All elimination
systems are generated from resultants or from the original remainder
identities. All asserted algebra, root counts, and inequalities are exact.
Floating-point output is illustrative only.

Requires Python 3 and SymPy. Run:
    python n5_complete_verification.py
Do not use Python's -O option, which would disable assertions.
"""
from __future__ import annotations

import time
import sympy as sp

if not __debug__:
    raise RuntimeError("Assertions must be enabled; do not run with python -O.")

X, Y, T = sp.symbols("X Y T")
a, b, c, d = sp.symbols("a b c d")
B, k, s = sp.symbols("B k s")


def resultant_data(p):
    """Return delta, R(Y,0), dR/dT(Y,0), and the three moment equations."""
    delta = sp.discriminant(p, X)
    R = sp.resultant(p, (Y + T*X)*sp.diff(p, X) - sp.diff(p, X, 2), X)
    R0 = sp.expand(R.subs(T, 0))
    Rt = sp.expand(sp.diff(R, T).subs(T, 0))
    assert sp.expand(R0.coeff(Y, 5) - delta) == 0
    assert R0.coeff(Y, 4) == 0
    assert Rt.coeff(Y, 4) == 0
    assert sp.expand(Rt.coeff(Y, 3) - 20*delta) == 0
    E = [Rt.coeff(Y, 2), Rt.coeff(Y, 1) - 6*R0.coeff(Y, 3),
         Rt.coeff(Y, 0) - 2*R0.coeff(Y, 2)]
    return delta, R0, Rt, E


def even_to_B(poly):
    """Convert an even polynomial in b into a polynomial in B=b^2."""
    P = sp.Poly(sp.expand(poly), b)
    assert all(m[0] % 2 == 0 for m, coeff in P.terms())
    return sp.expand(sum(coeff*B**(m[0]//2) for m, coeff in P.terms()))


def monic(poly, var):
    return sp.Poly(poly, var, domain=sp.QQ).monic().as_expr()


def verify_generic_symmetry():
    print("PART I. UNRESTRICTED COEFFICIENT ELIMINATION")
    print("Chart a != 0, b != 0: normalize a=1; put B=b^2, k=d/b.")
    p = X**5 + X**3 + b*X**2 + c*X + d
    delta, R0, Rt, E = resultant_data(p)
    F = []
    for j, f in enumerate(E):
        f = sp.Poly(sp.expand(f.subs(d, b*k)), b)
        if j in (0, 2):
            assert f.coeff_monomial(1) == 0
            f = sp.Poly(sp.expand(f.as_expr()/b), b)
        f = even_to_B(f.as_expr())
        F.append(sp.Poly(f, B, c, k, domain=sp.QQ).primitive()[1].as_expr())
    Delta = even_to_B(delta.subs(d, b*k))
    print("Three primitive input polynomials:")
    for j, f in enumerate(F, 1):
        print(f" F{j} = {f}")

    # Rabinowitsch saturation: the s-free part equals
    # <F1,F2,F3> : (B*Delta)^infinity.
    G = sp.groebner(F + [s*B*Delta - 1], s, k, c, B,
                    order="grevlex", domain=sp.QQ)
    assert G.is_zero_dimensional
    L = G.fglm("lex")
    assert len(L.polys) == 4
    assert all(L.reduce(f)[1] == 0 for f in F + [s*B*Delta - 1])

    P1 = 125*B**4 + 7900*B**3 - 37040*B**2 + 44608*B + 512
    P2 = 2500*B**4 - 114000*B**3 + 338925*B**2 - 11000*B + 6912
    C1 = 3*(125*B**3 + 8020*B**2 - 16496*B + 4224)/sp.Integer(57088)
    K1 = (75*B**3 + 4812*B**2 - 16320*B + 13952)/sp.Integer(28544)
    C2 = -(1250*B**3 - 69425*B**2 + 327305*B - 145296)/sp.Integer(443535)
    K2 = -(28500*B**3 - 1287200*B**2 + 3884705*B - 1798816)/sp.Integer(5913800)
    P = monic(P1*P2, B)
    assert sp.expand(L.polys[3].as_expr() - P) == 0
    cB = sp.expand(c - L.polys[2].as_expr())
    kB = sp.expand(k - L.polys[1].as_expr())
    assert cB.free_symbols <= {B} and kB.free_symbols <= {B}
    assert sp.gcd(P1, P2) == 1
    assert sp.gcd(P, sp.diff(P, B)) == 1
    for Pi, Ci, Ki in [(P1, C1, K1), (P2, C2, K2)]:
        assert sp.rem(cB - Ci, Pi, B) == 0
        assert sp.rem(kB - Ki, Pi, B) == 0
    print("\nExact saturated projection:")
    print(" I_sat = <P1, c-C1, k-K1> intersection <P2, c-C2, k-K2>.")
    for j, Pi, Ci, Ki in [(1,P1,C1,K1), (2,P2,C2,K2)]:
        print(f" P{j} = {Pi}")
        print(f" C{j} = {sp.factor(Ci)}")
        print(f" K{j} = {sp.factor(Ki)}")
    print("The degree-eight eliminant is square-free; the factors are coprime.")

    # Characteristic polynomial of the dual coordinates y_i:
    # R0(4Y)/(4^5*delta). Its coefficients are
    # a_y=r3/(16*delta), b_y=r2/(64*delta), etc.
    r3, r2, r1, r0 = (R0.coeff(Y, j) for j in (3, 2, 1, 0))
    dual_numerators = [delta*r2**2 - b**2*r3**3,
                       delta*r1 - c*r3**2,
                       delta*r0 - k*r3*r2]
    for j, f in enumerate(dual_numerators):
        f = sp.Poly(sp.expand(f.subs(d, b*k)), b)
        if j == 2:
            assert f.coeff_monomial(1) == 0
            f = sp.Poly(sp.expand(f.as_expr()/b), b)
        fB = even_to_B(f.as_expr())
        assert L.reduce(fB)[1] == 0
    print("\nExact dual-invariant identities, all with zero normal form:")
    print(" delta*r2^2 = b^2*r3^3")
    print(" delta*r1 = c*r3^2")
    print(" delta*r0 = k*r3*r2")
    print("For a physical solution y_i=kappa*conjugate(x_i), kappa>0,")
    print("the first identity gives B=conjugate(B); Cj(B), Kj(B) are real.")
    print("Thus p has real coefficients, or its roots become so after rotation by i.")
    print("This proves reflection symmetry in the generic physical chart.")


def verify_exceptional_charts():
    print("\nPART II. ALL EXCEPTIONAL COEFFICIENT CHARTS")
    cases = [
        ("a=1, b=0", {a:1, b:0}, [2]),
        ("a=0, b=1", {a:0, b:1}, [3]),
        ("a=0, b=0", {a:0, b:0}, [3,2]),
    ]
    for name, subs, zero_coeffs in cases:
        p = (X**5 + a*X**3 + b*X**2 + c*X + d).subs(subs)
        delta, R0, Rt, E = resultant_data(p)
        # A zero coefficient in p gives the corresponding zero coefficient
        # in the physical dual polynomial, since y=kappa*conjugate(x).
        E += [R0.coeff(Y, j) for j in zero_coeffs]
        E = [sp.Poly(f,c,d,domain=sp.QQ).primitive()[1].as_expr()
             for f in E if f != 0]
        L = sp.groebner(E + [s*delta - 1], s, d, c,
                       order="lex", domain=sp.QQ)
        print(f"\n{name}: saturated lexicographic basis")
        for f in L.polys:
            print(" ", sp.factor(f.as_expr()))
        if name == "a=1, b=0":
            expected = sp.groebner([s-sp.Rational(3125,27),d,c-sp.Rational(3,20)],
                                  s,d,c,order="lex",domain=sp.QQ)
            assert L == expected
            print("Only p=X^5+X^3+(3/20)X: all roots lie on the imaginary axis.")
        elif name == "a=0, b=1":
            assert len(L.polys) == 1 and L.polys[0].as_expr() == 1
            print("No physical solution in this chart.")
        else:
            free = [f.as_expr() for f in L.polys if s not in f.as_expr().free_symbols]
            assert len(free) == 1 and sp.expand(free[0] - c*d) == 0
            print("cd=0, delta!=0: regular pentagon or square with its center.")
    print("\nEVERY PHYSICAL COLLISION-FREE EQUILIBRIUM IS REFLECTION-SYMMETRIC.")


def verify_reflection_classification():
    print("\nPART III. REFLECTION-SYMMETRIC CLASSIFICATION AND DISCRIMINANTS")
    import sympy as sp
    
    z, h, d, e, H, J = sp.symbols('z h d e H J')
    Q = 10*H**4 - 65*H**3 + 140*H**2 - 97*H + 6
    JH = (-350*H**3 + 1775*H**2 - 2430*H + 975)/sp.Integer(92)
    DH = -sp.Rational(84375, 385875968) * (
        7632290*H**3 - 30197665*H**2 + 32980990*H - 12854393)
    
    def modq(f):
        """Reduce a rational function in H in QQ[H]/(Q)."""
        num, den = sp.fraction(sp.cancel(f))
        return sp.rem(num * sp.invert(den, Q, H), Q, H).expand()
    
    def iadd(x, y):
        return (x[0] + y[0], x[1] + y[1])
    
    def imul(x, y):
        vals = [a*b for a in x for b in y]
        return (min(vals), max(vals))
    
    def interval_poly(poly, var, interval):
        """Horner enclosure with rational endpoints (no floating point)."""
        out = (sp.S.Zero, sp.S.Zero)
        for c in sp.Poly(poly, var).all_coeffs():
            out = iadd(imul(out, interval), (c, c))
        return out
    
    # One real point and two conjugate pairs.
    u1, u2 = (h+d)/2, (h-d)/2
    v1, v2 = (5-h*h+2*e)/4, (5-h*h-2*e)/4
    q1, q2 = z*z + u1*z + v1, z*z + u2*z + v2
    p = sp.expand((z-h)*q1*q2)
    assert sp.Poly(p, z).coeff_monomial(z**4) == 0
    assert sp.expand(h*h + 2*v1 + 2*v2) == 5
    p1, p2 = sp.diff(p, z), sp.diff(p, z, 2)
    f0 = sp.factor((p2-4*h*p1).subs(z, h))
    f1 = sp.Poly(sp.rem(p2+4*(u1+z)*p1, q1, z), z).all_coeffs()
    f2 = sp.Poly(sp.rem(p2+4*(u2+z)*p1, q2, z), z).all_coeffs()
    equations = [f0] + f1 + f2
    G = sp.groebner(equations, e, d, h, order='lex', domain=sp.QQ)
    assert len(G.polys) == 6
    elim = sp.factor(G.polys[-1].as_expr())
    assert sp.expand(elim - h*(h*h-1)*Q.subs(H,h*h)/10) == 0
    print('ONE REAL POINT + TWO CONJUGATE PAIRS')
    print('Exact lexicographic Groebner basis:')
    for g in G.polys:
        print(' ', sp.factor(g.as_expr()))
    
    # On the nonclassical branch, verify the rational parametrization directly
    # in the ORIGINAL five remainder equations.
    Gbranch = sp.groebner([d*d-JH, h*h-H, Q], d, h, H, order='lex', domain=sp.QQ)
    for f in equations:
        f = sp.together(f.subs(e, 5*h*(H-1)/(2*d)))
        num, den = sp.fraction(f)
        assert Gbranch.reduce(sp.expand(num))[1] == 0
    print('\nVerified all original stationary remainder equations on Q(H)=0.')
    
    # Three scalar relations in H,J from the same remainder equations.
    g0 = (H-1)*(4*J**2-5*(H+7)*J+25*H*(H-1))
    g1 = (H+2)*J**2 + (-15*H**2+20*H-20)*J + 50*H*(H-1)**2
    g2 = 2*(H-4)*J**2 + (25*H**2-70*H+75)*J - 75*H*(H-1)**2
    assert sp.factor(sp.resultant(g1,g2,J)) == 3750*H*(H-1)**3*Q
    for g in (g0,g1,g2):
        assert modq(g.subs(J,JH)) == 0
    Gq = sp.groebner([Q,g0,g1,g2], J,H, order='lex', domain=sp.QQ)
    assert Gq.reduce(J-JH)[1] == 0
    print('Nonclassical branch: Q(H)=',Q)
    print('J(H)=',JH)
    
    # Exceptional branches h=0 and h^2=1: the Groebner basis specializes to
    # h=0: d^2+4e^2=10, de=0, d(d^2-10)=0;
    # h^2=1: d^2=5, e=0.
    for vals in ({h:0,d:0,e:sp.sqrt(sp.Rational(5,2))},
                 {h:0,d:sp.sqrt(10),e:0},
                 {h:1,d:sp.sqrt(5),e:0}):
        assert all(sp.simplify(f.subs(vals)) == 0 for f in equations)
    
    # Discriminant of (z-h)q1q2, by the product/discriminant identity.
    Dpair = ((5*H+J-20)**2 - 4*H*J + 80*H*(H-1)
             - 400*H*(H-1)**2/J)/16
    AtH = (25*(H+1)**2 - 4*H*J - 20*H*(H-1)
           - 25*H*(H-1)**2/J)/16
    Rpair = (25*H*(H-1)**2/J - 5*H*(H-1) + J*(5-H))/4
    Aq, Bq, Cq = [modq(f.subs(J,JH)) for f in (Dpair,AtH,Rpair)]
    assert modq(Aq*Bq**2*Cq**2-DH) == 0
    print('\nDiscriminant factors modulo Q:')
    print('disc(q1)disc(q2)=',sp.factor(Aq))
    print('q1(h)q2(h)=',sp.factor(Bq))
    print('Res(q1,q2)=',sp.factor(Cq))
    print('Delta(H)=',DH)
    
    # Independent verification of the factor formula from the actual quadratics.
    actual_factors = [sp.discriminant(q1,z)*sp.discriminant(q2,z),
                      q1.subs(z,h)*q2.subs(z,h),sp.resultant(q1,q2,z)]
    for actual, expected in zip(actual_factors,(Aq,Bq,Cq)):
        num, den = sp.fraction(sp.together(
            actual.subs(e,5*h*(H-1)/(2*d))-expected))
        assert Gbranch.reduce(sp.expand(num))[1] == 0
    
    # Sturm counts and exact isolating intervals; positivity and noncollision.
    intervals = [(sp.Rational(68395128919,10**12),sp.Rational(68395128920,10**12)),
                 (sp.Rational(1214112777579,10**12),sp.Rational(1214112777580,10**12))]
    assert sp.Poly(Q,H).count_roots(-sp.oo,sp.oo) == 2
    for I, expected_bound in zip(intervals,[(2347,2349),(801,803)]):
        assert sp.Poly(Q,H).count_roots(*I) == 1
        assert interval_poly(JH,H,I)[0] > 0
        # Sum and product of the two quadratic discriminants imply both < 0.
        assert interval_poly((5*H+JH-20)/2,H,I)[1] < 0
        for f in (Aq,Bq,Cq):
            assert interval_poly(f,H,I)[0] > 0
        lo,hi = interval_poly(DH,H,I)
        assert expected_bound[0] < lo <= hi < expected_bound[1]
        print('H interval:',I,'; rigorous Delta interval:',(lo,hi))
    
    # Three real points plus one nonreal conjugate pair.
    u,v,c,U = sp.symbols('u v c U')
    b = (u*u+2*v-5)/2
    cubic = z**3-u*z**2+b*z+c
    quad = z*z+u*z+v
    p32 = sp.expand(cubic*quad)
    pp, ppp = sp.diff(p32,z),sp.diff(p32,z,2)
    f32 = (sp.Poly(sp.rem(ppp-4*z*pp,cubic,z),z).all_coeffs()
           + sp.Poly(sp.rem(ppp+4*(u+z)*pp,quad,z),z).all_coeffs())
    G32=sp.groebner(f32,c,v,u,order='lex', domain=sp.QQ)
    Q32=10*U**4-55*U**3+115*U**2-37*U-9
    assert sp.expand(G32.polys[-1].as_expr()-u*Q32.subs(U,u*u)/10)==0
    print('\nTHREE REAL POINTS + ONE CONJUGATE PAIR')
    for g in G32.polys:
        print(' ',sp.factor(g.as_expr()))
    V32 = -(130*U**3-625*U**2+1074*U-211)/sp.Integer(304)
    G32nz=sp.groebner(f32+[Q32.subs(U,u*u)],c,v,u,order='lex', domain=sp.QQ)
    assert G32nz.reduce(v-V32.subs(U,u*u))[1] == 0
    Ip=(sp.Rational(607600,10**6),sp.Rational(607601,10**6))
    assert sp.Poly(Q32,U).count_roots(0,sp.oo)==1
    assert sp.Poly(Q32,U).count_roots(*Ip)==1
    assert interval_poly(V32,U,Ip)[1]<0
    print('Nonzero u forces v<0, incompatible with a nonreal conjugate pair.')
    assert all(sp.simplify(f.subs({u:0,v:sp.Rational(5,4),c:0}))==0 for f in f32)
    print('u=0 yields the square with its center.')
    
    # The three classical discriminants.
    classical = {
        'pentagon': (z**5-1,sp.Integer(3125)),
        'square + center': (z**5-sp.Rational(25,16)*z,sp.Rational(5**10,2**12)),
        'collinear': (z**5-sp.Rational(5,2)*z**3+sp.Rational(15,16)*z,
                      sp.Rational(84375,1024))}
    print('\nCLASSICAL DISCRIMINANTS')
    for name,(poly,expected) in classical.items():
        assert abs(sp.discriminant(poly,z))==expected
        print(name,expected,sp.N(expected,18))
    assert sp.expand(sp.diff(classical['collinear'][0],z,2)-4*z*sp.diff(
        classical['collinear'][0],z)+20*classical['collinear'][0]) == 0
    
    # Illustrative high-precision coordinates; not used in the exact proof.
    print('\nNONCLASSICAL COORDINATES (illustrative decimals)')
    for j in (0,1):
        root=sp.CRootOf(Q,H,j)
        hv=sp.sqrt(root).evalf(40)
        dv=sp.sqrt(JH.subs(H,root)).evalf(40)
        ev=(5*hv*(root.evalf(40)-1)/(2*dv)).evalf(40)
        uv1,uv2=(hv+dv)/2,(hv-dv)/2
        vv1,vv2=(5-hv**2+2*ev)/4,(5-hv**2-2*ev)/4
        print('H =',root.evalf(25),'Delta =',DH.subs(H,root).evalf(25))
        print('real point:',hv.evalf(16))
        for uu,vv in ((uv1,vv1),(uv2,vv2)):
            print('pair:',(-uu/2).evalf(16),'+/- i *',sp.sqrt(vv-uu**2/4).evalf(16))
    print('\nALL EXACT CHECKS PASSED.')


if __name__ == "__main__":
    start = time.monotonic()
    print("SymPy", sp.__version__, "; all proof computations over exact rationals.")
    verify_generic_symmetry()
    verify_exceptional_charts()
    verify_reflection_classification()
    print("\nFULL CLASSIFICATION: ALL EXACT CHECKS PASSED.")
    print("Exactly five physical shapes, modulo rotations and permutations.")
    print("Discriminant maximum at sum |z_i|^2=5: 3125, only the regular pentagon.")
    print(f"Elapsed time: {time.monotonic()-start:.3f} seconds.")
