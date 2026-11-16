# -*- coding: utf-8 -*-
"""Лабораторна робота №8. Чисельні методи розв'язування нелінійних рівнянь.
Частина 1: F(x) = x*sin(x) - 1 = 0.  Частина 2: алгебраїчне рівняння (Горнер, Лін)."""
import math, cmath, json

EPS, A, B, H, MAXIT = 1e-10, 0.0, 6.0, 0.1, 10000

F   = lambda x: x * math.sin(x) - 1.0
dF  = lambda x: math.sin(x) + x * math.cos(x)
d2F = lambda x: 2.0 * math.cos(x) - x * math.sin(x)


def tabulate(f, a, b, h, filename):
    """Табуляція f на [a, b] з кроком h; результат записується у файл."""
    xs = [a + i * h for i in range(int(round((b - a) / h)) + 1)]
    ys = [f(x) for x in xs]
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("   i      x_i          F(x_i)\n")
        for i, (x, y) in enumerate(zip(xs, ys)):
            fh.write("{:>4d} {:>8.2f} {:>16.8f}\n".format(i, x, y))
    return xs, ys


def find_crossings(xs, ys):
    """Наближені абсциси перетину з віссю Ox (зміна знаку + лінійна інтерполяція)."""
    return [{"x": xs[i] - ys[i] * (xs[i+1] - xs[i]) / (ys[i+1] - ys[i]),
             "kind": "зростає" if ys[i+1] > ys[i] else "спадає", "a": xs[i], "b": xs[i+1]}
            for i in range(len(xs) - 1) if ys[i] * ys[i+1] < 0]


def solve(f, step, start, eps=EPS):
    """Загальний цикл ітераційного методу. step(p) за останніми наближеннями p
    повертає наступне. Критерій зупинки: |F(x_{n+1})| < eps та |x_{n+1}-x_n| < eps.
    Повертає (корінь, кількість ітерацій, історія)."""
    p, hist = list(start), list(start)
    for n in range(1, MAXIT + 1):
        try:
            x = step(p)
        except ZeroDivisionError:
            return p[-1], n - 1, hist
        hist.append(x)
        if abs(f(x)) < eps and abs(x - p[-1]) < eps:
            return x, n, hist
        p = p[1:] + [x]
    return p[-1], MAXIT, hist


# --- кроки методів (p[-1] = x_n, p[-2] = x_{n-1}, ...) ---
def relax(tau):                       # проста ітерація: x + tau*F(x)
    return lambda p: p[-1] + tau * F(p[-1])

def newton(p):                        # Ньютон
    return p[-1] - F(p[-1]) / dF(p[-1])

def chebyshev(p):                     # Чебишев
    x = p[-1]
    return x - F(x) / dF(x) - 0.5 * F(x) ** 2 * d2F(x) / dF(x) ** 3

def secant(p):                        # хорд
    return p[-1] - F(p[-1]) * (p[-1] - p[-2]) / (F(p[-1]) - F(p[-2]))

def parabolas(p):                     # парабол (Мюллера)
    xpp, xp, x = p[-3:]
    f1 = (F(x) - F(xp)) / (x - xp)
    f2 = (f1 - (F(xp) - F(xpp)) / (xp - xpp)) / (x - xpp)
    a, b, c = f2, (x - xp) * f2 + f1, F(x)
    d = cmath.sqrt(b * b - 4 * a * c)
    den = b + d if abs(b + d) > abs(b - d) else b - d     # корінь, найменший за модулем
    return x + (-2 * c / den).real

def inverse(p):                       # зворотна інтерполяція (Лагранж, 3 вузли)
    (xa, xb, xc), (ya, yb, yc) = p[-3:], [F(v) for v in p[-3:]]
    return (yb * yc / ((ya - yb) * (ya - yc)) * xa + ya * yc / ((yb - ya) * (yb - yc)) * xb
            + ya * yb / ((yc - ya) * (yc - yb)) * xc)

def aitken(g):                        # Ейткен для x = g(x)
    def step(p):
        x0 = p[-1]; x1 = g(x0); x2 = g(x1)
        return x2 + (x2 - x1) ** 2 / (2 * x1 - x2 - x0)
    return step


# --- алгебраїчні рівняння ---
def read_coeffs(filename):
    """Зчитує a_0, a_1, ..., a_m з файлу (рядки з # - коментарі)."""
    with open(filename, encoding="utf-8") as fh:
        return [float(t) for line in fh if line.strip() and not line.startswith("#")
                for t in line.split()]

def poly_value(a, x):
    """Значення многочлена в точці x (схема Горнера)."""
    s = 0.0
    for coef in reversed(a):
        s = s * x + coef
    return s

def horner_b0_c1(a, x):
    """Схема Горнера: b0 = F(x), c1 = F'(x)."""
    b = c = a[-1]
    for i in range(len(a) - 2, 0, -1):
        b = a[i] + x * b
        c = b + x * c
    return a[0] + x * b, c

def newton_horner(a, x0, eps=EPS):
    """Метод Ньютона зі схемою Горнера: x_{n+1} = x_n - b0/c1."""
    x, hist = x0, [x0]
    for n in range(1, MAXIT + 1):
        b0, c1 = horner_b0_c1(a, x)
        xn = x - b0 / c1
        hist.append(xn)
        if abs(poly_value(a, xn)) < eps and abs(xn - x) < eps:
            return xn, n, hist
        x = xn
    return x, MAXIT, hist

def lin(a, alpha, beta, eps=EPS):
    """Метод Ліна: комплексні корені alpha ± i*beta."""
    m, hist = len(a) - 1, [(alpha, beta)]
    for n in range(1, MAXIT + 1):
        p0, q0 = -2 * alpha, alpha ** 2 + beta ** 2
        b = [0.0] * (m + 3)
        b[m], b[m-1] = a[m], a[m-1] - p0 * a[m]
        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p0 * b[i+1] - q0 * b[i+2]
        q1 = a[0] / b[2]
        p1 = (a[1] * b[2] - a[0] * b[3]) / b[2] ** 2
        a1, b1 = -p1 / 2, math.sqrt(q1 - p1 ** 2 / 4)
        hist.append((a1, b1))
        if abs(a1 - alpha) <= eps and abs(b1 - beta) <= eps:
            return a1, b1, n, hist
        alpha, beta = a1, b1
    return alpha, beta, MAXIT, hist


# --- графіки ---
def make_plots(xs, ys, roots, histories, a, xr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(name, f, lo, hi, pts, label, notes, loc):
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        t = [lo + i * (hi - lo) / 600 for i in range(601)]
        ax.plot(t, [f(v) for v in t], lw=1.8, label=label)
        if pts: ax.plot(*pts, ".", color="r", ms=3.5, label="вузли табуляції (h = 0,1)")
        ax.axhline(0, color="k", lw=0.8)
        for r in notes:
            ax.plot([r], [0], "go", ms=7, zorder=5)
            ax.annotate("x ≈ {:.6f}".format(r), (r, 0), textcoords="offset points", xytext=(6, 10))
        ax.set(xlabel="x", ylabel="F(x)"); ax.grid(alpha=0.3); ax.legend(loc=loc)
        fig.tight_layout(); fig.savefig(name, dpi=200); plt.close(fig)

    curve("fig_tab.png", F, A, B, (xs, ys), "F(x) = x·sin x − 1", roots, "lower left")
    curve("fig_poly.png", lambda x: poly_value(a, x), -2, 3, None, "F(x) = x³ − 2x² + 3x − 5", [xr], "upper left")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, (title, r, hs) in zip(axes, histories):
        for name, h in hs.items():
            ax.semilogy([max(abs(v - r), 1e-17) for v in h], marker="o", ms=3, lw=1.2, label=name)
        ax.set_title(title, fontsize=10); ax.set_xlabel("номер ітерації n"); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("|x_n − x*|"); axes[1].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig("fig_conv.png", dpi=200); plt.close(fig)


def main():
    out = {}
    # 1. табуляція та відокремлення коренів
    xs, ys = tabulate(F, A, B, H, "tabulation.txt")
    cr = find_crossings(xs, ys)
    out["crossings"] = cr
    for c in cr:
        print("x ≈ {:.5f}  (F {})".format(c["x"], c["kind"]))
    inc = next(c for c in cr if c["kind"] == "зростає")["x"]
    dec = next(c for c in cr if c["kind"] == "спадає")["x"]

    # 2-4. розв'язання F(x) = 0 різними методами
    out["methods"], histories, roots = [], [], []
    for title, x0, tau in (("зростаюча функція", inc, -0.5), ("спадна функція", dec, 0.4)):
        g = relax(tau)
        runs = {
            "Проста ітерація (релаксація)": solve(F, g, [x0]),
            "Метод Ньютона": solve(F, newton, [x0]),
            "Метод Чебишева": solve(F, chebyshev, [x0]),
            "Метод хорд": solve(F, secant, [x0, x0 + H]),
            "Метод парабол": solve(F, parabolas, [x0 - H, x0, x0 + H]),
            "Зворотна інтерполяція": solve(F, inverse, [x0 - H, x0, x0 + H]),
            "Метод Ейткена": solve(F, aitken(lambda x: g([x])), [x0]),
        }
        print("\nx0 = {:.5f} ({})".format(x0, title))
        for name, (r, it, _) in runs.items():
            print("{:<30} x* = {:.12f}  F = {:+.1e}  ітерацій: {}".format(name, r, F(r), it))
        ref = runs["Метод Ньютона"][0]
        roots.append(ref)
        out["methods"].append({
            "title": title, "x0": x0, "tau": tau, "ref": ref,
            "rows": [{"name": n, "root": r, "F": F(r), "iters": it} for n, (r, it, _) in runs.items()],
            "newton_hist": [(v, F(v)) for v in runs["Метод Ньютона"][2]]})
        histories.append((title + ", x* ≈ {:.5f}".format(ref), ref, {n: h for n, (_, _, h) in runs.items()}))

    # 5-9. алгебраїчне рівняння
    a = read_coeffs("coeffs.txt")
    R = 1 + max(abs(c) for c in a[:-1]) / abs(a[-1])         # радіус кола коренів
    xr, it_n, hist_n = newton_horner(a, R)
    al, be, it_l, hist_l = lin(a, 0.0, 1.5)
    z = poly_value(a, complex(al, be))
    print("\na =", a, " R =", R)
    print("Ньютон+Горнер: x = {:.12f}, ітерацій: {}".format(xr, it_n))
    print("Лін: x = {:.12f} ± {:.12f}i, ітерацій: {}".format(al, be, it_l))
    out["alg"] = {"coeffs": a, "R": R, "x_real": xr, "it_newton": it_n, "alpha": al, "beta": be,
                  "it_lin": it_l, "hist_newton": [(v, poly_value(a, v)) for v in hist_n],
                  "hist_lin": hist_l, "Pz": [z.real, z.imag], "P_real": poly_value(a, xr)}
    try:
        import numpy as np
        out["alg"]["numpy_roots"] = [[r.real, r.imag] for r in np.roots(a[::-1])]
    except ImportError:
        out["alg"]["numpy_roots"] = []
    make_plots(xs, ys, roots, histories, a, xr)
    json.dump(out, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()