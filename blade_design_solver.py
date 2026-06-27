"""
blade_design_solver.py
Companion code for "How Is a Wind Turbine Blade Designed? (From Scratch in Python)"

Designs an optimal wind-turbine blade from first principles and validates it
against a naive baseline, using blade-element-momentum (BEM) theory - no
commercial software.

Pipeline:
  1. Pick a design tip-speed ratio (lambda) and the airfoil's best angle of attack.
  2. Optimal TWIST  : theta(r) = phi_opt(r) - alpha_best, using the Betz
                      condition (axial induction a = 1/3).
  3. Optimal CHORD  : c(r) = (8 pi r)/(B * Cl) * (1 - cos(phi_opt(r))).
  4. Run both the optimized and a baseline blade through a BEM solver (with
     Prandtl tip loss) and compare power coefficients across wind speeds.

Verified result: optimized peak Cp ~ 0.476 vs baseline ~ 0.414 (about +15%),
both safely below the Betz limit 16/27 = 0.593.

Run:  python blade_design_solver.py
"""

import numpy as np

# ---------------- turbine / airfoil specification ----------------
R = 5.0            # rotor radius (m)
B = 3              # number of blades
rho = 1.225        # air density (kg/m^3)
V_design = 8.0     # design wind speed (m/s)
lam_design = 6.0   # design tip-speed ratio
omega = lam_design * V_design / R   # rotor angular speed (rad/s)

rs = np.linspace(0.4, 4.95, 24)     # blade element radii
dr = rs[1] - rs[0]


# ---------------- airfoil model ----------------
def airfoil(alpha):
    """Lift and drag coefficients vs angle of attack (rad). Stylised but
    representative: linear-ish Cl with soft stall, a drag bucket near 6 deg."""
    a_deg = np.rad2deg(alpha)
    Cl = 2 * np.pi * np.sin(alpha) / (1 + (a_deg / 16) ** 6) ** (1 / 6)
    Cd = 0.008 + 0.004 * ((a_deg - 6) / 6) ** 2
    return Cl, Cd


def best_alpha():
    """Angle of attack giving the maximum lift-to-drag ratio."""
    alphas = np.deg2rad(np.linspace(1, 14, 600))
    LD = [airfoil(a)[0] / airfoil(a)[1] for a in alphas]
    i = int(np.argmax(LD))
    return alphas[i], airfoil(alphas[i])[0]


# ---------------- optimal blade design ----------------
def design_optimal_blade():
    alpha_best, Cl_design = best_alpha()
    a_betz = 1.0 / 3.0
    lam_r = omega * rs / V_design
    phi_opt = np.arctan((1 - a_betz) / lam_r)            # ideal inflow angle
    chord = (8 * np.pi * rs) / (B * Cl_design) * (1 - np.cos(phi_opt))
    twist = phi_opt - alpha_best
    return chord, twist, alpha_best


def baseline_blade():
    """A reasonable but un-optimized blade: linear taper + simple twist washout."""
    chord = 0.55 - 0.05 * rs
    twist = np.deg2rad(np.maximum(1.0, 20.0 - 4.2 * rs))
    return chord, twist


# ---------------- BEM solver (with Prandtl tip loss) ----------------
def solve_element(r, c, tw, V):
    sigma = B * c / (2 * np.pi * r)
    a, ap = 0.3, 0.0
    for _ in range(400):
        phi = np.arctan2((1 - a) * V, (1 + ap) * omega * r)
        alpha = phi - tw
        Cl, Cd = airfoil(alpha)
        Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ct = Cl * np.sin(phi) - Cd * np.cos(phi)
        if Cn <= 0:
            Cn = 1e-3
        f = B / 2 * (R - r) / (r * np.sin(abs(phi)) + 1e-9)
        F = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f, 0, 30)), -1, 1))
        F = max(F, 1e-3)
        a_new = 1 / (4 * F * np.sin(phi) ** 2 / (sigma * Cn) + 1)
        denom = 4 * F * np.sin(phi) * np.cos(phi) / (sigma * Ct) - 1
        ap_new = 1 / denom if abs(denom) > 1e-6 else ap
        a = 0.4 * a_new + 0.6 * a      # under-relax for stability
        ap = 0.4 * ap_new + 0.6 * ap
        a = np.clip(a, -0.5, 0.9)
    return a, ap, phi, Cl, Cd


def power_curve(chord, twist, winds):
    cps = []
    for V in winds:
        dQ = 0.0
        for i, r in enumerate(rs):
            a, ap, phi, Cl, Cd = solve_element(r, chord[i], twist[i], V)
            W2 = ((1 - a) * V) ** 2 + ((1 + ap) * omega * r) ** 2
            Ct = Cl * np.sin(phi) - Cd * np.cos(phi)
            dQ += B * 0.5 * rho * W2 * chord[i] * Ct * r * dr
        cps.append(omega * dQ / (0.5 * rho * np.pi * R ** 2 * V ** 3))
    return np.array(cps)


# ---------------- run it ----------------
if __name__ == "__main__":
    c_opt, tw_opt, alpha_best = design_optimal_blade()
    c_base, tw_base = baseline_blade()

    print(f"Design tip-speed ratio   : {lam_design}")
    print(f"Best angle of attack     : {np.rad2deg(alpha_best):.1f} deg")
    print(f"Total twist (root->tip)  : {np.rad2deg(tw_opt[0] - tw_opt[-1]):.0f} deg")
    print(f"Chord root -> tip        : {c_opt[0]:.2f} -> {c_opt[-1]:.2f} m")

    winds = np.arange(4, 13, 1.0)
    cp_base = power_curve(c_base, tw_base, winds)
    cp_opt = power_curve(c_opt, tw_opt, winds)

    print("\nWind (m/s):", list(winds.astype(int)))
    print("baseline Cp:", [round(x, 3) for x in cp_base])
    print("designed Cp:", [round(x, 3) for x in cp_opt])
    print(f"\nbaseline peak Cp  = {max(cp_base):.3f}")
    print(f"designed peak Cp  = {max(cp_opt):.3f}")
    print(f"improvement       = {100*(max(cp_opt)/max(cp_base)-1):.0f}%")
    print(f"Betz limit (16/27)= {16/27:.3f}  ->  both below it: {max(cp_opt) < 16/27}")
