# simulator_with_annotations.py  --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, CheckButtons

# ---------------- pomožne funkcije ----------------------------------------
def compute_materials(E, nu):
    alpha = E / ((1+nu)*(1-2*nu))
    beta  = 2*nu / (1-nu)
    return alpha, beta

def A_solid(p0, alpha, nu, s_R):
    den = alpha*((1-nu)*(1+2*s_R) + 2*nu*(1-s_R))
    return -p0/den

def u_solid_outer(p0, alpha, nu, r_p0, R0):
    s_R = (r_p0**3)/(R0**3)
    A   = A_solid(p0, alpha, nu, s_R)
    return A*(R0 - r_p0**3/R0**2)

def u_hollow_const(r, p0, p_i0, alpha, nu, r_p0, R0):
    s_R = (r_p0**3)/(R0**3)
    beta = 2*nu/(1-nu)
    A = (-p0 + p_i0*s_R)/(alpha*(1+nu)*(1-s_R))
    return (A*r*(1 - (1+beta)/(beta-2)*(r_p0**3)/r**3)
            - r*p_i0/(alpha*(beta-2))*(r_p0**3)/r**3)

# --- Izotermnostiskanje ------------------------------------------------
def hollow_isothermal_newton(p0, p_i0, alpha, nu, r_p0, R0,
                             tol=1e-6, nmax=50000):
    """
    Vrne (A, p_in) za izotermno (PV-konstantno) stiskanje votline.
    Zagotovi, da ostane r_p > 0 ne glede na to, ali je konst.-p' model
    že napovedal zaprtje votline.
    """
    s_R  = (r_p0**3)/(R0**3)
    beta = 2*nu/(1-nu)
    xi   = 1 - (1+beta)/(beta-2)

    # ----- poišči začetni p_curr, pri katerem je r_p0+u_r > 0 --------------
    p_curr = max(p_i0, 1.0)
    for _ in range(50):
        A_start = (-p0 + p_curr*s_R)/(alpha*(1+nu)*(1-s_R))
        u_start = A_start*r_p0*xi - r_p0*p_curr/(alpha*(beta-2))
        if r_p0 + u_start > 0:
            break
        p_curr *= 2.0


    prev_good_p = p_curr
    for _ in range(nmax):
        A      = (-p0 + p_curr*s_R)/(alpha*(1+nu)*(1-s_R))
        u_rp   = A*r_p0*xi - r_p0*p_curr/(alpha*(beta-2))
        if r_p0 + u_rp <= 0:
            p_curr = 0.5*(p_curr + prev_good_p)
            continue

        p_new  = p_i0*(r_p0/(r_p0+u_rp))**3
        f_val  = p_curr - p_new

        dp     = p_curr*1e-6 if p_curr != 0 else tol
        p_pert = p_curr + dp
        A_pert = (-p0 + p_pert*s_R)/(alpha*(1+nu)*(1-s_R))
        u_pert = A_pert*r_p0*xi - r_p0*p_pert/(alpha*(beta-2))
        if r_p0 + u_pert <= 0:
            p_curr = 0.5*(p_curr + prev_good_p)
            continue

        p_new_p = p_i0*(r_p0/(r_p0+u_pert))**3
        f_pert  = p_pert - p_new_p
        df_dp   = (f_pert - f_val)/dp
        if df_dp == 0:
            break

        p_next = p_curr - f_val/df_dp
        if abs(p_next - p_curr) < tol:
            return A, p_new

        if p_next <= 0:
            p_next = 0.5*(p_curr + prev_good_p)

        prev_good_p = p_curr
        p_curr = p_next

def u_hollow_iso(r, A, p_in, alpha, nu, r_p0):
    beta = 2*nu/(1-nu)
    return (A*r*(1 - (1+beta)/(beta-2)*(r_p0**3)/r**3)
            - r*p_in/(alpha*(beta-2))*(r_p0**3)/r**3)

# ---------------- setup figure --------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))                       # večja slika obeh kroglic
plt.subplots_adjust(left=0.15, right=0.8, bottom=-0.2, top=1.0)  # več prostora za risanje
ax.set_aspect('equal')
ax.set_xlim(-0.1, 0.18)
ax.set_ylim(-0.14, 0.08)
ax.axis('off')

# --- referenčni obrisi ----------------------------------------------------
ref_s      = Circle((0.00, 0), 0.04, ec='0.7', ls='--', lw=1, fill=False)
ref_h_in   = Circle((0.10, 0), 0.02, ec='0.7', ls='--', lw=1, fill=False)
ref_h_out  = Circle((0.10, 0), 0.04, ec='0.7', ls='--', lw=1, fill=False)
ax.add_patch(ref_s); ax.add_patch(ref_h_in); ax.add_patch(ref_h_out)

# --- polnilo trde sredice -------------------------------------------------
core_fill = Circle((0.00, 0), 0.02, facecolor='0.8', edgecolor='none', alpha=0.5)
ax.add_patch(core_fill)

# --- aktivni obrisi -------------------------------------------------------
cir_s        = Circle((0.00, 0), 0.04, ec='C0', lw=2, fill=False)
cir_h_c_in   = Circle((0.10, 0), 0.02, ec='C0', lw=2, fill=False)
cir_h_c_out  = Circle((0.10, 0), 0.04, ec='C0', lw=2, fill=False)
cir_h_nr_in  = Circle((0.10, 0), 0.02, ec='C1', lw=2, fill=False)
cir_h_nr_out = Circle((0.10, 0), 0.04, ec='C1', lw=2, fill=False)
for c in (cir_s, cir_h_c_in, cir_h_c_out, cir_h_nr_in, cir_h_nr_out):
    ax.add_patch(c)

# --- naslovi kroglic ------------------------------------------------------
ax.text(0.00, 0.065, "Trda sredica",  ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='black')
ax.text(0.10, 0.065, "Votla sredica", ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='black')

# --- delta-tekst pod kroglicama ------------------------------------------
delta_s_txt    = ax.text(0.00, -0.06,  "", ha='center', va='top',
                         fontsize=11, color='C0')
delta_h_c_txt  = ax.text(0.10, -0.06,  "", ha='center', va='top',
                         fontsize=11, color='C0')
delta_h_nr_txt = ax.text(0.10, -0.065, "", ha='center', va='top',
                         fontsize=11, color='C1')

# --- p' izpis pod votlo kroglico -----------------------------------------
p_prime_txt = ax.text(0.10, -0.075, "", ha='center', va='top',
                      fontsize=12, color='C1')

# ----------------- drsniki ------------------------------------------------
ax_p0  = plt.axes([0.10, 0.06, 0.60, 0.03], facecolor='0.8')
ax_pi0 = plt.axes([0.10, 0.03, 0.60, 0.03], facecolor='0.8')
sld_p0  = Slider(ax_p0,  "p₀ [kPa]",   0, 25000, valinit=0,   valstep=10)
sld_pi0 = Slider(ax_pi0, "p'₀ [kPa]",  0,   5000, valinit=100, valstep=10)

ax_E  = plt.axes([0.80, 0.50, 0.03, 0.35], facecolor='0.8')
ax_nu = plt.axes([0.85, 0.50, 0.03, 0.35], facecolor='0.8')
ax_rp = plt.axes([0.90, 0.50, 0.03, 0.35], facecolor='0.8')
ax_R  = plt.axes([0.95, 0.50, 0.03, 0.35], facecolor='0.8')
sld_E  = Slider(ax_E,  "E\n[MPa]", 1, 20, valinit=5,  valstep=0.5,
                orientation='vertical')
sld_nu = Slider(ax_nu, "ν", 0.45, 0.69, valinit=0.49, valstep=0.02,
                orientation='vertical')
sld_rp = Slider(ax_rp, "r' [mm]", 5, 30, valinit=20, valstep=1,
                orientation='vertical')
sld_R  = Slider(ax_R,  "R [mm]", 25, 60, valinit=40, valstep=1,
                orientation='vertical')

# checkbox samo constant in Newton
ax_cb = plt.axes([0.02, 0.48, 0.22, 0.15], facecolor='0.8')
chk   = CheckButtons(ax_cb,
        ["Konstanten p'′", "Izotermno stiskanje"],
        [True, True])
for i, label in enumerate(chk.labels):
    label.set_color(['C0', 'C1'][i])

# -------------------- posodobitev -----------------------------------------
def update(_=None):
    # preberi vrednosti iz drsnikov
    p0_kPa  = sld_p0.val
    pi0_kPa = sld_pi0.val
    E       = sld_E.val   * 1e6
    nu      = sld_nu.val
    r_p0    = sld_rp.val  / 1e3
    R0      = sld_R.val   / 1e3

    # zagotovimo r_p0 < R0
    sep = 1e-3
    if r_p0 > R0 - sep:
        r_p0 = R0 - sep; sld_rp.set_val(r_p0*1e3)
    if R0 < r_p0 + sep:
        R0 = r_p0 + sep; sld_R.set_val(R0*1e3)

    # pretvori v Pa
    p0   = p0_kPa  * 1e3
    p_i0 = pi0_kPa * 1e3
    alpha, _ = compute_materials(E, nu)

    # posodobi referenčne kroge
    ref_s.radius     = R0
    ref_h_in.radius  = r_p0
    ref_h_out.radius = R0
    core_fill.radius = r_p0

    # trda sredica
    R_s = R0 + u_solid_outer(p0, alpha, nu, r_p0, R0)
    cir_s.radius = R_s
    delta_s_txt.set_text(f"ΔR = {(R_s-R0)*1e3:5.2f} mm")

    # ------------------------------------------------------------
    #  modra votlina – Konstanten p'′
    # ------------------------------------------------------------
    converged_c = True
    try:
        R_c = R0 + u_hollow_const(R0,   p0, p_i0, alpha, nu, r_p0, R0)
        r_c = r_p0 + u_hollow_const(r_p0, p0, p_i0, alpha, nu, r_p0, R0)
        if r_c <= 0 or R_c <= 0 or r_c >= R_c:
            raise ValueError("votlina se je zaprla – ne-fizičen rezultat")
        cir_h_c_out.radius = R_c
        cir_h_c_in.radius  = r_c
        delta_h_c_txt.set_text(
            f"ΔR = {(R_c-R0)*1e3:5.2f} mm, Δr′ = {(r_c-r_p0)*1e3:5.2f} mm")
        delta_h_c_txt.set_color('C0')
    except Exception:
        converged_c = False
        cir_h_c_out.set_visible(False)
        cir_h_c_in.set_visible(False)
        delta_h_c_txt.set_text("⚠ VOTLINA SE JE SESEDLA ⚠")
        delta_h_c_txt.set_color('red')

    # oranžna votlina – Izotermno stiskanje
    converged_nr = True
    try:
        A_nr, p_in_nr = hollow_isothermal_newton(p0, p_i0, alpha, nu, r_p0, R0)
        R_nr = R0 + u_hollow_iso(R0,   A_nr, p_in_nr, alpha, nu, r_p0)
        r_nr = r_p0 + u_hollow_iso(r_p0, A_nr, p_in_nr, alpha, nu, r_p0)
        cir_h_nr_out.radius = R_nr
        cir_h_nr_in.radius  = r_nr
        delta_h_nr_txt.set_text(
            f"ΔR = {(R_nr-R0)*1e3:5.2f} mm, Δr′ = {(r_nr-r_p0)*1e3:5.2f} mm")
        delta_h_nr_txt.set_visible(True)
    except RuntimeError:
        converged_nr = False
        cir_h_nr_out.set_visible(False)
        cir_h_nr_in.set_visible(False)
        delta_h_nr_txt.set_visible(False)

    # ---------------- vidnost elementov -------------------------
    vis = dict(zip([lbl.get_text() for lbl in chk.labels], chk.get_status()))
    cir_h_c_in.set_visible(vis["Konstanten p'′"] and converged_c)
    cir_h_c_out.set_visible(vis["Konstanten p'′"] and converged_c)
    cir_h_nr_in.set_visible(vis["Izotermno stiskanje"] and converged_nr)
    cir_h_nr_out.set_visible(vis["Izotermno stiskanje"] and converged_nr)
    delta_h_c_txt.set_visible(vis["Konstanten p'′"])

    if not converged_nr and vis["Izotermno stiskanje"]:
        p_prime_txt.set_text("⚠ votlina (NR) zaprta ⚠"); p_prime_txt.set_color('red')
    elif pi0_kPa < 0:
        p_prime_txt.set_text("⚠ p'₀ < 0 ⚠");             p_prime_txt.set_color('red')
    else:
        p_prime_txt.set_text(f"p' = {p_in_nr/1e3:.0f} kPa")
        p_prime_txt.set_color('C1')

    fig.canvas.draw_idle()

# povezave drsnikov in checkboxa
for s in (sld_p0, sld_pi0, sld_E, sld_nu, sld_rp, sld_R):
    s.on_changed(update)
chk.on_clicked(update)

# začetni prikaz
update()
plt.show()
