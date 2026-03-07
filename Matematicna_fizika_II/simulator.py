import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# ---------------- Matplotlib ----------------
mpl.rcParams['text.usetex']      = False
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family']      = 'serif'

# ---------------- Začetne vrednosti ----------------
R1_init, R2_init         = 0.5, 1.0
omega1_init, omega2_init = 1.0, 0.0
eta_init                 = 1.0
N                        = 600
GAP_MIN                  = 0.05
EDGE_COLOR               = '#9e9e9e'
EDGE_WIDTH               = 1.5

# Dinamične “silnice” (samo v ekvatorialnem preseku)
N_RINGS   = 5
PTS_RING  = 28
DT        = 0.03
SPEED_K   = 1.0
ARROW_K   = 0.25

# ---------------- Pomožne funkcije ----------------
def coeffs(R1, R2, omega1, omega2):
    """
    Stabilna oblika za A, B:
      A = (ω2 R2^3 - ω1 R1^3) / (R2^3 - R1^3)
      B = - (ω2 - ω1) R1^3 R2^3 / (R2^3 - R1^3)
    """
    d3 = R2**3 - R1**3
    if abs(d3) < 1e-12:
        d3 = np.sign(d3) * 1e-12 if d3 != 0 else 1e-12
    A = (omega2 * R2**3 - omega1 * R1**3) / d3
    B = - (omega2 - omega1) * (R1**3 * R2**3) / d3
    return A, B

def compute_fields(R1, R2, omega1, omega2):
    # ekvatorialna mreža
    x = np.linspace(-R2, R2, N)
    y = np.linspace(-R2, R2, N)
    X, Y = np.meshgrid(x, y)
    r    = np.hypot(X, Y)

    A, B = coeffs(R1, R2, omega1, omega2)
    mask_eq = (r >= R1) & (r <= R2)
    v_phi_eq = np.full_like(r, np.nan, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        v_phi_eq[mask_eq] = (A*r + B/r**2)[mask_eq]

    # meridionalna mreža
    z = np.linspace(-R2, R2, N)
    X2, Z = np.meshgrid(x, z)
    r2 = np.hypot(X2, Z)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = np.divide(Z, r2, out=np.zeros_like(Z), where=r2>0)
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    mask_mer = (r2 >= R1) & (r2 <= R2)

    v_phi_mer = np.full_like(r2, np.nan, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        v_phi_mer[mask_mer] = (A*r2 + B/r2**2)[mask_mer] * np.sin(theta[mask_mer])

    vmax = np.nanmax(np.abs(v_phi_eq)) if np.any(~np.isnan(v_phi_eq)) else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return v_phi_eq, v_phi_mer, vmax

def make_particle_rings(R1, R2, n_rings, pts_ring):
    gap = max(R2 - R1, 1e-6)
    r_min = R1 + 0.02*gap
    r_max = R2 - 0.02*gap
    if r_max <= r_min:  # rezerva za zelo majhno režo
        r_min = R1 + 0.25*gap
        r_max = R2 - 0.25*gap
    r_min = max(R1, r_min)
    r_max = min(R2, r_max)
    radii = np.linspace(r_min, r_max, n_rings)
    base  = np.linspace(0, 2*np.pi, pts_ring, endpoint=False)[None, :]
    phases = np.random.uniform(0, 2*np.pi, size=(n_rings, 1))
    ang = (base + phases) % (2*np.pi)
    return radii, ang

def ring_positions_xy(radii, ang):
    x = (radii[:, None] * np.cos(ang)).ravel()
    y = (radii[:, None] * np.sin(ang)).ravel()
    return x, y

def omega_equatorial(r, A, B):
    with np.errstate(divide='ignore', invalid='ignore'):
        return A + B / (r**3)

def torque_M(R1, R2, omega1, omega2, eta):
    """
    Stabilna oblika za navor:
      M = 8π η * [ R1^3 R2^3 (ω2 - ω1) / (R2^3 - R1^3) ]
    """
    d3 = R2**3 - R1**3
    if abs(d3) < 1e-12:
        d3 = np.sign(d3) * 1e-12 if d3 != 0 else 1e-12
    return 8*np.pi*eta * (R1**3 * R2**3 * (omega2 - omega1) / d3)

# ---------------- Prvi izračun ----------------
v_eq, v_mer, vmax = compute_fields(R1_init, R2_init, omega1_init, omega2_init)

# ---------------- Prikaz ----------------
fig = plt.figure(figsize=(12, 6))
gs  = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 0.045], wspace=0.08)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
cax = fig.add_subplot(gs[0, 2])
fig.subplots_adjust(bottom=0.2, top=0.88)

c1 = ax1.imshow(v_eq, extent=[-R2_init, R2_init, -R2_init, R2_init],
                origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax,
                interpolation='bilinear')
c2 = ax2.imshow(v_mer, extent=[-R2_init, R2_init, -R2_init, R2_init],
                origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax,
                interpolation='bilinear')

inner_circle1 = Circle((0, 0), R1_init, fill=False, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
outer_circle1 = Circle((0, 0), R2_init, fill=False, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
inner_circle2 = Circle((0, 0), R1_init, fill=False, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
outer_circle2 = Circle((0, 0), R2_init, fill=False, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
ax1.add_patch(inner_circle1); ax1.add_patch(outer_circle1)
ax2.add_patch(inner_circle2); ax2.add_patch(outer_circle2)

ax1.plot(0, 0, 'o', markersize=12, markerfacecolor='white', markeredgecolor='black', linewidth=1.5)
ax1.plot(0, 0, 'o', markersize=6, markerfacecolor='black')
z_text1 = ax1.text(0.07*R2_init, 0.07*R2_init, '$z$', fontsize=16, ha='center', va='center')

q = ax2.quiver(0, -1.1*R2_init, 0, 2.2*R2_init,
               angles='xy', scale_units='xy', scale=1,
               headwidth=4, headlength=6, width=0.005,
               color='black', clip_on=False)
z_text2 = ax2.text(0.05*R2_init, 1.1*R2_init, '$z$', fontsize=16, ha='center', va='bottom')

padding = 1.3*R2_init
for ax in (ax1, ax2):
    ax.set_xlim(-padding, padding)
    ax.set_ylim(-padding, padding)
    ax.set_aspect('equal')
    ax.axis('off')

ax1.set_title('Ekvatorialni presek (θ=π/2)', y=1.03, pad=0)
ax2.set_title('Meridionalni presek (y=0)',  y=1.03, pad=0)

cbar = fig.colorbar(c1, cax=cax, orientation='vertical')
cbar.set_label(r'$v_\phi(r,\theta)$', fontsize=16)
cbar.ax.tick_params(labelsize=12)

# ---------- Dinamične silnice ----------
radii_eq, ang_eq = make_particle_rings(R1_init, R2_init, N_RINGS, PTS_RING)
x1p, y1p = ring_positions_xy(radii_eq, ang_eq)
A0, B0   = coeffs(R1_init, R2_init, omega1_init, omega2_init)
r_flat   = np.repeat(radii_eq[:, None], PTS_RING, axis=1).ravel()
tx       = (-np.sin(ang_eq)).ravel()
ty       = ( np.cos(ang_eq)).ravel()
with np.errstate(divide='ignore', invalid='ignore'):
    vphi0    = (A0 * r_flat + np.divide(B0, r_flat**2, out=np.zeros_like(r_flat), where=r_flat>0))
U0, V0   = tx * vphi0 * ARROW_K, ty * vphi0 * ARROW_K

quiv_eq = ax1.quiver(x1p, y1p, U0, V0,
                     angles='xy', scale_units='xy', scale=1,
                     width=0.004, headwidth=4, headlength=6, color='black')

# ----------- Izpis navora -----------
M0 = torque_M(R1_init, R2_init, omega1_init, omega2_init, eta_init)
torque_text = fig.text(
    0.5, 0.25,
    r"$M = 8\pi\,\eta\,\frac{{R_1^3 R_2^3\,(\omega_2-\omega_1)}}{{R_2^3-R_1^3}}\;=\;{:.2f}$".format(M0),
    ha='center', va='center', fontsize=16,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#666666')
)

# ---------------- Sliderji ----------------
ax_R2     = plt.axes([0.12, 0.16, 0.76, 0.03])
ax_R1     = plt.axes([0.12, 0.12, 0.76, 0.03])
ax_omega1 = plt.axes([0.12, 0.08, 0.76, 0.03])
ax_omega2 = plt.axes([0.12, 0.04, 0.76, 0.03])
ax_eta    = plt.axes([0.12, 0.00, 0.76, 0.03])

sR2     = Slider(ax_R2,     r'$R_2$',      0.6, 2.0,  valinit=R2_init,     valstep=0.01)
sR1     = Slider(ax_R1,     r'$R_1$',      0.1, 1.95, valinit=R1_init,     valstep=0.01)
sOmega1 = Slider(ax_omega1, r'$\omega_1$', -5.0, 5.0, valinit=omega1_init, valstep=0.01)
sOmega2 = Slider(ax_omega2, r'$\omega_2$', -5.0, 5.0, valinit=omega2_init, valstep=0.01)
sEta    = Slider(ax_eta,    r'$\eta$',     0.01, 5.0, valinit=eta_init,    valstep=0.01)

# shranimo prejšnje vrednosti, da zaznamo spremembe
prev_R1, prev_R2 = R1_init, R2_init
prev_omega1, prev_omega2 = omega1_init, omega2_init
_updating = False

def recalc_all(R1_new, R2_new, omega1_new, omega2_new):
    """Preračun polj, barvne skale, obrob, osi in obročev silnic."""
    global radii_eq, ang_eq, q

    v_eq, v_mer, vmax = compute_fields(R1_new, R2_new, omega1_new, omega2_new)
    c1.set_data(v_eq);  c1.set_extent([-R2_new, R2_new, -R2_new, R2_new]);  c1.set_clim(-vmax, vmax)
    c2.set_data(v_mer); c2.set_extent([-R2_new, R2_new, -R2_new, R2_new]);  c2.set_clim(-vmax, vmax)
    cbar.update_normal(c1)

    inner_circle1.set_radius(R1_new); outer_circle1.set_radius(R2_new)
    inner_circle2.set_radius(R1_new); outer_circle2.set_radius(R2_new)

    z_text1.set_position((0.07*R2_new, 0.07*R2_new))
    z_text2.set_position((0.05*R2_new, 1.1*R2_new))

    try: q.remove()
    except Exception: pass
    q = ax2.quiver(0, -1.1*R2_new, 0, 2.2*R2_new,
                   angles='xy', scale_units='xy', scale=1,
                   headwidth=4, headlength=6, width=0.005,
                   color='black', clip_on=False)

    padding = 1.3*R2_new
    for ax in (ax1, ax2):
        ax.set_xlim(-padding, padding)
        ax.set_ylim(-padding, padding)

    # ob spremembi R resetiramo obroče (da ostanejo v območju)
    radii_eq, ang_eq = make_particle_rings(R1_new, R2_new, N_RINGS, PTS_RING)

def on_change(val):
    global prev_R1, prev_R2, prev_omega1, prev_omega2, _updating
    if _updating:
        return

    _updating = True  # zadržimo rekurzivne klice med set_val

    # Preberi trenutno stanje
    R1_new     = sR1.val
    R2_new     = sR2.val
    omega1_new = sOmega1.val
    omega2_new = sOmega2.val
    eta_new    = sEta.val

    dR1 = abs(R1_new - prev_R1)
    dR2 = abs(R2_new - prev_R2)

    # ohranjaj minimalno režo – brez zgodnjega return!
    if R2_new - R1_new < GAP_MIN:
        if dR2 > dR1:
            # premikal se je R2 -> pomakni R1
            R1_new = max(sR1.valmin, min(R2_new - GAP_MIN, sR1.valmax))
            sR1.set_val(R1_new)
        else:
            # premikal se je R1 -> pomakni R2
            R2_new = max(sR2.valmin, min(R1_new + GAP_MIN, sR2.valmax))
            sR2.set_val(R2_new)

    # Preračun polj, če se je karkoli spremenilo
    if (R1_new != prev_R1) or (R2_new != prev_R2) or (omega1_new != prev_omega1) or (omega2_new != prev_omega2):
        recalc_all(R1_new, R2_new, omega1_new, omega2_new)
        prev_R1, prev_R2 = R1_new, R2_new
        prev_omega1, prev_omega2 = omega1_new, omega2_new

    # Navor (odvisen tudi od η) — vedno osveži
    M = torque_M(R1_new, R2_new, omega1_new, omega2_new, eta_new)
    torque_text.set_text(
        r"$M = 8\pi\,\eta\,\frac{{R_1^3 R_2^3\,(\omega_2-\omega_1)}}{{R_2^3-R_1^3}}\;=\;{:.2f}$".format(M)
    )

    _updating = False
    fig.canvas.draw_idle()

for s in (sR1, sR2, sOmega1, sOmega2, sEta):
    s.on_changed(on_change)

# ---------------- Animacija ----------------
def animate(frame):
    R1 = sR1.val; R2 = sR2.val
    omega1 = sOmega1.val; omega2 = sOmega2.val
    A, B = coeffs(R1, R2, omega1, omega2)

    # posodobitev kotov: d(alpha) = (A + B/r^3)*dt
    dalpha = (omega_equatorial(radii_eq, A, B) * DT * SPEED_K)[:, None]
    ang_eq[:] = (ang_eq + dalpha) % (2*np.pi)

    # pozicije in tangentne smeri iz kota (brez arctan2)
    cosA = np.cos(ang_eq); sinA = np.sin(ang_eq)
    x = (radii_eq[:, None] * cosA).ravel()
    y = (radii_eq[:, None] * sinA).ravel()

    r_flat = np.repeat(radii_eq[:, None], PTS_RING, axis=1).ravel()
    with np.errstate(divide='ignore', invalid='ignore'):
        vphi   = (A * r_flat + np.divide(B, r_flat**2, out=np.zeros_like(r_flat), where=r_flat>0))
    tx = (-sinA).ravel()
    ty = ( cosA).ravel()
    U = tx * vphi * ARROW_K
    V = ty * vphi * ARROW_K

    quiv_eq.set_offsets(np.c_[x, y])
    quiv_eq.set_UVC(U, V)
    return quiv_eq,

anim = FuncAnimation(fig, animate, interval=16, blit=False)
plt.show()
