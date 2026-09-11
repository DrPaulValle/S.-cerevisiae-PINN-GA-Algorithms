"""
Mechanistic Nonlinear Dynamical Modeling for Parameter Estimation: A Comparison 
of Physics Informed Neural Networks and Genetic Algorithms

Paul A. Valle¹, Yolocuauhtli Salazar², Jesus B. Páez-Lerma³, N. Oscar Soto-Cruz³,
Luis N. Coria¹, Iván A. García¹, Michell V. González-Campos³

¹Posgrado en Ciencias de la Ingeniería, BioMath Research Group, Tecnológico Nacional de México/IT Tijuana, Tijuana, México. paul.valle@tectijuana.edu.mx, luis.coria@tectijuana.edu.mx, D25210003@tectijuana.edu.mx
²Departamento de Ingeniería Eléctrica y Electrónica, Tecnológico Nacional de México/IT Durango, Durango, México. ysalazar@itdurango.edu.mx
³Departamento de Ingenierías Química y Bioquímica, Tecnológico Nacional de México/IT Durango, Durango, México. jpaez@itdurango.edu.mx, nsoto@itdurango.edu.mx

Augmented mathematical model
To describe the dynamical interactions among the variables involved in an alcoholic 
fermentation process carried out by the ITD00185 yeast of S. cerevisiae, we consider 
biomass w(t), glucose x(t) and fructose y(t) as substrates, and ethanol z(t) as the 
main metabolic product in a batch system. The evolution of these state variables is 
represented by the following nonlinear first-order ordinary differential equations:
    
        dw/dt = (p1 + p3*(x+y)/(x0+y0))*w*exp(-p2*W) - p4*w
        dW/dt = w
        dx/dt = -p5*x*w - p6*x
        dy/dt = -p7*y*w - p8*y
        dz/dt = p9*(x+y)*w - p10*z
        
where W(t) represents the cumulative biomass concentration over time, and all state 
variables are expressed in g/L, while time t is measured in hours.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress TensorFlow INFO/WARNING messages
import re
import time
import gc
import io
import copy
import contextlib
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
tf.get_logger().setLevel("ERROR") # Show only TensorFlow errors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Utilities
def natural_key(name: str):
    """Sorting key for sheet names like 'E1', 'E2', ..., 'E10'."""
    m = re.match(r"^E(\d+)$", str(name).strip(), re.IGNORECASE)
    return int(m.group(1)) if m else float("inf")

def preprocess_series(arr, sigma=1.0, normalize=False):
    """Optionally smooth and/or normalize a 1D numeric series."""
    arr = np.asarray(arr, dtype=float)
    if sigma is not None and sigma > 0:
        arr = gaussian_filter1d(arr, sigma=sigma)
    if normalize:
        m = np.max(arr)
        arr = arr / (m if m != 0 else 1.0)
    return arr

def plotsys(
    t_data, w_data, x_data, y_data, z_data,
    t_plot, w_p, x_p, y_p, z_p,
    t_grid, w_e, x_e, y_e, z_e,
    title="ODE model: PINN vs Euler"):
    """Plot a 2×2 comparison for (w, x, y, z): data vs PINN vs Euler."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    series = [
        (0, 0, "w(t): Biomass",   t_data, w_data, t_plot, w_p, t_grid, w_e, "w(t)"),
        (0, 1, "x(t): Glucose",   t_data, x_data, t_plot, x_p, t_grid, x_e, "x(t)"),
        (1, 0, "y(t): Fructose",  t_data, y_data, t_plot, y_p, t_grid, y_e, "y(t)"),
        (1, 1, "z(t): Ethanol",   t_data, z_data, t_plot, z_p, t_grid, z_e, "z(t)")]

    for r, c, ttl, td, yd, tp, yp, tg, ye, lab in series:
        ax = axes[r, c]
        ax.scatter(td, yd, s=50, color='#005F02', edgecolors="k", alpha=0.85, label=f"{lab}: data")
        ax.plot(tp, yp, lw=2, color='#134E8E', label=f"{lab}: PINN")
        ax.plot(tg, ye, "--", lw=2, color='#C00707', label=f"{lab}: Euler")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig

def make_table_page(
    stats, df_params, sheet_name, ic_estimates=None,
    main_fontsize=10, header_fontsize=12, ic_fontsize=10,
    ic_row_height_scale=1.3):
    """One-page PDF: header metrics + parameter table + optional IC table."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    R2 = stats.get("R2", np.nan)
    R2a = stats.get("R2_adj", np.nan)
    RSS = stats.get("RSS", np.nan)
    AIC = stats.get("AIC", np.nan)
    cond = stats.get("cond_JTJ", np.nan)
    used_pinv = stats.get("used_pinv", False)

    header = (
        f"{sheet_name}\n"
        f"R2={R2:.4f}, R2_adj={R2a:.4f}, RSS={RSS:.3e}, AIC={AIC:.3e}\n"
        f"cond(JTJ)={cond:.2e}, used_pinv={used_pinv}")
    ax.text(0.02, 0.95, header, fontsize=header_fontsize, va="top", family="monospace")

    def fmt(v):
        try:
            v = float(v)
            return f"{v:.6e}" if np.isfinite(v) else "nan"
        except Exception:
            return str(v)

    df_disp = df_params.copy()
    for c in ["estimate", "SE", "MOE_95", "CI_low", "CI_high", "p_value"]:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].map(fmt)

    has_ic = bool(ic_estimates)
    main_bbox = [0.02, 0.30, 0.96, 0.55] if has_ic else [0.02, 0.08, 0.96, 0.80]

    t_main = ax.table(
        cellText=df_disp.values.tolist(),
        colLabels=list(df_disp.columns),
        cellLoc="center",
        colLoc="center",
        bbox=main_bbox)
    t_main.auto_set_font_size(False)
    t_main.set_fontsize(main_fontsize)

    if has_ic:
        ic_items = list(ic_estimates.items())
        df_ic = pd.DataFrame({
            "IC": [k for k, _ in ic_items],
            "estimate": [fmt(v) for _, v in ic_items]})

        t_ic = ax.table(
            cellText=df_ic.values.tolist(),
            colLabels=list(df_ic.columns),
            cellLoc="center",
            colLoc="center",
            bbox=[0.02, 0.05, 0.45, 0.20])
        t_ic.auto_set_font_size(False)
        t_ic.set_fontsize(ic_fontsize)
        t_ic.scale(1.0, ic_row_height_scale)

    return fig

# Euler solver
def euler_ODE(t_grid, w0, W0, x0, y0, z0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    """
    Forward Euler integration for the 5-state model. Model:
        dw/dt = (p1 + p3*(x+y)/(x0+y0))*w*exp(-p2*W) - p4*w
        dW/dt = w
        dx/dt = -p5*x*w - p6*x
        dy/dt = -p7*y*w - p8*y
        dz/dt = p9*(x+y)*w - p10*z
        Returns: (w, x, y, z) evaluated on t_grid (1D numpy arrays).
    """
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 2:
        raise ValueError("t_grid must have at least 2 points.")
    if np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid must be strictly increasing.")

    N = len(t_grid)
    w = np.zeros(N); W = np.zeros(N); x = np.zeros(N); y = np.zeros(N); z = np.zeros(N)
    w[0], W[0], x[0], y[0], z[0] = map(float, (w0, W0, x0, y0, z0))

    for i in range(N - 1):
        dt = t_grid[i + 1] - t_grid[i]
        wi, Wi, xi, yi, zi = w[i], W[i], x[i], y[i], z[i]

        exp_term = np.exp(np.clip(-p2 * Wi, -50.0, 50.0))
        dw = (p1 + p3 * (xi + yi) / (x0 + y0))* wi * exp_term - p4 * wi
        dW = wi
        dx = -p5 * xi * wi - p6 * xi
        dy = -p7 * yi * wi - p8 * yi
        dz = p9 * (xi + yi) * wi - p10 * zi

        w[i + 1] = wi + dt * dw; w[i + 1] = max(w[i + 1], 0.0)
        W[i + 1] = Wi + dt * dW
        x[i + 1] = xi + dt * dx; x[i + 1] = max(x[i + 1], 0.0)
        y[i + 1] = yi + dt * dy; y[i + 1] = max(y[i + 1], 0.0)
        z[i + 1] = zi + dt * dz; z[i + 1] = max(z[i + 1], 0.0)

    return w, x, y, z

# Parameter statistics (Euler-based, central differences)
def stats_at_optimum(t_data, t_grid, f_obs, P, w0, W0, x0, y0, z0, eps_base=1e-6):
    """Fit metrics + parameter uncertainty via central-difference Jacobian (Euler solver)."""
    t_data = np.asarray(t_data, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    f_obs = np.asarray(f_obs, dtype=float).reshape(-1, 1)
    n = f_obs.shape[0]

    P = np.asarray(P, dtype=float).ravel()
    if P.size != 10:
        raise ValueError("P must have length 10: [p1,...,p10].")

    idx_est = [0, 1, 2, 4, 6, 8]
    pnames = ["p1", "p2", "p3", "p5", "p7", "p9"]
    base = P[idx_est].copy()
    p = base.size

    def simulate_pack(P_full):
        p1_, p2_, p3_, p4_, p5_, p6_, p7_, p8_, p9_, p10_ = P_full
        w_sim, x_sim, y_sim, z_sim = euler_ODE(
            t_grid, w0, W0, x0, y0, z0,
            p1_, p2_, p3_, p4_, p5_, p6_, p7_, p8_, p9_, p10_)
        w_i = np.interp(t_data, t_grid, np.asarray(w_sim).ravel())
        x_i = np.interp(t_data, t_grid, np.asarray(x_sim).ravel())
        y_i = np.interp(t_data, t_grid, np.asarray(y_sim).ravel())
        z_i = np.interp(t_data, t_grid, np.asarray(z_sim).ravel())
        return np.concatenate([w_i, x_i, y_i, z_i], axis=0).reshape(-1, 1)

    f_pred = simulate_pack(P)
    resid = f_obs - f_pred
    rss = float(np.sum(resid**2))
    n_time = t_data.size
    n_states = 4
    
    if n != n_states * n_time:
        raise ValueError(
            f"Expected {n_states*n_time} observations "
            f"({n_states} states x {n_time} time points), but received {n}.")
    
    f_obs_mat = f_obs.reshape(n_states, n_time)
    ss_tot = float(np.sum((f_obs_mat - np.mean(f_obs_mat, axis=1, keepdims=True))**2 ))
    R2 = (
        np.nan
        if ss_tot <= 0
        else 1.0 - rss / max(ss_tot, 1e-16))
    
    k = p #For consistency with literature formulas
    R2_adj = (
        np.nan
        if (not np.isfinite(R2) or (n - k - 1) <= 0)
        else 1.0 - (1.0 - R2) * (n - 1.0) / (n - k - 1.0))

    J = np.zeros((n, p), dtype=float)
    for j in range(p):
        h = eps_base * (1.0 + abs(base[j]))
        Pp, Pm = P.copy(), P.copy()
        Pp[idx_est[j]] += h
        Pm[idx_est[j]] -= h
        J[:, j] = ((simulate_pack(Pp) - simulate_pack(Pm)) / (2.0 * h)).ravel()

    df = n - p
    if df <= 0:
        nanv = np.full(p, np.nan)
        return dict(
            R2=R2, R2_adj=R2_adj, RSS=rss, AIC=np.nan,
            se=nanv, moe=nanv, ci_lo=nanv, ci_hi=nanv, pvals=nanv,
            names=pnames, params=base, cond_JTJ=np.nan, used_pinv=False)

    JTJ = J.T @ J
    try:
        cond = float(np.linalg.cond(JTJ))
    except np.linalg.LinAlgError:
        cond = float("inf")
    try:
        JTJ_inv = np.linalg.inv(JTJ)
        used_pinv = False
    except np.linalg.LinAlgError:
        JTJ_inv = np.linalg.pinv(JTJ)
        used_pinv = True

    sigma2 = rss / df
    cov = sigma2 * JTJ_inv
    se = np.sqrt(np.diag(cov).clip(min=0.0))
    alpha = 0.05
    tcrit = tdist.ppf(1-alpha/2, df=df)
    moe = tcrit * se
    ci_lo, ci_hi = base - moe, base + moe

    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, np.abs(base / se), np.nan)
    pvals = 2.0 * tdist.sf(tvals, df=df)

    aic = n * np.log(max(rss, 1e-16) / n) + 2 * k
    if (k > 0) and ((n / k) < 40) and ((n - k - 1) > 0):
        aic += (2 * k * (k + 1)) / (n - k - 1)
    
    return dict(
        R2=R2, R2_adj=R2_adj, RSS=rss, AIC=aic,
        se=se, moe=moe, ci_lo=ci_lo, ci_hi=ci_hi, pvals=pvals,
        names=pnames, params=base, cond_JTJ=cond, used_pinv=used_pinv)

def stats_to_table(stats):
    """Convert stats dict into a tidy parameter table."""
    names = stats["names"]
    return pd.DataFrame({
        "param": names,
        "estimate": stats.get("params", [np.nan] * len(names)),
        "SE": stats["se"],
        "MOE_95": stats["moe"],
        "CI_low": stats["ci_lo"],
        "CI_high": stats["ci_hi"],
        "p_value": stats["pvals"]})

# PINN model (MLP with configurable activation)
def get_activation_fn(name: str):
    """
    Return a TensorFlow activation function by name.
    Supported: tanh, relu, swish, gelu, sigmoid, softplus, elu
    """
    name = str(name).strip().lower()
    if name == "tanh":
        return tf.tanh
    if name == "relu":
        return tf.nn.relu
    if name == "swish":
        return tf.nn.swish
    if name == "gelu":
        # TF gelu exists
        return tf.nn.gelu
    if name == "sigmoid":
        return tf.nn.sigmoid
    if name == "softplus":
        return tf.nn.softplus
    if name == "elu":
        return tf.nn.elu
    raise ValueError(f"Unknown activation '{name}'. Supported: tanh, relu, swish, gelu, sigmoid, softplus, elu.")

class ODEPINN:
    def __init__(
        self,
        alpha=0.999,
        beta_ic=0.5,
        lr=1e-4,
        arch=(1, 128, 128, 128, 5),
        activation="tanh",
        seed=130425):
        """
        MLP-PINN for the 5-state ODE system with latent integral state W(t).
        Trainable:
            - NN weights/biases
            - ODE parameters: p2, p5, p7, p9 (clipped positive)
            - Reparameterized: p1(theta1), p3(theta3) (bounded via sigmoid)
            - Initial condition: w0
        Fixed:
            - p4, p6, p8, p10 constants
            - x0,y0,z0 fixed from first data point per sheet
            - W0 provided externally [W0 = 0]
        """
        self.alpha = float(alpha)
        self.beta_ic = float(beta_ic)
        self.seed = int(seed)
        self.arch = list(arch)
        if len(self.arch) < 3 or self.arch[0] != 1 or self.arch[-1] != 5:
            raise ValueError("arch must start with 1 and end with 5, e.g., (1,128,128,128,5).")
        self.act_name = str(activation)
        self.act = get_activation_fn(self.act_name)

        # fixed parameters
        self.p4 = tf.constant(np.log(2) / 192, dtype=tf.float32)
        self.p6 = tf.constant(np.log(2) / 840960.0, dtype=tf.float32)
        self.p8 = tf.constant(np.log(2) / 1680.0, dtype=tf.float32)
        self.p10 = tf.constant(-np.log(1 - 1/100) / (30.0), dtype=tf.float32)

        # initial conditions
        ic_clip = lambda v: tf.clip_by_value(v, 0.0, 10)
        self.w0_var = tf.Variable(1e-1, dtype=tf.float32, constraint=ic_clip, name="w0_var")
        self.x0_fixed = None
        self.y0_fixed = None
        self.z0_fixed = None
        self.x0_tf = None
        self.y0_tf = None
        self.z0_tf = None
        self._ic_initialized = False

        # trainable positive parameters
        init_val = 1e-1
        clip = lambda v: tf.clip_by_value(v, 1e-2, 1.0)
        self.p2 = tf.Variable(init_val, dtype=tf.float32, constraint=clip, name="p2")
        self.p5 = tf.Variable(init_val, dtype=tf.float32, constraint=clip, name="p5")
        self.p7 = tf.Variable(init_val, dtype=tf.float32, constraint=clip, name="p7")
        self.p9 = tf.Variable(init_val, dtype=tf.float32, constraint=clip, name="p9")

        # --- p1 bounds
        self.p1_min = tf.constant(1e-2, dtype=tf.float32)
        self.p1_max = tf.constant(1.0, dtype=tf.float32)
        p1_min = float(self.p1_min.numpy())
        p1_max = float(self.p1_max.numpy())
        range_p1 = p1_max - p1_min
        eps1 = 1e-8 * range_p1
        p1_init = float(np.clip(0.5, p1_min + eps1, p1_max - eps1))
        s1 = (p1_init - p1_min) / range_p1
        theta1_init = np.log(s1 / (1.0 - s1))
        self.theta1 = tf.Variable(theta1_init, dtype=tf.float32, name="theta1")
        
        # --- p3 bounds
        self.p3_min = tf.constant(1e-2, dtype=tf.float32)
        self.p3_max = tf.constant(1.0, dtype=tf.float32)
        p3_min = float(self.p3_min.numpy())
        p3_max = float(self.p3_max.numpy())
        range_p3 = p3_max - p3_min
        eps3 = 1e-8 * range_p3
        p3_init = float(np.clip(0.5, p3_min + eps3, p3_max - eps3))
        s3 = (p3_init - p3_min) / range_p3
        theta3_init = np.log(s3 / (1.0 - s3))
        self.theta3 = tf.Variable(theta3_init, dtype=tf.float32, name="theta3")

        # build MLP
        self.build_model(seed=self.seed)
        self.optimizer = tf.optimizers.Adam(learning_rate=float(lr))

        # histories
        self.loss_history = []
        self.loss_phys_hist = []
        self.loss_data_hist = []
        self.loss_ic_hist = []
        self._warned_none_derivs = False
        self._warned_none_grads = False

    def p1_value(self):
        s = tf.sigmoid(self.theta1)
        return self.p1_min + (self.p1_max - self.p1_min) * s

    def p3_value(self):
        s = tf.sigmoid(self.theta3)
        return self.p3_min + (self.p3_max - self.p3_min) * s

    def build_model(self, seed=130425):
        """Initialize manual Dense-layer weights/biases according to self.arch."""
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.weights, self.biases = [], []
        for i in range(len(self.arch) - 1):
            in_dim, out_dim = self.arch[i], self.arch[i + 1]
            W = tf.Variable(
                tf.random.truncated_normal([in_dim, out_dim], stddev=0.1, dtype=tf.float32),
                name=f"W_{i}", trainable=True)
            b = tf.Variable(
                tf.zeros([1, out_dim], dtype=tf.float32),
                name=f"b_{i}", trainable=True)
            self.weights.append(W)
            self.biases.append(b)
        self.nn_vars = self.weights + self.biases

    def neural_net(self, t):
        """
        Forward pass: t -> [w, W, x, y, z].
        Hidden activation: configurable (default tanh). Output layer: linear.
        """
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        if len(t.shape) == 1:
            t = tf.reshape(t, (-1, 1))

        X = t
        for i in range(len(self.arch) - 2):
            X = self.act(tf.matmul(X, self.weights[i]) + self.biases[i])
        return tf.matmul(X, self.weights[-1]) + self.biases[-1]

    def compute_first_derivatives(self, t, warn_none=True):
        """
        Compute states and FIRST time-derivatives.
        Returns: w,W,x,y,z and dw,dW,dx,dy,dz (all [N,1]).
        """
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        if len(t.shape) == 1:
            t = tf.reshape(t, (-1, 1))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            pred = self.neural_net(t)
            w, W = pred[:, 0:1], pred[:, 1:2]
            x, y, z = pred[:, 2:3], pred[:, 3:4], pred[:, 4:5]

        dw = tape.gradient(w, t)
        dW = tape.gradient(W, t)
        dx = tape.gradient(x, t)
        dy = tape.gradient(y, t)
        dz = tape.gradient(z, t)
        del tape

        if warn_none and any(g is None for g in (dw, dW, dx, dy, dz)):
            if not self._warned_none_derivs:
                names = ["dw", "dW", "dx", "dy", "dz"]
                vals = [dw, dW, dx, dy, dz]
                missing = [n for n, v in zip(names, vals) if v is None]
                print("WARNING: None derivatives:", missing)
                self._warned_none_derivs = True

        zlike = tf.zeros_like(t)
        dw = zlike if dw is None else dw
        dW = zlike if dW is None else dW
        dx = zlike if dx is None else dx
        dy = zlike if dy is None else dy
        dz = zlike if dz is None else dz

        for v in (w, W, x, y, z, dw, dW, dx, dy, dz):
            tf.ensure_shape(v, [None, 1])

        return w, W, x, y, z, dw, dW, dx, dy, dz

    def total_loss(self, t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0):
        """
        Composite loss: L = alpha*L_phys + (1-alpha)*L_data + beta_ic*L_ic
        """
        wS, WS, xS, yS, zS = self.scales
        w, W, x, y, z, dw, dW, dx, dy, dz = self.compute_first_derivatives(t_physics)
        p1 = self.p1_value()
        p3 = self.p3_value()
        exp_term = tf.exp(tf.clip_by_value(-self.p2 * W, -50.0, 50.0))

        r1 = dw - ((p1 + p3 * (x + y) / (x_fit[0] + y_fit[0]))* w * exp_term - self.p4 * w)
        r2 = dW - w
        r3 = dx - (-self.p5 * x * w - self.p6 * x)
        r4 = dy - (-self.p7 * y * w - self.p8 * y)
        r5 = dz - (self.p9 * (x + y) * w - self.p10 * z)

        L_phys = (
            tf.reduce_mean(tf.square(r1 / wS)) +
            tf.reduce_mean(tf.square(r2 / WS)) +
            tf.reduce_mean(tf.square(r3 / xS)) +
            tf.reduce_mean(tf.square(r4 / yS)) +
            tf.reduce_mean(tf.square(r5 / zS))
        )

        pred_fit = self.neural_net(t_fit)
        w_hat = pred_fit[:, 0:1]
        x_hat = pred_fit[:, 2:3]
        y_hat = pred_fit[:, 3:4]
        z_hat = pred_fit[:, 4:5]

        L_data = (
            tf.reduce_mean(tf.square((w_hat - w_fit) / wS)) +
            tf.reduce_mean(tf.square((x_hat - x_fit) / xS)) +
            tf.reduce_mean(tf.square((y_hat - y_fit) / yS)) +
            tf.reduce_mean(tf.square((z_hat - z_fit) / zS))
        )

        pred0 = self.neural_net(t0)
        w0_hat = pred0[:, 0:1]
        W0_hat = pred0[:, 1:2]
        x0_hat = pred0[:, 2:3]
        y0_hat = pred0[:, 3:4]
        z0_hat = pred0[:, 4:5]
        w0 = tf.reshape(self.w0_var, (1, 1))

        L_ic = (
            tf.reduce_mean(tf.square((w0_hat - w0) / wS)) +
            tf.reduce_mean(tf.square((W0_hat - W0) / WS)) +
            tf.reduce_mean(tf.square((x0_hat - self.x0_tf) / xS)) +
            tf.reduce_mean(tf.square((y0_hat - self.y0_tf) / yS)) +
            tf.reduce_mean(tf.square((z0_hat - self.z0_tf) / zS))
        )

        L_total = self.alpha * L_phys + (1.0 - self.alpha) * L_data + self.beta_ic * L_ic
        return L_total, L_phys, L_data, L_ic

    @tf.function
    def train_step(self, t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0):
        with tf.GradientTape() as tape:
            L_total, L_phys, L_data, L_ic = self.total_loss(
                t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0
            )

        param_vars = [self.theta1, self.p2, self.theta3, self.p5, self.p7, self.p9]
        ic_vars = [self.w0_var]
        trainable_vars = list(self.nn_vars) + param_vars + ic_vars

        grads = tape.gradient(L_total, trainable_vars)

        if any(g is None for g in grads) and not self._warned_none_grads:
            missing = [v.name for g, v in zip(grads, trainable_vars) if g is None]
            tf.print("WARNING: None gradients for:", missing)
            self._warned_none_grads = True

        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)

        return L_total, L_phys, L_data, L_ic

    def _snapshot_state(self):
        return {
            "weights": [v.numpy().copy() for v in self.weights],
            "biases": [v.numpy().copy() for v in self.biases],
            "theta1": float(self.theta1.numpy()),
            "p2": float(self.p2.numpy()),
            "theta3": float(self.theta3.numpy()),
            "p5": float(self.p5.numpy()),
            "p7": float(self.p7.numpy()),
            "p9": float(self.p9.numpy()),
            "w0": float(self.w0_var.numpy()),
        }

    def _restore_state(self, st):
        for v, arr in zip(self.weights, st["weights"]):
            v.assign(arr)
        for v, arr in zip(self.biases, st["biases"]):
            v.assign(arr)
        self.theta1.assign(st["theta1"])
        self.p2.assign(st["p2"])
        self.theta3.assign(st["theta3"])
        self.p5.assign(st["p5"])
        self.p7.assign(st["p7"])
        self.p9.assign(st["p9"])
        self.w0_var.assign(st["w0"])

    def generate_data(self, t_data, w_data, x_data, y_data, z_data, n_physics, W0_value):
        t_data = np.asarray(t_data, dtype=np.float32).reshape(-1, 1)
        w_data = np.asarray(w_data, dtype=np.float32).reshape(-1, 1)
        x_data = np.asarray(x_data, dtype=np.float32).reshape(-1, 1)
        y_data = np.asarray(y_data, dtype=np.float32).reshape(-1, 1)
        z_data = np.asarray(z_data, dtype=np.float32).reshape(-1, 1)

        if t_data.shape[0] == 0:
            raise ValueError("Empty t_data.")
        if any(a.shape != t_data.shape for a in (w_data, x_data, y_data, z_data)):
            raise ValueError("t,w,x,y,z must have same length.")

        idx = np.argsort(t_data[:, 0])
        t_data, w_data, x_data, y_data, z_data = t_data[idx], w_data[idx], x_data[idx], y_data[idx], z_data[idx]

        t_min = float(t_data[0, 0])
        t_max = float(t_data[-1, 0])
        n_physics = int(n_physics)
        if n_physics < 2:
            raise ValueError("n_physics must be >= 2.")
        t_physics = np.linspace(t_min, t_max, n_physics, dtype=np.float32).reshape(-1, 1)

        t0 = t_data[0:1, :]
        W0 = np.array([[float(W0_value)]], dtype=np.float32)

        if not self._ic_initialized:
            self.w0_var.assign(float(w_data[0, 0]))
            self.x0_fixed = float(x_data[0, 0])
            self.y0_fixed = float(y_data[0, 0])
            self.z0_fixed = float(z_data[0, 0])

            self.x0_tf = tf.constant([[self.x0_fixed]], dtype=tf.float32)
            self.y0_tf = tf.constant([[self.y0_fixed]], dtype=tf.float32)
            self.z0_tf = tf.constant([[self.z0_fixed]], dtype=tf.float32)
            self._ic_initialized = True

        return t_physics, t_data, w_data, x_data, y_data, z_data, t0, W0

    def train(
        self,
        t_data, w_data, x_data, y_data, z_data,
        epochs=100001,
        verbose_every=None,
        n_physics=600,
        W0_value=0.0,
        early_stop=True,
        window=500,
        patience=1000,
        min_rel_improve=1e-2,
    ):
        """
        early_stop:        Enable early stopping based on the moving-average training loss
        window:            Number of epochs used to compute the moving-average loss
        patience:          Epochs allowed without a new best moving-average loss before stopping
        min_rel_improve:   Relative-improvement threshold used by the early-stopping criterion
        """

        t_physics, t_data, w_data, x_data, y_data, z_data, t0, W0 = \
            self.generate_data(t_data, w_data, x_data, y_data, z_data, n_physics, W0_value)

        t_physics = tf.convert_to_tensor(t_physics, dtype=tf.float32)
        t_data = tf.convert_to_tensor(t_data, dtype=tf.float32)
        w_data = tf.convert_to_tensor(w_data, dtype=tf.float32)
        x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
        y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)
        z_data = tf.convert_to_tensor(z_data, dtype=tf.float32)
        t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
        W0 = tf.convert_to_tensor(W0, dtype=tf.float32)

        eps = tf.constant(1e-12, dtype=tf.float32)
        wS = tf.maximum(tf.reduce_max(tf.abs(w_data)), eps)
        xS = tf.maximum(tf.reduce_max(tf.abs(x_data)), eps)
        yS = tf.maximum(tf.reduce_max(tf.abs(y_data)), eps)
        zS = tf.maximum(tf.reduce_max(tf.abs(z_data)), eps)
        t_span = tf.maximum(tf.reduce_max(t_data) - tf.reduce_min(t_data), eps)
        WS = tf.maximum(wS * t_span, eps)
        self.scales = (wS, WS, xS, yS, zS)

        print(f"Loss scales | wS={float(wS.numpy()):.3e}, WS={float(WS.numpy()):.3e}, "
              f"xS={float(xS.numpy()):.3e}, yS={float(yS.numpy()):.3e}, zS={float(zS.numpy()):.3e}")
        print(f"Training 5-ODE PINN (MLP {self.arch}, activation={self.act_name})")
        print(f"fixed p4={float(self.p4.numpy()):.8e}, fixed p6={float(self.p6.numpy()):.8e}, " 
              f"fixed p8={float(self.p8.numpy()):.8e}, fixed p10={float(self.p10.numpy()):.8e}")
        print("-" * 80)

        if int(t_data.shape[0]) < 2:
            raise ValueError("Need at least 2 samples to use t_fit=t_data[1:].")

        t_fit = t_data[1:, :]
        w_fit = w_data[1:, :]
        x_fit = x_data[1:, :]
        y_fit = y_data[1:, :]
        z_fit = z_data[1:, :]

        best_ma = float("inf")
        best_ep = 0
        stale = 0
        ma_buffer = []
        best_state = None

        for ep in range(int(epochs)):
            L_total, L_phys, L_data, L_ic = self.train_step(
                t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0
            )
            L_total_f = float(L_total.numpy())
            L_phys_f = float(L_phys.numpy())
            L_data_f = float(L_data.numpy())
            L_ic_f = float(L_ic.numpy())

            self.loss_history.append(L_total_f)
            self.loss_phys_hist.append(L_phys_f)
            self.loss_data_hist.append(L_data_f)
            self.loss_ic_hist.append(L_ic_f)

            if early_stop:
                ma_buffer.append(L_total_f)
                if len(ma_buffer) > window:
                    ma_buffer.pop(0)

                if len(ma_buffer) == window:
                    ma = float(sum(ma_buffer) / window)
                    rel_improve = (best_ma - ma) / max(abs(best_ma), 1e-12)

                    if ma < best_ma:
                        best_ma = ma
                        best_ep = ep
                        stale = 0
                        best_state = self._snapshot_state()
                    else:
                        stale = stale + 1 if rel_improve < min_rel_improve else 0

                    if stale >= patience:
                        print(f"Early stop at epoch {ep} | best_ma={best_ma:.3e} at epoch {best_ep}")
                        break

        if best_state is not None:
            self._restore_state(best_state)
            print(f"RESTORED best state from epoch {best_ep} | best_ma={best_ma:.3e}")

    def predict(self, t, extend_factor=2.0, dt=None):
        """
        If extend_factor > 1, interprets `t` as observed time samples and builds
        a uniform grid from min(t) to extend_factor*max(t).
        If extend_factor == 1, behaves like the original: predicts at the given `t`.
        """
        t = np.asarray(t, dtype=float).ravel()
    
        if extend_factor is not None and extend_factor > 1.0:
            if dt is None:
                raise ValueError("When extend_factor>1, you must provide dt.")
            t0 = float(np.min(t))
            tend = float(np.max(t))
            t_end = float(extend_factor * tend)
            n = max(2, int(np.round((t_end - t0) / dt)) + 1)
            t = np.linspace(t0, t_end, n)
    
        t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        pred = self.neural_net(tf.convert_to_tensor(t, dtype=tf.float32)).numpy()
        return t, pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4], pred[:, 4:5]

    def get_params(self):
        return (
            float(self.p1_value().numpy()),
            float(self.p2.numpy()),
            float(self.p3_value().numpy()),
            float(self.p4.numpy()),
            float(self.p5.numpy()),
            float(self.p6.numpy()),
            float(self.p7.numpy()),
            float(self.p8.numpy()),
            float(self.p9.numpy()),
            float(self.p10.numpy()),
        )

    def euler_solve(self, t_grid, w0, W0, x0, y0, z0, params=None):
        if params is None:
            params = self.get_params()
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
        return euler_ODE(t_grid, float(w0), float(W0), float(x0), float(y0), float(z0),
                        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)

# =============================================================================
# Batch runner - MULTIPLE SEEDS
# Save/simulate ONLY the best seed
# =============================================================================

def run_all_sheets(
    excel_path,
    output_dir,
    sigma=1.0,
    normalize=True,
    epochs=100001,
    verbose_every=None,
    n_physics=600,
    W0_value=0.0,
    dt=0.1,
    extend_factor=2.0,
    alpha=0.999,
    beta_ic=0.5,
    lr=1e-4,
    arch=(1, 128, 128, 128, 5),
    activation="tanh",
    n_seeds=100,
    base_seed=130425,
):
    """
    Multiple-seed PINN analysis.

    For each experiment:
        1. Train all seeds.
        2. Calculate statistics for every seed.
        3. Save one Excel summary containing all seed results.
        4. Select the best seed using minimum RSS.
        5. Restore the exact best trained PINN.
        6. Save predictions, Euler simulation, statistics table and PDF
           ONLY for the best seed.

    The same seed sequence is used for every experiment:
        seed = base_seed + run - 1
    """

    n_seeds = int(n_seeds)
    base_seed = int(base_seed)

    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1.")

    t0_total = time.perf_counter()

    os.makedirs(output_dir, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    sheet_names = sorted(xls.sheet_names, key=natural_key)

    sheet_re = re.compile(r"^E\d+$", re.IGNORECASE)
    required_cols = ["t", "w", "x", "y", "z"]

    # Global summaries
    summary_all_rows = []
    summary_best_rows = []

    # =========================================================================
    # EXPERIMENT LOOP
    # =========================================================================

    for sheet in sheet_names:

        sheet_str = str(sheet).strip()

        if not sheet_re.match(sheet_str):
            continue

        t0_sheet = time.perf_counter()

        print(
            f"\n{'=' * 80}\n"
            f"Experiment: {sheet_str}\n"
            f"{'=' * 80}"
        )

        # ---------------------------------------------------------------------
        # Read experimental data
        # ---------------------------------------------------------------------

        df = xls.parse(sheet)

        missing = [
            c for c in required_cols
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Sheet {sheet_str} missing columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        t_data = df["t"].to_numpy(dtype=float)

        if t_data.size < 2:
            raise ValueError(
                f"Sheet {sheet_str}: need at least 2 time points."
            )

        # ---------------------------------------------------------------------
        # Preprocess experimental data ONCE
        # ---------------------------------------------------------------------

        w_data = preprocess_series(
            df["w"].to_numpy(dtype=float),
            sigma,
            normalize
        )

        x_data = preprocess_series(
            df["x"].to_numpy(dtype=float),
            sigma,
            normalize
        )

        y_data = preprocess_series(
            df["y"].to_numpy(dtype=float),
            sigma,
            normalize
        )

        z_data = preprocess_series(
            df["z"].to_numpy(dtype=float),
            sigma,
            normalize
        )

        # Save preprocessed experimental data once
        preprocess_series_csv_path = os.path.join(
            output_dir,
            f"{sheet_str}_preprocess_series.csv"
        )

        df_preprocess_series = pd.DataFrame({
            "t": np.asarray(t_data).ravel(),
            "w": np.asarray(w_data).ravel(),
            "x": np.asarray(x_data).ravel(),
            "y": np.asarray(y_data).ravel(),
            "z": np.asarray(z_data).ravel(),
        })

        df_preprocess_series.to_csv(
            preprocess_series_csv_path,
            index=False
        )

        # ---------------------------------------------------------------------
        # Time information
        # ---------------------------------------------------------------------

        tmin = float(np.min(t_data))
        tmax = float(np.max(t_data))
        t_end = extend_factor * tmax

        # ---------------------------------------------------------------------
        # Statistics Euler grid
        #
        # This remains only on the OBSERVED interval.
        # ---------------------------------------------------------------------

        n_grid_stats = max(
            2,
            int(
                np.round(
                    (tmax - tmin) / (dt / 10.0)
                )
            ) + 1
        )

        t_grid_stats = np.linspace(
            tmin,
            tmax,
            n_grid_stats
        ).astype(float)

        # Experimental observations packed once
        f_obs = np.asarray(
            [
                w_data,
                x_data,
                y_data,
                z_data
            ],
            dtype=np.float32
        ).reshape(-1, 1)

        # ---------------------------------------------------------------------
        # Storage for this experiment
        # ---------------------------------------------------------------------

        sheet_rows = []

        best_state = None
        best_stats = None
        best_params = None
        best_row = None

        best_run = None
        best_seed = None
        best_rss = np.nan

        best_w0 = None
        best_x0 = None
        best_y0 = None
        best_z0 = None

        # =====================================================================
        # SEED LOOP
        # =====================================================================

        for run in range(1, n_seeds + 1):

            # Same seed sequence for EVERY experiment
            seed = int(
                base_seed + (run - 1)
            )

            t0_run = time.perf_counter()

            # -------------------------------------------------------------
            # Seed initialization
            # -------------------------------------------------------------

            np.random.seed(seed)
            tf.random.set_seed(seed)

            model = ODEPINN(
                alpha=alpha,
                beta_ic=beta_ic,
                lr=lr,
                arch=arch,
                activation=activation,
                seed=seed,
            )

            # -------------------------------------------------------------
            # Train silently
            #
            # We suppress the detailed output produced inside model.train().
            # The PINN algorithm itself is NOT modified.
            # -------------------------------------------------------------

            print(
                f"Training {sheet_str} | "
                f"run {run}/{n_seeds} | "
                f"seed={seed} ..."
            )

            with contextlib.redirect_stdout(io.StringIO()):

                model.train(
                    t_data,
                    w_data,
                    x_data,
                    y_data,
                    z_data,
                    epochs=epochs,
                    verbose_every=None,
                    n_physics=n_physics,
                    W0_value=W0_value,
                )
                
            # Print PINN configuration only once per experiment
            if run == 1:
            
                wS, WS, xS, yS, zS = model.scales
            
                print(
                    f"Loss scales | "
                    f"wS={float(wS.numpy()):.3e}, "
                    f"WS={float(WS.numpy()):.3e}, "
                    f"xS={float(xS.numpy()):.3e}, "
                    f"yS={float(yS.numpy()):.3e}, "
                    f"zS={float(zS.numpy()):.3e}"
                )
            
                print(
                    f"Training 5-ODE PINN "
                    f"(MLP {model.arch}, activation={model.act_name})"
                )
            
                print(
                    f"fixed p4={float(model.p4.numpy()):.8e}, "
                    f"fixed p6={float(model.p6.numpy()):.8e}, "
                    f"fixed p8={float(model.p8.numpy()):.8e}, "
                    f"fixed p10={float(model.p10.numpy()):.8e}"
                )
            
                print("-" * 80)

            # -------------------------------------------------------------
            # Parameters
            # -------------------------------------------------------------

            params = np.asarray(
                model.get_params(),
                dtype=float
            )

            # -------------------------------------------------------------
            # Initial conditions
            # -------------------------------------------------------------

            w0_hat = float(
                model.w0_var.numpy()
            )

            x0_hat = float(
                model.x0_fixed
            )

            y0_hat = float(
                model.y0_fixed
            )

            z0_hat = float(
                model.z0_fixed
            )

            # -------------------------------------------------------------
            # Statistics for this seed
            #
            # This uses the observed experimental horizon only.
            # -------------------------------------------------------------

            stats = stats_at_optimum(
                t_data,
                t_grid_stats,
                f_obs,
                P=params,
                w0=w0_hat,
                W0=W0_value,
                x0=x0_hat,
                y0=y0_hat,
                z0=z0_hat,
                eps_base=1e-6,
            )

            elapsed_run = (
                time.perf_counter()
                - t0_run
            )

            # -------------------------------------------------------------
            # Summary row
            #
            # Original column positions are preserved.
            # -------------------------------------------------------------

            row = {
                "sheet": sheet_str,
                "elapsed_sec": elapsed_run,

                "R2": stats.get(
                    "R2",
                    np.nan
                ),

                "R2_adj": stats.get(
                    "R2_adj",
                    np.nan
                ),

                "RSS": stats.get(
                    "RSS",
                    np.nan
                ),

                "AIC": stats.get(
                    "AIC",
                    np.nan
                ),

                "cond_JTJ": stats.get(
                    "cond_JTJ",
                    np.nan
                ),

                "used_pinv": stats.get(
                    "used_pinv",
                    False
                ),

                "w0": w0_hat,
                "x0": x0_hat,
                "y0": y0_hat,
                "z0": z0_hat,
            }

            # p1 ... p10
            for i, pv in enumerate(
                params,
                start=1
            ):
                row[f"p{i}"] = float(pv)

            # Multi-seed information at the END
            row["run"] = int(run)
            row["seed"] = int(seed)

            sheet_rows.append(row)
            summary_all_rows.append(row.copy())

            # -------------------------------------------------------------
            # Check whether this is the best run
            #
            # Selection criterion:
            #
            #              minimum RSS
            # -------------------------------------------------------------

            current_rss = float(
                stats.get(
                    "RSS",
                    np.nan
                )
            )

            is_better = False

            if best_state is None:
                # Always retain the first completed model as fallback
                is_better = True

            elif (
                np.isfinite(current_rss)
                and (
                    not np.isfinite(best_rss)
                    or current_rss < best_rss
                )
            ):
                is_better = True

            if is_better:

                best_rss = current_rss
                best_run = int(run)
                best_seed = int(seed)

                # Exact NN + parameter state
                best_state = model._snapshot_state()

                # Statistics
                best_stats = copy.deepcopy(stats)

                # Parameters
                best_params = params.copy()

                # ICs
                best_w0 = w0_hat
                best_x0 = x0_hat
                best_y0 = y0_hat
                best_z0 = z0_hat

                # Summary row
                best_row = row.copy()

            # -------------------------------------------------------------
            # Clean model before next seed
            # -------------------------------------------------------------

            del model

            tf.keras.backend.clear_session()

            gc.collect()

        # =====================================================================
        # END SEED LOOP
        # =====================================================================

        # ---------------------------------------------------------------------
        # Summary containing ALL seeds for this experiment
        # ---------------------------------------------------------------------

        sheet_summary_df = pd.DataFrame(
            sheet_rows
        )

        # Mark best seed
        sheet_summary_df["best_seed"] = (
            sheet_summary_df["run"]
            == best_run
        )

        # ---------------------------------------------------------------------
        # Save Excel summary for ALL seeds
        # ---------------------------------------------------------------------

        sheet_summary_excel = os.path.join(
            output_dir,
            f"{sheet_str}_summary_all_seeds.xlsx"
        )

        sheet_summary_df.to_excel(
            sheet_summary_excel,
            index=False
        )

        # ---------------------------------------------------------------------
        # Print ONLY the experiment summary
        # ---------------------------------------------------------------------

        print(
            sheet_summary_df.to_string(
                index=False
            )
        )

        print(
            f"\nBest {sheet_str}: "
            f"run={best_run}, "
            f"seed={best_seed}, "
            f"RSS={best_rss:.6e}"
        )

        # =====================================================================
        # RESTORE THE EXACT BEST PINN
        # =====================================================================

        best_model = ODEPINN(
            alpha=alpha,
            beta_ic=beta_ic,
            lr=lr,
            arch=arch,
            activation=activation,
            seed=best_seed,
        )

        best_model._restore_state(
            best_state
        )

        # =====================================================================
        # PINN PREDICTION - BEST SEED ONLY
        # =====================================================================

        (
            t_plot,
            w_p,
            W_p,
            x_p,
            y_p,
            z_p
        ) = best_model.predict(
            t_data,
            extend_factor=extend_factor,
            dt=dt
        )

        pred_csv_path = os.path.join(
            output_dir,
            f"{sheet_str}_PINN_prediction.csv"
        )

        df_pred = pd.DataFrame({
            "t": np.asarray(t_plot).ravel(),
            "w": np.asarray(w_p).ravel(),
            "W": np.asarray(W_p).ravel(),
            "x": np.asarray(x_p).ravel(),
            "y": np.asarray(y_p).ravel(),
            "z": np.asarray(z_p).ravel(),
        })

        df_pred.to_csv(
            pred_csv_path,
            index=False
        )

        # =====================================================================
        # EXTENDED EULER SIMULATION - BEST SEED ONLY
        # =====================================================================

        n_grid = max(
            2,
            int(
                np.round(
                    (t_end - tmin)
                    / (dt / 10.0)
                )
            ) + 1
        )

        t_grid = np.linspace(
            tmin,
            t_end,
            n_grid
        ).astype(float)

        (
            w_e,
            x_e,
            y_e,
            z_e
        ) = best_model.euler_solve(
            t_grid,
            best_w0,
            W0_value,
            best_x0,
            best_y0,
            best_z0,
            params=best_params,
        )

        # =====================================================================
        # STATISTICS TABLE - BEST SEED ONLY
        # =====================================================================

        df_params = stats_to_table(
            best_stats
        )

        df_ic_excel = pd.DataFrame({
            "param": [
                "w0 (estimated)",
                "x0 (fixed)",
                "y0 (fixed)",
                "z0 (fixed)",
                "W0 (fixed)"
            ],

            "estimate": [
                best_w0,
                best_x0,
                best_y0,
                best_z0,
                W0_value
            ],

            "SE": [np.nan] * 5,
            "MOE_95": [np.nan] * 5,
            "CI_low": [np.nan] * 5,
            "CI_high": [np.nan] * 5,
            "p_value": [np.nan] * 5,
        })

        df_stats_csv = pd.concat(
            [
                df_params,
                df_ic_excel
            ],
            ignore_index=True
        )

        csv_table_path = os.path.join(
            output_dir,
            f"{sheet_str}_stats_table.csv"
        )

        df_stats_csv.to_csv(
            csv_table_path,
            index=False
        )

        # =====================================================================
        # PDF REPORT - BEST SEED ONLY
        # =====================================================================

        pdf_path = os.path.join(
            output_dir,
            f"{sheet_str}_PINN_report.pdf"
        )

        ic_dict = {
            "w0 (estimated)": best_w0,
            "x0 (fixed)": best_x0,
            "y0 (fixed)": best_y0,
            "z0 (fixed)": best_z0,
            "W0 (fixed)": float(W0_value),
        }

        with PdfPages(pdf_path) as pdf:

            fig1 = plotsys(
                t_data,
                w_data,
                x_data,
                y_data,
                z_data,

                t_plot,
                w_p,
                x_p,
                y_p,
                z_p,

                t_grid,
                w_e,
                x_e,
                y_e,
                z_e,

                title=(
                    f"{sheet_str} | "
                    f"Best run={best_run} | "
                    f"seed={best_seed} | "
                    f"PINN vs Euler"
                ),
            )

            pdf.savefig(fig1)
            plt.close(fig1)

            fig2 = make_table_page(
                best_stats,
                df_params,

                (
                    f"{sheet_str} | "
                    f"Best run={best_run} | "
                    f"seed={best_seed}"
                ),

                ic_estimates=ic_dict,
            )

            pdf.savefig(fig2)
            plt.close(fig2)

        # ---------------------------------------------------------------------
        # Save best-seed information
        # ---------------------------------------------------------------------

        best_seed_excel = os.path.join(
            output_dir,
            f"{sheet_str}_best_seed.xlsx"
        )

        pd.DataFrame(
            [best_row]
        ).to_excel(
            best_seed_excel,
            index=False
        )

        summary_best_rows.append(
            best_row.copy()
        )

        # ---------------------------------------------------------------------
        # Experiment runtime
        # ---------------------------------------------------------------------

        elapsed_sheet = (
            time.perf_counter()
            - t0_sheet
        )

        print(
            f"\nSaved best-seed outputs for {sheet_str}"
        )

        print(
            f"Experiment elapsed time: "
            f"{elapsed_sheet:.2f} s "
            f"({elapsed_sheet/60.0:.2f} min)"
        )

        # Clean best model
        del best_model

        tf.keras.backend.clear_session()

        gc.collect()

    # =========================================================================
    # GLOBAL SUMMARIES
    # =========================================================================

    all_summary_df = pd.DataFrame(
        summary_all_rows
    )

    best_summary_df = pd.DataFrame(
        summary_best_rows
    )

    # -------------------------------------------------------------------------
    # All seed results
    # -------------------------------------------------------------------------

    all_summary_excel = os.path.join(
        output_dir,
        "summary_pinn_all_seeds.xlsx"
    )

    all_summary_df.to_excel(
        all_summary_excel,
        index=False
    )

    # -------------------------------------------------------------------------
    # Best seed results
    # -------------------------------------------------------------------------

    best_summary_excel = os.path.join(
        output_dir,
        "summary_pinn_best_seed.xlsx"
    )

    best_summary_df.to_excel(
        best_summary_excel,
        index=False
    )

    # -------------------------------------------------------------------------
    # Legacy summary_pinn.csv
    #
    # Preserve original column structure for downstream analysis.
    # -------------------------------------------------------------------------

    legacy_cols = [
        "sheet",
        "elapsed_sec",
        "R2",
        "R2_adj",
        "RSS",
        "AIC",
        "cond_JTJ",
        "used_pinv",
        "w0",
        "x0",
        "y0",
        "z0",
    ] + [
        f"p{i}"
        for i in range(1, 11)
    ]

    legacy_summary_df = (
        best_summary_df[
            legacy_cols
        ].copy()
    )

    summary_csv = os.path.join(
        output_dir,
        "summary_pinn.csv"
    )

    legacy_summary_df.to_csv(
        summary_csv,
        index=False
    )

    # =========================================================================
    # TOTAL RUNTIME
    # =========================================================================

    elapsed_total = (
        time.perf_counter()
        - t0_total
    )

    print(
        f"\n{'=' * 80}\n"
        f"BEST PINN RESULT FOR EACH EXPERIMENT\n"
        f"{'=' * 80}"
    )

    print(
        best_summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nTOTAL elapsed time: "
        f"{elapsed_total:.2f} s "
        f"({elapsed_total/60.0:.2f} min)"
    )

    runtime_path = os.path.join(
        output_dir,
        "runtime_total.txt"
    )

    with open(
        runtime_path,
        "w"
    ) as f:

        f.write(
            f"n_seeds: {n_seeds}\n"
        )

        f.write(
            f"base_seed: {base_seed}\n"
        )

        f.write(
            f"completed_runs: "
            f"{len(all_summary_df)}\n"
        )

        f.write(
            f"TOTAL elapsed time: "
            f"{elapsed_total:.6f} seconds\n"
        )

        f.write(
            f"TOTAL elapsed time: "
            f"{elapsed_total/60.0:.6f} minutes\n"
        )

if __name__ == "__main__":
    run_all_sheets(
        excel_path="data_matlab_smooth.xlsx", # Data smoothed previously in MATLAB for consistency
        output_dir="results_PINN_100seeds E9-E10",
        sigma = 0.0,                     # Gaussian smoothing strength; 0 disables smoothing
        normalize = False,               # If True, scale each state by its maximum value
        epochs = 100001,                 # Maximum number of PINN training epochs
        n_physics = 600,                 # Number of physics collocation points
        W0_value = 0.0,                  # Initial cumulative biomass, W(0)=0
        dt = 0.01,                       # Time step used for prediction/Euler simulation grids
        extend_factor = 2.0,             # Extend simulation horizon to 2× the observed final time
        alpha = 0.999,                   # Weight assigned to the physics-loss contribution
        beta_ic = 0.5,                   # Weight assigned to the initial-condition loss
        lr = 1e-4,                       # Adam optimizer learning rate
        arch=(1, 128, 128, 128, 5),    # Neural-network architecture: 1 input, 3 hidden layers, 5 outputs
        activation = "tanh",             # Activation function used in the hidden layers
        n_seeds = 100,                     # Number of independent random-seed training runs per experiment
        base_seed = 130425,              # First random seed; subsequent runs use base_seed + run - 1
    )