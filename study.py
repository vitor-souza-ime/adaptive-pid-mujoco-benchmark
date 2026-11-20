"""Paired, physically consistent MuJoCo study of fixed and adaptive PID control.

Run: python study.py --seeds 20 --duration 16 --output results
All scientific metrics and plots are derived from the generated raw trajectories.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

from model import robot_xml


CONTROLLERS = ("fixed_no_gravity", "fixed_gravity", "adaptive_no_gravity", "adaptive_gravity")
SCENARIOS = ("nominal", "fast", "payload", "disturbance", "combined")
GEAR = np.array([20.0, 20.0, 10.0])
TORQUE_LIMIT = np.array([6.0, 6.0, 3.0])  # joint-space N m, NOT motor ctrl units
KP0 = np.array([12.0, 14.0, 8.0])
KI0 = np.array([0.6, 0.6, 0.35])
KD0 = np.array([2.0, 2.3, 1.2])
JOINT_NAMES = ("J1", "J2", "J3")


@dataclass(frozen=True)
class Scenario:
    name: str
    mass: float
    speed: float
    noise_pos: float
    noise_vel: float
    pulse: bool


def scenario_spec(name: str) -> Scenario:
    specs = {
        "nominal": Scenario(name, 0.1, 1.0, 0.0, 0.0, False),
        "fast": Scenario(name, 0.1, 2.5, 0.0, 0.0, False),
        "payload": Scenario(name, 0.4, 1.0, 0.0, 0.0, False),
        "disturbance": Scenario(name, 0.1, 1.0, 0.0, 0.0, True),
        "combined": Scenario(name, 0.4, 2.5, 0.003, 0.03, True),
    }
    return specs[name]


def reference(t: float, speed: float) -> tuple[np.ndarray, np.ndarray]:
    w = speed * np.array([0.2, 0.25, 0.3])
    amplitude = np.array([0.2, 0.15, 0.1])
    phase = np.array([0.0, np.pi / 4, np.pi / 2])
    q = amplitude * np.sin(w * t + phase)
    dq = amplitude * w * np.cos(w * t + phase)
    return q, dq


def desired_ee_xyz(q: np.ndarray) -> np.ndarray:
    """Forward kinematics for the original yaw-pitch-pitch model, meters."""
    reach = 0.3 * np.cos(q[1]) + 0.3 * np.cos(q[1] + q[2])
    return np.array([reach * np.cos(q[0]), reach * np.sin(q[0]),
                     0.1 - 0.3 * np.sin(q[1]) - 0.3 * np.sin(q[1] + q[2])])


class JointPID:
    """Both controllers share identical PID, gravity, saturation and anti-windup.

    Only the bounded gains update differs between treatment and fixed control.
    The update is a heuristic, not a Lyapunov stability guarantee.
    """

    def __init__(self, model: mujoco.MjModel, adaptive: bool, gravity: bool):
        self.model = model
        self.adaptive = adaptive
        self.gravity = gravity
        self.kp = KP0.copy()
        self.ki = KI0.copy()
        self.kd = KD0.copy()
        self.integral = np.zeros(3)
        self.history = deque(maxlen=100)
        self.events = 0
        # The controller is calibrated to the *nominal* 0.1 kg end mass.
        # In the payload tests, the actual plant has 0.4 kg; the controller
        # receives no privileged knowledge of this change.
        self.gravity_model = mujoco.MjModel.from_xml_string(robot_xml(0.1))
        self.gravity_data = mujoco.MjData(self.gravity_model)

    def gravity_torques(self, q_measured: np.ndarray) -> np.ndarray:
        if not self.gravity:
            return np.zeros(3)
        gdata = self.gravity_data
        gdata.qpos[:3] = q_measured
        gdata.qvel[:3] = 0.0
        gdata.qacc[:3] = 0.0
        mujoco.mj_forward(self.gravity_model, gdata)
        return gdata.qfrc_bias[:3].copy()

    def command(self, q: np.ndarray, dq: np.ndarray, ref: np.ndarray,
                dref: np.ndarray, step: int, dt: float) -> np.ndarray:
        error = ref - q
        vel_error = dref - dq
        gravity = self.gravity_torques(q)
        # Conditional integration suppresses wind-up during torque saturation.
        integral_candidate = np.clip(self.integral + error * dt, -0.35, 0.35)
        candidate = (self.kp * error + self.ki * integral_candidate
                     + self.kd * vel_error + gravity)
        integrate = ~((candidate > TORQUE_LIMIT) & (error > 0) |
                      (candidate < -TORQUE_LIMIT) & (error < 0))
        self.integral[integrate] = integral_candidate[integrate]
        unsaturated = self.kp * error + self.ki * self.integral + self.kd * vel_error + gravity
        torque = np.clip(unsaturated, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.history.append(error.copy())
        if self.adaptive and (step + 1) % 50 == 0 and len(self.history) == 100:
            self.adapt()
        return torque

    def adapt(self) -> None:
        e = np.asarray(self.history)
        old = np.sqrt(np.mean(e[:50] ** 2, axis=0))
        recent = np.sqrt(np.mean(e[50:] ** 2, axis=0))
        threshold = np.array([0.015, 0.015, 0.015])
        # Adapt only if RMS exceeds a declared tracking target and is not
        # already declining by at least 5% across successive windows.
        not_improving = recent >= 0.95 * np.maximum(old, 1e-9)
        factor = np.clip((recent - threshold) / threshold, 0.0, 3.0)
        increase = (recent > threshold) & not_improving
        self.kp[increase] *= 1.0 + 0.012 * factor[increase]
        # Detect a sustained error sign reversal pattern; increase damping.
        zero_crossings = np.sum(e[1:] * e[:-1] < 0, axis=0)
        oscillatory = (zero_crossings >= 8) & (recent > threshold)
        self.kd[oscillatory] *= 1.015
        self.kp = np.clip(self.kp, KP0, 2.5 * KP0)
        self.kd = np.clip(self.kd, KD0, 2.0 * KD0)
        self.events += 1


def disturbance_torque(t: float, enabled: bool, seed: int) -> np.ndarray:
    if not enabled:
        return np.zeros(3)
    # Identical disturbance realization for all controllers with a given seed.
    scale = 1.0 + 0.03 * (seed % 7 - 3)
    tau = np.zeros(3)
    if 5.0 <= t < 7.0:
        tau[1] = -1.4 * scale
    if 9.0 <= t < 11.0:
        tau[2] = 0.75 * scale
    return tau


def simulate(name: str, controller_name: str, seed: int, duration: float = 16.0,
             store_trace: bool = False):
    s = scenario_spec(name)
    model = mujoco.MjModel.from_xml_string(robot_xml(s.mass))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Paired initial conditions are deterministic for (scenario, seed).
    initial_rng = np.random.default_rng(1000 + seed)
    data.qpos[:3] = initial_rng.uniform(-0.055, 0.055, size=3)
    mujoco.mj_forward(model, data)
    noise_rng = np.random.default_rng(200000 + seed)
    adaptive = controller_name.startswith("adaptive")
    gravity = controller_name.endswith("gravity") and "no_gravity" not in controller_name
    pid = JointPID(model, adaptive, gravity)
    dt = float(model.opt.timestep)
    total_steps = int(round(duration / dt))
    rec = {k: [] for k in ("time", "q", "ref", "joint_error", "ee_error",
                             "torque", "kp", "kd", "limit_hit")}
    limit_violations = 0
    saturation_steps = 0
    full_errors = []
    sum_torque_sq = 0.0
    sum_ee_error_sq = 0.0
    for step in range(total_steps):
        t = step * dt
        qref, dqref = reference(t, s.speed)
        noise_q = noise_rng.normal(0, s.noise_pos, size=3)
        noise_dq = noise_rng.normal(0, s.noise_vel, size=3)
        qmeas = data.qpos[:3].copy() + noise_q
        dqmeas = data.qvel[:3].copy() + noise_dq
        tau = pid.command(qmeas, dqmeas, qref, dqref, step, dt)
        saturation_steps += int(np.any(np.abs(tau) >= TORQUE_LIMIT - 1e-10))
        sum_torque_sq += float(np.dot(tau, tau))
        # MuJoCo motors apply joint force = gear * ctrl; convert N m to ctrl.
        data.ctrl[:] = tau / GEAR
        data.qfrc_applied[:3] = disturbance_torque(t, s.pulse, seed)
        mujoco.mj_step(model, data)
        q = data.qpos[:3].copy()
        # The state is now at t + dt, so evaluate errors at that same instant.
        qref_after, _ = reference(t + dt, s.speed)
        err = qref_after - q
        # mj_step integrates qpos but leaves site_xpos at the prior forward
        # pass until the next mj_forward. Evaluate both cartesian positions
        # at the *same*, newly integrated time using validated kinematics.
        ee_actual = desired_ee_xyz(q)
        ee_err = np.linalg.norm(ee_actual - desired_ee_xyz(qref_after))
        full_errors.append(err)
        sum_ee_error_sq += float(ee_err ** 2)
        near_limit = bool(np.any(np.abs(q) > np.array([np.pi, np.pi / 2, np.pi / 2]) * 0.99))
        limit_violations += int(near_limit)
        if step % 10 == 0:
            rec["time"].append(t + dt)
            rec["q"].append(q)
            rec["ref"].append(qref_after)
            rec["joint_error"].append(err)
            rec["ee_error"].append(ee_err)
            rec["torque"].append(tau.copy())
            rec["kp"].append(pid.kp.copy())
            rec["kd"].append(pid.kd.copy())
            rec["limit_hit"].append(float(np.any(np.abs(tau) >= TORQUE_LIMIT - 1e-10)))
    record = {k: np.asarray(v) for k, v in rec.items()}
    e = np.asarray(full_errors)
    row = {
        "scenario": name, "controller": controller_name, "seed": seed,
        "joint_rmse": float(np.sqrt(np.mean(e ** 2))),
        "joint_mae": float(np.mean(np.abs(e))),
        "joint_p95": float(np.quantile(np.abs(e), 0.95)),
        "joint_max_abs": float(np.max(np.abs(e))),
        "ee_rmse_m": float(np.sqrt(sum_ee_error_sq / total_steps)),
        "effort_rms_Nm": float(np.sqrt(sum_torque_sq / (3 * total_steps))),
        "effort_integral_Nm2s": float(sum_torque_sq * dt),
        "saturation_fraction": float(saturation_steps / total_steps),
        "joint_limit_fraction": float(limit_violations / total_steps),
        "kp_relative_change": float(np.mean(pid.kp / KP0 - 1.0)),
        "kd_relative_change": float(np.mean(pid.kd / KD0 - 1.0)),
        "adaptation_events": int(pid.events),
        "finite": int(np.isfinite(e).all() and np.isfinite(record["torque"]).all()),
    }
    for j in range(3):
        row[f"rmse_{JOINT_NAMES[j]}"] = float(np.sqrt(np.mean(e[:, j] ** 2)))
    return row, record if store_trace else None


def paired_summary(metrics: pd.DataFrame, baseline="fixed_gravity", treatment="adaptive_gravity") -> pd.DataFrame:
    """Resample independent seed-level paired differences, never timesteps."""
    rows = []
    for scenario in SCENARIOS:
        df = metrics[metrics.scenario == scenario]
        a = df[df.controller == baseline].set_index("seed")
        b = df[df.controller == treatment].set_index("seed")
        both = a.merge(b, left_index=True, right_index=True,
                       suffixes=("_base", "_adapt"), how="inner")
        for metric in ("joint_rmse", "ee_rmse_m", "effort_rms_Nm"):
            vals_a = both[f"{metric}_base"].to_numpy()
            vals_b = both[f"{metric}_adapt"].to_numpy()
            improvements = 100 * (vals_a - vals_b) / np.maximum(vals_a, 1e-12)
            avg = float(np.mean(improvements))
            rng = np.random.default_rng(100 + len(scenario) + len(metric))
            resamples = rng.choice(improvements, size=(10000, len(improvements)), replace=True)
            means = np.mean(resamples, axis=1)
            ci_low, ci_high = np.quantile(means, [0.025, 0.975])
            rows.append({"scenario": scenario, "metric": metric, "n_seed_pairs": len(improvements),
                         "baseline_mean": float(np.mean(vals_a)), "adaptive_mean": float(np.mean(vals_b)),
                         "improvement_mean_pct": avg, "ci95_low_pct": float(ci_low),
                         "ci95_high_pct": float(ci_high),
                         "adaptive_better_pairs": int(np.sum(improvements > 0))})
    return pd.DataFrame(rows)


def make_figures(metrics: pd.DataFrame, paired: pd.DataFrame, traces: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"fixed_no_gravity": "#9ca3af", "fixed_gravity": "#34495e",
              "adaptive_no_gravity": "#f39c12", "adaptive_gravity": "#15803d"}
    labels = {"fixed_no_gravity": "PID fixo", "fixed_gravity": "PID fixo + G",
              "adaptive_no_gravity": "PID adaptativo", "adaptive_gravity": "PID adaptativo + G"}
    scenario_labels = ("nominal", "rápido", "carga", "perturbação", "combinado")
    # One figure per joint, physical tracking vs target for one representative seed.
    for scenario in ("nominal", "combined"):
        fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)
        target = traces[(scenario, "fixed_gravity")]
        for j, ax in enumerate(axes):
            ax.plot(target["time"], target["ref"][:, j], "k--", lw=1.2, label="Referência")
            for controller in ("fixed_gravity", "adaptive_gravity"):
                d = traces[(scenario, controller)]
                ax.plot(d["time"], d["q"][:, j], color=colors[controller], lw=1,
                        label=labels[controller])
            ax.set_ylabel(f"Junta {j+1} (rad)")
            ax.grid(alpha=.18)
        axes[0].legend(ncol=3, fontsize=8)
        axes[-1].set_xlabel("Tempo (s)")
        fig.tight_layout()
        fig.savefig(directory / f"tracking_{scenario}.png", dpi=200)
        plt.close(fig)
    # 5 scenario comparisons with 95% CI across independent seeds.
    for metric, filename, unit in (("joint_rmse", "joint_rmse_comparison", "RMSE angular (rad)"),
                                   ("ee_rmse_m", "ee_rmse_comparison", "RMSE do efetuador (m)"),
                                   ("effort_rms_Nm", "effort_comparison", "RMS do torque (N m)")):
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        x = np.arange(len(SCENARIOS))
        for i, ctrl in enumerate(CONTROLLERS):
            m = np.array([metrics[(metrics.scenario == sc) & (metrics.controller == ctrl)][metric].mean() for sc in SCENARIOS])
            sd = np.array([metrics[(metrics.scenario == sc) & (metrics.controller == ctrl)][metric].std(ddof=1) for sc in SCENARIOS])
            n = max(1, metrics.seed.nunique())
            ax.bar(x + (i-1.5)*.18, m, width=.175, label=labels[ctrl],
                   color=colors[ctrl], alpha=.95, yerr=1.96*sd/np.sqrt(n), capsize=2)
        ax.set_xticks(x, scenario_labels)
        ax.set_ylabel(unit)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=200)
        plt.close(fig)
    # Seed-level paired improvements show non-anecdotal result distribution.
    fig, ax = plt.subplots(figsize=(9, 4.1))
    for i, scenario in enumerate(SCENARIOS):
        ss = metrics[metrics.scenario == scenario]
        a = ss[ss.controller == "fixed_gravity"].set_index("seed")["joint_rmse"]
        b = ss[ss.controller == "adaptive_gravity"].set_index("seed")["joint_rmse"]
        improvement = 100 * (a-b)/a
        ax.scatter(np.full(len(improvement), i) + np.linspace(-.12, .12, len(improvement)), improvement,
                   color="#15803d", alpha=.7, s=17)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(range(len(SCENARIOS)), scenario_labels)
    ax.set_ylabel("Redução pareada do RMSE (%, positivo = melhor)")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / "paired_improvements.png", dpi=200)
    plt.close(fig)
    # Gains, torque, error waveform.
    d = traces[("combined", "adaptive_gravity")]
    for key, filename, ylabel in (("kp", "adaptive_kp", "Ganhos proporcionais"),
                                  ("kd", "adaptive_kd", "Ganhos derivativos"),
                                  ("torque", "joint_torque", "Torque nas juntas (N m)"),
                                  ("joint_error", "joint_error", "Erro angular (rad)")):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for j in range(3):
            ax.plot(d["time"], d[key][:, j], lw=1, label=JOINT_NAMES[j])
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=200)
        plt.close(fig)
    # An evidence-based summary chart for cross-scenario effects.
    sub = paired[paired.metric == "joint_rmse"]
    fig, ax = plt.subplots(figsize=(8, 3.7))
    x = np.arange(len(sub))
    y = sub.improvement_mean_pct.to_numpy()
    err = np.vstack((y-sub.ci95_low_pct.to_numpy(), sub.ci95_high_pct.to_numpy()-y))
    ax.bar(x, y, color=np.where(y >= 0, "#15803d", "#c2410c"), width=.6)
    ax.errorbar(x, y, yerr=err, fmt="none", color="black", lw=1.1, capsize=3)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x, scenario_labels)
    ax.set_ylabel("Redução do RMSE (%, IC bootstrap 95%)")
    fig.tight_layout()
    fig.savefig(directory / "improvement_ci.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--duration", type=float, default=16)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.seeds < 2 or args.duration < 12:
        parser.error("Use at least 2 independent seeds and duration >= 12 s")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    traces = {}
    for scenario in SCENARIOS:
        print(f"Running {scenario} ...", flush=True)
        for seed in range(args.seeds):
            for controller in CONTROLLERS:
                row, trace = simulate(scenario, controller, seed, args.duration, store_trace=(seed == 0))
                metrics.append(row)
                if trace is not None:
                    traces[(scenario, controller)] = trace
                    frame = pd.DataFrame({"time": trace["time"], "scenario": scenario,
                                          "controller": controller, "seed": seed,
                                          "q1": trace["q"][:,0], "q2": trace["q"][:,1], "q3": trace["q"][:,2],
                                          "ref1": trace["ref"][:,0], "ref2": trace["ref"][:,1], "ref3": trace["ref"][:,2],
                                          "err1": trace["joint_error"][:,0], "err2": trace["joint_error"][:,1],
                                          "err3": trace["joint_error"][:,2], "ee_error_m": trace["ee_error"],
                                          "torque1": trace["torque"][:,0], "torque2": trace["torque"][:,1],
                                          "torque3": trace["torque"][:,2], "kp1": trace["kp"][:,0],
                                          "kp2": trace["kp"][:,1], "kp3": trace["kp"][:,2],
                                          "kd1": trace["kd"][:,0], "kd2": trace["kd"][:,1],
                                          "kd3": trace["kd"][:,2], "saturated": trace["limit_hit"]})
                    frame.to_csv(out / f"trace_{scenario}_{controller}_seed0.csv", index=False)
    df = pd.DataFrame(metrics)
    df.to_csv(out / "trials.csv", index=False)
    paired = paired_summary(df)
    paired.to_csv(out / "paired_bootstrap.csv", index=False)
    make_figures(df, paired, traces, out / "figures")
    config = {"original_repository": "https://github.com/vitor-souza-ime/pidmujoco",
              "source_commit": "055319e251a11f1042b974f8a897a6300fa2ed1d",
              "engine": "MuJoCo", "mujoco_version": mujoco.__version__,
              "python": platform.python_version(), "seeds": args.seeds,
              "duration_s": args.duration, "timestep_s": 0.005, "log_stride": 10,
              "scenarios": [scenario_spec(sc).__dict__ for sc in SCENARIOS],
              "controllers": list(CONTROLLERS), "gear": GEAR.tolist(),
              "joint_torque_limit_Nm": TORQUE_LIMIT.tolist(), "Kp_initial": KP0.tolist(),
              "Ki_initial": KI0.tolist(), "Kd_initial": KD0.tolist(),
              "adaptation_interval_steps": 50, "history_length_steps": 100,
              "gravity_compensator_nominal_payload_kg": 0.1,
              "payload_scenarios_mismatch_controller": True,
              "bootstrap_resamples": 10000,
              "causal_comparison": "paired independently seeded full trajectories, not initial versus final windows",
              "stability_claim": "empirical only; no formal stability guarantee"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nPaired adaptive vs fixed PID + G, positive indicates better:")
    print(paired[paired.metric == "joint_rmse"].to_string(index=False))
    print("\nWrote", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
