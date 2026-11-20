# 🤖 Adaptive PID MuJoCo Benchmark

Reproducible evaluation of fixed and bounded adaptive PID control for a three-degree-of-freedom robotic manipulator, with emphasis on payload uncertainty, external disturbances, gravity compensation, and tracking accuracy.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![MuJoCo](https://img.shields.io/badge/MuJoCo-4.0.1-2374ab) ![Study](https://img.shields.io/badge/Simulations-400-15803d) ![Research](https://img.shields.io/badge/Research-reproducible-555555)

> 📌 **Repository status:** This repository currently contains its README only. The completed experiment's source files, datasets, and figures have not yet been uploaded here. The commands below will work once the listed files are added to the repository.

## 🔎 Research question

Under identical initial conditions, actuator limits, and nominal gravity compensation, does bounded PID gain adaptation improve joint tracking when a robot experiences an unknown payload or external disturbances?

Rather than comparing the beginning and end of a single trajectory, the benchmark uses **matched controllers and paired random seeds**, including cases where adaptation provides no measurable benefit.

## 🧪 Experimental design

- **Robot:** spatial 3-DoF yaw–pitch–pitch arm in MuJoCo.
- **Simulation:** 16 seconds per run, timestep 0.005 s (3,200 integration steps).
- **Factorial design:** 4 controller variants × 5 scenarios × 20 paired seeds = **400 simulation runs**.
- **Total physics steps:** 1,280,000.
- **Actuation:** joint torque limits [6, 6, 3] N·m, implemented by converting desired joint torque into motor control units using gears [20, 20, 10].
- **Nominal end-effector mass:** 0.10 kg; uncertain-payload condition: 0.40 kg. The controller's gravity model remains calibrated to 0.10 kg.
- **Primary metric:** joint-space tracking RMSE over all three joints and all 3,200 time steps of each run.
- **Uncertainty:** 95% *paired bootstrap* confidence intervals over the 20 seed-level comparisons (10,000 resamples).

### Controllers

| ID | Controller | Gravity compensation | PID gains |
| --- | --- | --- | --- |
| F0 | Fixed PID | None | Constant |
| F1 | Fixed PID | Nominal model | Constant |
| A0 | Adaptive PID | None | Bounded updates |
| A1 | Adaptive PID | Nominal model | Bounded updates |

**Primary comparison:** A1 versus F1. Both use the same nominal gravity model, reference trajectories, initial PID gains, anti-windup, sensors, and torque constraints; only the bounded gain adaptation is different.

### Evaluation scenarios

| Scenario | Reference speed | End-effector mass | External torque pulses | Sensor noise |
| --- | ---: | ---: | :---: | :---: |
| Nominal | 1.0× | 0.10 kg | No | No |
| Fast | 2.5× | 0.10 kg | No | No |
| Payload mismatch | 1.0× | 0.40 kg | No | No |
| Disturbances | 1.0× | 0.10 kg | Yes | No |
| Combined stress | 2.5× | 0.40 kg | Yes | Yes |

The disturbance experiment applies joint torque pulses (−1.4 N·m on joint 2 during 5–7 s, and +0.75 N·m on joint 3 during 9–11 s), with controlled seed-level variation. The combined scenario also includes position noise (standard deviation 0.003 rad) and velocity noise (standard deviation 0.03 rad/s).

## 📊 Measured results

The following values come from the completed MuJoCo experiment. Positive improvements mean that adaptive PID **reduced** tracking error relative to fixed PID with the same gravity compensation.

| Scenario | Fixed PID + G RMSE (rad) | Adaptive PID + G RMSE (rad) | Mean improvement | 95% paired CI |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0.00742 | 0.00742 | 0.00% | [0.00%, 0.00%] |
| Fast | 0.00757 | 0.00757 | 0.00% | [0.00%, 0.00%] |
| **Payload mismatch** | **0.08324** | **0.04885** | **41.31%** | **[41.14%, 41.47%]** |
| **Disturbances** | **0.02722** | **0.02475** | **9.09%** | **[8.94%, 9.23%]** |
| **Combined stress** | **0.08628** | **0.05152** | **40.29%** | **[40.11%, 40.48%]** |

Adaptive PID matched the fixed-gravity baseline in the nominal and fast scenarios. Its benefit was conditional on persistent tracking error caused by payload mismatch and/or disturbances. Gravity compensation itself was also an important factor and is isolated by the F0/F1 and A0/A1 comparisons.

All 400 runs finished with finite recorded values; no joint-limit proximity events were observed in this defined simulation protocol. These statements are not hardware safety guarantees.

## ⚙️ Reproducibility

Once the experiment files are committed, place the contents of the `pidmujoco_enhanced/` package at the repository root, then run:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python study.py --seeds 20 --duration 16 --output results
```

The implementation uses Python, MuJoCo, NumPy, SciPy, pandas, Matplotlib, and pytest. It uses fixed seed mappings for initial joint configurations, measurement noise, and perturbations, so controller comparisons remain paired.

### Intended repository layout

```text
adaptive-pid-mujoco-benchmark/
├── README.md
├── model.py
├── study.py
├── requirements.txt
├── tests/
│   └── test_study.py
└── results/
    ├── config.json
    ├── trials.csv
    ├── paired_bootstrap.csv
    ├── trace_*.csv
    └── figures/
        └── *.png
```

`trials.csv` includes all 400 simulation-level metrics; `paired_bootstrap.csv` contains scenario-level means, improvements, and confidence intervals. The trace files provide time-series data for a representative paired seed. Eleven plots are generated from the measured outputs.

## 🧠 Control implementation

- Identical initial PID gains across fixed and adaptive controllers: Kp = [12, 14, 8], Ki = [0.60, 0.60, 0.35], Kd = [2.0, 2.3, 1.2].
- Conditional integration and an integral clamp prevent simple saturation-induced windup.
- The adaptive strategy inspects consecutive 50-step error-RMSE windows and applies bounded changes only when tracking error remains above a declared threshold.
- Motor input and actual joint torque are treated as different units: **motor control = desired joint torque / actuator gear**.
- Nominal-model gravity compensation is deliberately *not* retuned when the physical payload changes.

## ✅ Validation and limitations

Automated checks cover forward kinematics against MuJoCo, motor-gear/joint-torque conversion, and deterministic paired execution with torque bounds. The reported analysis uses full-trajectory metrics rather than manually reading values from plots.

This is a **simulation-only empirical study**. The adaptation rule is heuristic; the experiments do not prove Lyapunov stability, real-world robustness, or superiority over advanced adaptive controllers that were not implemented as comparators. Results should not be generalized to other payloads, robots, or disturbance profiles without further testing.

## 📚 Research context and attribution

- Original robot experiment: [vitor-souza-ime/pidmujoco](https://github.com/vitor-souza-ime/pidmujoco), source commit `055319e251a11f1042b974f8a897a6300fa2ed1d`.
- Todorov, Erez & Tassa (2012). *MuJoCo: A physics engine for model-based control*. [DOI](https://doi.org/10.1109/IROS.2012.6386109).
- Lee et al. (2020). *An Adaptive PID Control for Robot Manipulators Under Substantial Payload Variations*. [DOI](https://doi.org/10.1109/ACCESS.2020.3014348).
- Qiao et al. (2022). *Adaptive PID control of robotic manipulators without equality/inequality constraints on control gains*. [DOI](https://doi.org/10.1002/rnc.5849).

### Related manuscript

*Comparative Evaluation of Fixed and Adaptive PID Control in a Robot Manipulator Under Payload Mismatch and Disturbances* — manuscript prepared for journal submission; **not yet published**.

---

Maintained by [Vitor Amadeu Souza](https://github.com/vitor-souza-ime).
