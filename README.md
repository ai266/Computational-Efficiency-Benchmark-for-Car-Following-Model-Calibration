# Computational Efficiency Benchmark for Car-Following Model Calibration

In the transportation community, microscopic traffic simulation model calibration lacks both standards
and awareness of how code-level implementation combines with high-performance computing (HPC), usually
resulting in inefficient code. This project reports a benchmark for the computational efficiency of
car-following model calibration that integrates common HPC techniques, supporting efficient exploration
in transportation science and development in engineering. It covers the Intelligent Driver Model (IDM)
and the Gipps model calibrated on the Next Generation Simulation (NGSIM) and HighD datasets.

**Contents** ·
[Installation](#installation) ·
[Quick start](#quick-start) ·
[Call sequence](#call-sequence) ·
[Model and parameters](#model-and-parameters) ·
[Objective](#objective) ·
[HPC components](#hpc-components) ·
[Data](#data) ·
[Benchmark experiments](#benchmark-experiments) ·
[References](#references)

---

## Installation

Implemented and measured with MATLAB R2025a with the Global Optimization and Parallel Computing toolboxes,
and Microsoft Visual C++ 2022 with the C++ workload; other recent releases should work. The kernel is an
OpenMP program; `build_kernel.m` compiles it with `/openmp`.

```matlab
addpath('<path-to>/cfm-bench');   % the package; run_calibration adds its kernels folder itself
build_kernel();                   % compiles kernels/fused_kernel.cpp and checks it against the model
```

`data/` holds four converted splits: NGSIM I-80 train and test, and HighD train and test, the
car-following events that [1] extracts from the two corpora. To build them, put the published `.npy`
files in `data/raw/` and run, from the package folder,

```bash
python data_to_mat.py
```

The script converts each split to `.mat` at 10 Hz over 150 frames, 15 seconds per event. Requires Python
3.10 or later with `numpy` and `scipy`.

---

## Quick start

```matlab
out = run_calibration('ngsim_train', 'ngsim_test', 'idm');

out.theta        % one entry per parameter, d x 1
out.train_rmse   % spacing RMSE on the training split, metres
out.test_rmse    % the same on the test split
out.t_calib      % wall clock of the search, seconds; the pool starts before the clock
```

`run_calibration` calibrates with a genetic algorithm, using MATLAB's `ga` as its default operators. The
calibration minimises the spacing root mean squared error (RMSE) over the sampling box of the chosen
model. Any field can be overridden for one call:

```matlab
out = run_calibration('highd_train', 'highd_test', 'gipps', struct('workers', 8));
out = run_calibration('ngsim_train', 'ngsim_test', 'idm', struct('threads', 1, 'block', 256));
out = run_calibration('ngsim_train', 'ngsim_test', 'idm', struct('pop', 200, 'gens', 50, 'seed', 11));
```

---

## Call sequence

One calibration, in the order the work happens; the right column is where each step is defined.

```text
run_calibration(train, test, model, cfg)                       run_calibration.m
│
├─ cfm.load_trajectories(train)                                +cfm/load_trajectories.m
├─ cfm.config('model', model)                                  +cfm/config.m
├─ parpool(cfg.workers)                                        MATLAB
│
├─ ga(...)                                                     MATLAB
│   └─ f(x) = cfm.objective(x, D, o)                           +cfm/objective.m
│        ├─ fused_kernel(theta, o.kernel, ...)                 kernels/fused_kernel.cpp
│        │    build_kernel('ensure') on first use              build_kernel.m
│        └─ o.rollout(theta, D, D.dt)                          +cfm/idm.m, +cfm/gipps.m
│
├─ fused_kernel('release')                                     kernels/fused_kernel.cpp
│
└─ out  theta, train/test RMSE, t_eval, t_gen, t_calib, seed   run_calibration.m
```

**Settings.** Every number a run takes is in `+cfm/config.m` and is overridden by name, as
`cfm.config('threads', 8, 'model', 'gipps')` shows; the field is called `model` and selects one of the two
models.

| Field | Default | Meaning |
|:---|:---:|:---|
| `model` | `'idm'` | `'idm'` or `'gipps'` |
| `compile` | `true` | use the compiled kernel |
| `workers` | `4` | processes over candidates; `0` runs in this process |
| `threads` | `4` | OpenMP threads inside the kernel |
| `block` | `[]` | events per call, `[]` meaning the whole set |
| `window` | `150` | frames per event |
| `frameStep` | `0.1` | seconds per frame |
| `pop` | `100` | candidates per generation |
| `gens` | `100` | generations |
| `seed` | `7` | seed of the draw |

`run_calibration` reads `pop`, `gens` and `seed` from `cfg`, along with the two width settings, `workers`
and `threads`. `cfg.population` supplies a `d` × pop starting generation, one column per candidate, where
`d = numel(lb)` is the number of parameters of the model, 6 for both here. Five further fields change how
one call runs rather than what it searches: `block` sets the events per call, `D` passes an event set
already in memory instead of a split name, `override` passes a whole settings struct, `objective` replaces
the objective, and `keep_pool` keeps the pool alive on return.

**Extending.** A model is one row of the table at the foot of `+cfm/config.m`: its name, its
compiled-kernel branch (or `NaN` for an interpreted-only model), its sampling box and a handle to its
rollout. A caller can replace the objective for one run by passing `cfg.objective`, a handle
`f(theta, D, o)` that returns the scalar to minimise.

---

## Model and parameters

Two classical car-following models, the Intelligent Driver Model (IDM) [2] and the model of Gipps [3],
cover the arithmetic that a calibration benchmark has to price. The follower's state is its spacing $s$
and its speed $v^{\mathrm{f}}$; the leader's speed $v^{\ell}$ is an input, and
$\Delta v = v^{\mathrm{f}} - v^{\ell}$. A subscript on a state variable denotes a frame, and each frame
advances time by $\Delta t$.

**IDM** [2], $\theta=(v_0,T,a,b,\delta,s_0)$:

$$
a_{\mathrm{IDM}}\left(s(t), v^{\mathrm{f}}(t), \Delta v(t)\right) =
a\left[1-\left(\frac{v^{\mathrm{f}}(t)}{v_0}\right)^{\delta} -\left(\frac{s^{\ast}(t)}{s(t)}\right)^{2}\right],
\qquad
s^{\ast}(t) = s_0 + v^{\mathrm{f}}(t)T + \frac{v^{\mathrm{f}}(t)\Delta v(t)}{2\sqrt{ab}} .
$$

The desired speed $v_0$, the desired time headway $T$, the maximum acceleration $a$, the comfortable
deceleration $b$, the acceleration exponent $\delta$ and the standstill spacing $s_0$ set the desired gap
and the response to it. The model returns the acceleration $a_{\mathrm{IDM}}$ at the current frame.

**Gipps** [3], $\theta=(a_n,b_n,S_{n1},b,V_0,\tau)$:

$$
v^{\mathrm{f}}_{\mathrm{Gipps}}\left(s(t), v^{\mathrm{f}}(t), v^{\ell}(t)\right) =
\min\lbrace
v^{\mathrm{f}}(t) + 2.5 a_n\tau\left(1-\frac{v^{\mathrm{f}}(t)}{V_0}\right)\sqrt{0.025+\frac{v^{\mathrm{f}}(t)}{V_0}}, -b_n\tau+\sqrt{b_n^{2}\tau^{2}+b_n\left(2(s(t)-S_{n1})-v^{\mathrm{f}}(t)\tau+\frac{(v^{\ell}(t))^{2}}{b}\right)}
\rbrace.
$$

The maximum acceleration $a_n$, the maximum deceleration $b_n$, the standstill gap $S_{n1}$, the leader's
estimated deceleration $b$, the desired speed $V_0$ and the reaction time $\tau$ set two bounds per step,
and the model takes the lower. The model returns the follower's speed one reaction time ahead,
$v^{\mathrm{f}}_{\mathrm{Gipps}}\left(s(t), v^{\mathrm{f}}(t), v^{\ell}(t)\right) = v^{\mathrm{f}}(t+\tau)$.

The two models represent the two classical stimulus-response forms: IDM responds to the current
stimulus without delay, while Gipps delays its response by the reaction time $\tau$.

**Sampling box.** One box per model, because the parameter meanings differ:

| $\theta$ | IDM | range | Gipps | range |
|:---:|:---|--:|:---|--:|
| 1 | $v_0$ (m/s) | 0.3–41.7 | $a_n$ (m/s²) | 0.1–5.0 |
| 2 | $T$ (s) | 0.1–5.0 | $b_n$ (m/s²) | 0.1–5.0 |
| 3 | $a$ (m/s²) | 0.1–5.0 | $S_{n1}$ (m) | 0.5–3.0 |
| 4 | $b$ (m/s²) | 0.1–5.0 | $b$ (m/s²) | 0.1–5.0 |
| 5 | $\delta$ (–) | 1.0–40.0 | $V_0$ (m/s) | 0.3–41.7 |
| 6 | $s_0$ (m) | 0.1–10.0 | $\tau$ (s) | 0.3–3.0 |

---

## Objective

The objective needs a spacing sequence to compare against the observed spacing sequence, and neither model
returns spacing directly. The evaluation therefore advances the state from the event's first observed
frame at $\Delta t = 0.1$ s, using the model output and a trapezoidal spacing increment:

$$
s_{t+1}=s_t+\frac{\Delta t}{2}\left(\Delta v_t+\Delta v_{t+1}\right),
\qquad
v^{\mathrm{f}}_{t+1}=
\begin{cases}
\max\left(0, v^{\mathrm{f}}_{t}+a_{\mathrm{IDM}}\left(s_t,v^{\mathrm{f}}_{t},\Delta v_t\right)\Delta t\right), & \text{IDM},\\
v^{\mathrm{f}}_{\mathrm{Gipps}}\left(s_{t+1-\lambda}, v^{\mathrm{f}}_{t+1-\lambda}, v^{\ell}_{t+1-\lambda}\right), & \text{Gipps}.
\end{cases}
$$

where $\lambda$ is the reaction time $\tau$ expressed in frames, that is $\tau/\Delta t$ rounded up to an
integer, and the Gipps branch evaluates the model of the previous section at frame $t+1-\lambda$,
while IDM reads the current frame.

With $n_e$ frames in event $e$ and simulated spacing $s^{\mathrm{sim}}_{e,t}$, the spacing error is pooled
over events so that no event dominates by being long. The mean squared error (MSE) and the RMSE follow the
pooling of [1]:

$$
\mathrm{MSE}(\theta)=\frac{1}{|\mathcal{E}|}\sum_{e}\frac{1}{n_e}\sum_{t=1}^{n_e}
\bigl(s^{\mathrm{sim}}_{e,t}(\theta)-s^{\mathrm{obs}}_{e,t}\bigr)^{2},
\qquad
\mathrm{RMSE}(\theta)=\sqrt{\mathrm{MSE}(\theta)} .
$$

A failed car-following event, whether a collision or a numerical failure, is scored $10^{5}$ and stays in
the denominator, so every implementation is compared on the full event set.

---

## HPC components

The benchmark follows two HPC principles: do the least work per unit of input, and overlap as much work
as the hardware allows. It applies seven HPC components across three levels.

| Level | Component | Effect | Switch |
|:---|:---|:---|:---:|
| algorithmic structure | Loop fusion | eliminates the intermediate trajectory array and the second traversal | — |
| | Event batching | amortises per-call dispatch overhead over a block of events | `block` |
| code generation | Ahead-of-time compilation | eliminates interpreter dispatch and temporary allocation | `compile` |
| | Guard elision | removes per-frame control-flow divergence from the event loop | — |
| | Loop-invariant hoisting | hoists loop-invariant computation out of the frame loop | — |
| parallel execution | Kernel threads | shared-memory parallelism over the event dimension | `threads` |
| | Task parallelism | task-level parallelism over the candidate dimension | `workers` |

Three further techniques are not considered as components. Auto-vectorisation is not isolated: the event
loop is written without a per-frame branch and left to the Microsoft C++ compiler's vectoriser at its `/O2`
optimisation level, which `build_kernel.m` sets. No GPU execution or mixed precision, because the working
dataset is small enough for the CPU pool.

---

## Data

Two public corpora supply the trajectories: the I-80 section of NGSIM [4] and the German highway
recordings of HighD [5]. Both enter through the four event splits published with FollowNet [1], which fix
one extraction criterion across the two sources, namely a constant leader, at least fifteen seconds of
recording, and no lateral offset beyond two metres. The benchmark takes those splits as given and does
not re-extract events, so a comparison against FollowNet's tables reads the same events on both sides.
`window` sets the maximum number of frames an event is cut to, and `frameStep` sets the frame rate; an
event shorter than `window` keeps its own length, so the frame counts below vary by event.

The four inputs are `.npy` files with the names below. They go in `data/raw/`, or in `data/`, which the
conversion script also checks, and `python data_to_mat.py` writes one `.mat` file per split.

```text
NGSIM_I_80_train_data.npy      NGSIM_I_80_test_data.npy
HighD_train_data.npy           HighD_test_data.npy
```

Each output file holds `events`, an $N \times T \times 4$ array, where $N$ is the number of events and
$T$ the number of frames kept per event, with the four channels in this order.

| Channel | Contents |
|:---:|:---|
| 1 | spacing to the leader, metres |
| 2 | follower speed, m/s |
| 3 | relative speed, m/s |
| 4 | leader speed, m/s |

`cfm.load_trajectories` maps that array onto the contract every measurement reads. `space`, `evSpd` and
`lvSpd` are $N \times T$ arrays; `len` is $N \times 1$ and carries the live frames of each event, so an
event shorter than `window` is scored over its own frames; `N`, `T`, `dt` and `name` carry the dimensions,
the frame step and the set name. `space` is the observed spacing, $s^{\mathrm{obs}}_{e,t}$ in the
objective, while the simulated spacing $s^{\mathrm{sim}}_{e,t}$ comes from the objective's state
advance. In that formula $|\mathcal{E}| = N$ and $n_e$ is the entry of `len` for event $e$. The sizes
below are what the conversion produces on the published splits, and a reader can check a rebuilt split
against them.

```text
ngsim_train.mat    (1351, 914, 4)       N=1351  T=914  kept=150
ngsim_test.mat     (289, 847, 4)        N=289   T=847  kept=150
highd_train.mat    (8779, 150, 4)       N=8779  T=150  kept=150
highd_test.mat     (1880, 150, 4)       N=1880  T=150  kept=150
```

```matlab
D = cfm.load_trajectories('highd_train');      % 8779 x 150, dt 0.1
```

The loader reads one field, `events`, and ignores any other field in the file, so a new corpus enters the
benchmark once it is written in this shape at the frame rate and window set by `frameStep` and `window`;
`data/README.md` carries the sources and the licence, and it is the place to record them for a new corpus.
A dataset at another sampling rate is resampled before conversion, and nothing outside `data_to_mat.py`
changes.

---

## Benchmark experiments

This package and the published implementations of Punzo et al. [6], FollowNet [1] and Hou et al. [7]
evaluate the same events through the data interface above, under the same objective and the same budget
of 10,000 evaluations per calibration, set as 100 candidates over 100 generations. Each implementation
runs with its own HPC components as they appear in its open-source code, in its own language and on its
own runtime: this package and [6] in MATLAB R2025a, [1] and [7] in Python 3.14. The search is the
implementation's own default: this package and [6] call MATLAB's `ga` with its shipped operators, [1]
calls its `geneticalgorithm`, and [7] calls its `pygad`. Where a default search cannot run without a setting,
the value is the one in the implementation's own example.

All measurements are taken on a laptop with an Intel Core Ultra 9 275HX (24 cores) and 32 GB of RAM,
running Windows 11, and every implementation executes on the CPU.

Each calibration is timed from the call that starts the search to the call that returns its best
candidate. The process pool starts, and the kernel is built, before the clock starts; the first
evaluation runs untimed, and the kernel buffers are released after the clock stops, so process start-up,
compilation, the first call and buffer release stay outside the measurement. The test split is scored
after the clock stops.

Which of the seven components each implementation has, as its own source shows. A tick means it has the
component; the `Switch` column above applies to this package only, and fusion is in the kernel by
construction rather than behind a setting.

| Component | this package | Punzo et al. [6] | FollowNet [1] | Hou et al. [7] |
|:---|:---:|:---:|:---:|:---:|
| Loop fusion | ✓ | | | |
| Event batching | ✓ | | | ✓ |
| Guard elision | ✓ | | | ✓ |
| Ahead-of-time compilation | ✓ | | | |
| Loop-invariant hoisting | ✓ | | | ✓ |
| Kernel threads | ✓ | | | |
| Task parallelism | ✓ | ✓ | | |
| *width* | 4 processes × 4 threads | 12 processes × 1 thread | 1 process | 1 process |
| *language* | MATLAB, C++/MEX | MATLAB | Python, NumPy | Python, PyTorch |
| *device* | CPU | CPU | CPU | CPU |

One calibration of each implementation on each cell:

| Implementation | Cell / Model | RMSE (m) | Time (s) |
|:---|:---|--:|--:|
| **this package** | NGSIM / IDM | 4.870 | 16.2 |
| | NGSIM / Gipps | 5.432 | 15.0 |
| | HighD / IDM | 5.005 | 25.5 |
| | HighD / Gipps | 6.097 | 21.3 |
| **Punzo et al. [6]** | NGSIM / IDM | 4.869 | 117.7 |
| | NGSIM / Gipps | 5.432 | 109.7 |
| | HighD / IDM | 5.005 | 708.1 |
| | HighD / Gipps | 6.097 | 639.0 |
| **FollowNet [1]** | NGSIM / IDM | 4.946 | 2147.2 |
| | NGSIM / Gipps | 5.425 | 2473.6 |
| | HighD / IDM | 5.150 | 17064.1 |
| | HighD / Gipps | 5.568 | 19286.8 |
| **Hou et al. [7]** | NGSIM / IDM | 4.874 | 1957.7 |
| | NGSIM / Gipps | 5.359 | 2010.9 |
| | HighD / IDM | 5.025 | 13053.0 |
| | HighD / Gipps | 5.752 | 13458.7 |

Speed-up against this package:

| Implementation | NGSIM / IDM | NGSIM / Gipps | HighD / IDM | HighD / Gipps |
|:---|--:|--:|--:|--:|
| Punzo et al. [6] | 7.3× | 7.3× | 27.8× | 30.0× |
| FollowNet [1] | 132.5× | 164.9× | 669.2× | 905.5× |
| Hou et al. [7] | 120.8× | 134.1× | 511.9× | 631.9× |

The four implementations reach close to the same fit: this package and [6] reach the same RMSE on every
cell, both running MATLAB's `ga` at its defaults, and [1] and [7] lie within a few per cent of this
package's RMSE.

This package is the fastest on every cell, and its per-frame cost is the lowest. That cost orders the four
implementations the same way on every cell, while the event count scales it and widens the absolute margin,
from 120.8× on NGSIM to 511.9×–905.5× on HighD. [6] moves from 7.3× to 30.0× across the same four cells
for the same reason.

---

## References

1. X. Chen, M. Zhu, K. Chen, et al., "FollowNet: A Comprehensive Benchmark for Car-Following Behavior
   Modeling," *Sci. Data*, vol. 10, no. 1, Art. no. 828, 2023.
2. M. Treiber, A. Hennecke, and D. Helbig, "Congested traffic states in empirical observations and
   microscopic simulations," *Phys. Rev. E*, vol. 62, no. 2, pp. 1805–1824, 2000.
3. P. G. Gipps, "A behavioural car-following model for computer simulation," *Transp. Res. Part B:
   Methodol.*, vol. 15, no. 2, pp. 105–111, 1981.
4. U.S. Department of Transportation, Federal Highway Administration, "Next Generation Simulation
   (NGSIM) vehicle trajectories and supporting data," 2006. The I-80 trajectories are the rebuilt version
   distributed with [1].
5. R. Krajewski, J. Bock, L. Kloeker, and L. Eckstein, "The highD dataset: A drone dataset of
   naturalistic vehicle trajectories on German highways for validation of highly automated driving
   systems," in *Proc. IEEE 21st Int. Conf. Intell. Transp. Syst. (ITSC)*, 2018, pp. 2118–2125.
6. V. Punzo, Z. Zheng, and M. Montanino, "About calibration of car-following dynamics of
   automated and human-driven vehicles: Methodology, guidelines and codes," *Transp. Res.
   Part C: Emerg. Technol.*, vol. 128, Art. no. 103165, 2021.
7. H. Hou, A. Kusari, and B. T. W. Lin, "Evaluating parametric car-following models in
   naturalistic congestion: Insights in driver behavior and model limitations," arXiv:2511.18135, 2025.

---

## License

MIT. See `LICENSE`.
