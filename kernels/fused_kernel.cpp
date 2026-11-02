// fused_kernel.cpp
//
//   q = fused_kernel(theta, model, lvSpd, len, dt, space, [threads], [evSpd])
//
//   model = 0  IDM,   theta = [v0, T, a, b, delta, s0]
//   model = 1  Gipps, theta = [a_n, b_n, S_n1, b, V_0, tau]
//
//   theta is 6 x 1; len is N x 1; lvSpd, space and evSpd are N x T. All real double;
//   anything else is an error.
//
// The rollout and the residual happen in one pass: no trajectory is written.
//
// An event that has ended contributes a zero weight, `live`, to the state, the residual sum and the
// collision flag, in both branches; no frame is skipped by a branch.
//
// Threads come from the caller: one parallel region per call, a static split inside each frame.

#include "mex.h"
#include <cmath>
#include <cstddef>
#include <vector>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif


// One set of buffers per MEX instance, grown on demand and reused. MATLAB runs one MEX call at a
// time per process, and each pool worker is its own process, so a static buffer is safe; inside a
// call the buffers are indexed per thread explicitly.
static std::vector<double> g_buf_d;
static std::vector<char>   g_buf_c;

static const mxArray* require_real(const mxArray* a, const char* name)
{
    if (!mxIsDouble(a) || mxIsComplex(a))
        mexErrMsgIdAndTxt("fused_kernel:type", "%s must be a real double array", name);
    if (mxGetNumberOfDimensions(a) > 2)
        mexErrMsgIdAndTxt("fused_kernel:dims", "%s must have two dimensions", name);
    return a;
}

static const mxArray* require_shape(const mxArray* a, mwSize m, mwSize n, const char* name)
{
    require_real(a, name);
    if (mxGetM(a) != m || mxGetN(a) != n)
        mexErrMsgIdAndTxt("fused_kernel:dims", "%s must be %u x %u, got %u x %u",
                          name, (unsigned)m, (unsigned)n,
                          (unsigned)mxGetM(a), (unsigned)mxGetN(a));
    return a;
}

static double require_scalar(const mxArray* a, const char* name)
{
    if (!mxIsNumeric(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1)
        mexErrMsgIdAndTxt("fused_kernel:type", "%s must be a real numeric scalar", name);
    return mxGetScalar(a);
}

// Same test as the dead mask in +cfm/objective.m.
static inline bool is_dead(double s)
{
    return !std::isfinite(s) || s <= 0.0;
}

// One set of buffers per MEX instance, grown on demand. A request for less than a quarter of what is
// held shrinks the buffer, and fused_kernel('release') frees it entirely.
static double* arena_d(std::size_t n)
{
    if (g_buf_d.size() < n) {
        g_buf_d.resize(n);
    } else if (g_buf_d.size() > 4 * n) {
        std::vector<double>(n).swap(g_buf_d);     // hand the excess back to the system
    }
    return g_buf_d.data();
}

static char* arena_c(std::size_t n)
{
    if (g_buf_c.size() < n) {
        g_buf_c.resize(n);
    } else if (g_buf_c.size() > 4 * n) {
        std::vector<char>(n).swap(g_buf_c);
    }
    return g_buf_c.data();
}

static void arena_release()
{
    std::vector<double>().swap(g_buf_d);
    std::vector<char>().swap(g_buf_c);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // fused_kernel('release') frees the arena; any other call is an evaluation.
    if (nrhs == 1 && mxIsChar(prhs[0])) {
        char cmd[32];
        mxGetString(prhs[0], cmd, sizeof(cmd));
        if (std::strcmp(cmd, "release") == 0) {
            arena_release();
            return;
        }
        mexErrMsgIdAndTxt("fused_kernel:cmd", "unknown command %s", cmd);
    }
    (void)nlhs;
    // This binary and MATLAB load different OpenMP runtimes; the lock keeps this one loaded until
    // MATLAB unloads, so the two cannot tear down in an order that faults.
    {
        static bool locked = false;
        if (!locked) { mexLock(); locked = true; }
    }
    if (nrhs != 8)
        mexErrMsgIdAndTxt("fused_kernel:nrhs", "eight inputs: theta, model, lvSpd, len, dt, space, threads, evSpd");

    int nthreads = (int)require_scalar(prhs[6], "threads");
    if (nthreads < 1) nthreads = 1;
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif

    const mwSize N = mxGetM(require_real(prhs[2], "lvSpd"));
    const mwSize T = mxGetN(prhs[2]);

#ifdef _OPENMP
    // A call with few events cannot use a thread team: the team costs more than the work. The
    // threshold keeps a one-event call (block=1) from paying for threads it cannot use.
    if (N < 64) { nthreads = 1; omp_set_num_threads(1); }
#endif

    require_shape(prhs[0], 6, 1, "theta");
    const double* para  = mxGetPr(prhs[0]);
    const int     model = (int)require_scalar(prhs[1], "model");
    const double* lvSpd = mxGetPr(prhs[2]);
    const double* len   = mxGetPr(require_shape(prhs[3], N, 1, "len"));
    const double  dt    = require_scalar(prhs[4], "dt");
    const double* space = mxGetPr(require_shape(prhs[5], N, T, "space"));

    if (!(dt > 0.0))
        mexErrMsgIdAndTxt("fused_kernel:dt", "dt must be positive, got %g", dt);

    // The observed ego speed per frame, which Gipps reads for the frames before its reaction time.
    const double* evSpd = mxGetPr(require_shape(prhs[7], N, T, "evSpd"));

    plhs[0] = mxCreateDoubleMatrix(N, 1, mxREAL);
    double* q = mxGetPr(plhs[0]);

    switch (model) {
    case 0: {
        const double v0 = para[0], Th = para[1], am = para[2], bb = para[3];
        const double de = para[4], s0 = para[5];
        const double sab = std::sqrt(am * bb);   // hoisted once per call, not per frame

        // as[] is written before it is read; collided[] must start at zero.
        double* sbuf = arena_d((std::size_t)3 * (std::size_t)N);
        double* sb = sbuf;
        double* vb = sbuf + N;
        double* as = sbuf + 2 * (std::size_t)N;
        char*   collided = arena_c(N);
        std::memset(collided, 0, (std::size_t)N);
        for (mwSize i = 0; i < N; ++i) {
            sb[i] = space[i];          // the initial frame is the recorded one
            vb[i] = evSpd[i];
            double ds = sb[i] - space[i];
            as[i] = ds * ds;
        }

        double* sp = sb; double* vp = vb;
        double* ap = as;

#ifdef _OPENMP
        #pragma omp parallel
        {
#endif
            for (mwSize t = 1; t < T; ++t) {
                const mwSize off = N * t;
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                // flat loop; MSVC gates `omp simd` behind -openmp:experimental
                for (long long ii = 0; ii < (long long)N; ++ii) {
                    const mwSize i = (mwSize)ii;
                    // an ended event is a weight, so the loop stays vectorisable
                    const double live = ((double)(t + 1) <= (double)len[i]) ? 1.0 : 0.0;
                    double sv = sp[i], vv = vp[i];

                    double wv = lvSpd[i + N * (t - 1)] - vv;
                    double u = Th * vv - vv * wv / (2.0 * sab);
                    double sstar = (u > 0.0 ? u : 0.0) + s0;
                    double acc = am * (1.0 - std::pow(vv / v0, de) - (sstar / sv) * (sstar / sv));
                    double vn = vv + acc * dt;
                    if (vn < 0.0) vn = 0.0;
                    double wn = lvSpd[i + off] - vn;
                    double sn = sv + dt * (wv + wn) / 2.0;
                    // recorded, not stopped: the reduction scores 1e5, and the work stays equal to
                    // an implementation that cannot stop
                    if (live > 0.0 && is_dead(sn)) { collided[i] = 1; }
                    sp[i] = live * sn + (1.0 - live) * sv;
                    vp[i] = live * vn + (1.0 - live) * vv;
                    double dsn = sn - space[i + off];
                    ap[i] += live * dsn * dsn;
                }
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (long long ii = 0; ii < (long long)N; ++ii) {
                const mwSize i = (mwSize)ii;
                q[i] = collided[i] ? 1e5 : ap[i] / len[i];    // per-event MSE
            }
#ifdef _OPENMP
        }
#endif
        return;
    }
    case 1: {
        const double a_n = para[0], b_n = para[1], S_n1 = para[2];
        const double bb = para[3], V_0 = para[4], tau = para[5];
        mwSize lag = (mwSize)std::ceil(tau / dt);
        if (lag < 1) lag = 1;
        if (lag > T - 1) lag = T - 1;

        // The history is indexed by frame, as the IDM branch indexes its state. One arena request for
        // all of it: a second request may resize the buffer and invalidate the first pointer.
        const std::size_t fr = (std::size_t)N * (std::size_t)T;
        double* buf = arena_d(2 * fr + (std::size_t)N);
        double* sh = buf;
        double* vh = buf + fr;
        double* ap = buf + 2 * fr;
        char*   collided = arena_c(N);
        std::memset(collided, 0, (std::size_t)N);
        // Frames 0..lag-1 are the observed record: a driver at frame lag reacts to what the two vehicles
        // were doing at frame 0, and that is data, not a simulation. Simulation starts at `lag`, and only
        // the frames it produces are scored, so the warm-up never enters the objective.
        for (mwSize t = 0; t < lag; ++t) {
            const mwSize off = N * t;
            for (mwSize i = 0; i < N; ++i) {
                sh[i + off] = space[i + off];
                vh[i + off] = evSpd[i + off];
            }
        }
        for (mwSize i = 0; i < N; ++i) { ap[i] = 0.0; }

#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            for (mwSize t = lag; t < T; ++t) {
                const mwSize off = N * t;
                const mwSize offp = N * (t - 1);
                const mwSize hoff = N * (t - lag);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (long long ii = 0; ii < (long long)N; ++ii) {
                    const mwSize i = (mwSize)ii;
                    const double live = ((double)(t + 1) <= (double)len[i]) ? 1.0 : 0.0;
                    double v_d = vh[i + hoff], s_d = sh[i + hoff], lv_d = lvSpd[i + hoff];
                    double v_free = v_d + 2.5 * a_n * tau * (1.0 - v_d / V_0)
                                          * std::sqrt(0.025 + v_d / V_0);
                    double disc = b_n * b_n * tau * tau
                                + b_n * (2.0 * (s_d - S_n1) - v_d * tau + lv_d * lv_d / bb);
                    if (disc < 0.0) disc = 0.0;
                    double v_safe = -b_n * tau + std::sqrt(disc);
                    double v_new = (v_free < v_safe) ? v_free : v_safe;
                    if (v_new < 0.0) v_new = 0.0;

                    // Only the speeds the model reads are delayed; the spacing advances on the live
                    // state, as in +cfm/gipps.m.
                    double v_lead = lvSpd[i + off];
                    double rel_now = lvSpd[i + offp] - vh[i + offp];
                    double s_new = sh[i + offp] + dt * (rel_now + (v_lead - v_new)) / 2.0;

                    if (live > 0.0 && is_dead(s_new)) { collided[i] = 1; }
                    sh[i + off] = live * s_new + (1.0 - live) * sh[i + offp];
                    vh[i + off] = live * v_new + (1.0 - live) * vh[i + offp];
                    double dsn = s_new - space[i + off];
                    ap[i] += live * dsn * dsn;
                }
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (long long ii = 0; ii < (long long)N; ++ii) {
                const mwSize i = (mwSize)ii;
                q[i] = collided[i] ? 1e5 : ap[i] / len[i];     // per-event MSE, the generic formula
            }
#ifdef _OPENMP
        }
#endif
        return;
    }
    default:
        mexErrMsgIdAndTxt("fused_kernel:model", "unknown model id %d", (int)model);
    }
}
