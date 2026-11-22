// fused_kernel_batch.cpp
//
//   A = fused_kernel_batch(theta, model, lvSpd, len, dt, space, evSpd, threads, reduce, precision)
//
//   theta is 6 x P. This kernel is the device kernel of gpu_kernel_batch.cu with the device removed:
//   the work item is still one (candidate, event) pair, the state of a trajectory still lives in the
//   registers of the worker that simulates it, and the reduction still turns those per-event mean
//   squares into five numbers per candidate. OpenMP replaces the launch, so the same code runs the
//   generation on the processor over `threads` threads.
//
//   reduce = 1 returns P x 5: [sum of per-event mean squares, weighted sum, frames, live, dead].
//   reduce = 0 returns N x P, the per-event mean squares.
//   precision = 'single' uses float arithmetic, 'double' uses double.

#include "mex.h"
#include <cmath>
#include <cstring>
#include <vector>
#include "models/idm_step.h"
#include "models/gipps_step.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define LAGMAX 32

template <typename T> inline bool dead_s(T s) { return !std::isfinite((double)s) || s <= (T)0; }

template <typename T>
static void body_idm(const T* sp, const T* lv, const T* evs, const T* len, int L, int N,
                     const T* th, T dt, T* q, int p, int i)
{
    (void)L;
    const T* s = sp + (size_t)i;
    const T* w = lv + (size_t)i;
    const T* v0s = evs + (size_t)i;
    int Tn = (int)len[i];
    T sse = (T)0;
    bool dead = false;

    const cfm::IdmParams<T> par = cfm::idm_params<T>(th);
    T sv = s[0], vv = v0s[0];
    for (int t = 1; t < Tn; ++t) {
        T wv = w[(size_t)N * (t - 1)] - vv;
        T vn = cfm::idm_next_speed(par, sv, vv, w[(size_t)N * (t - 1)], dt);
        T wn = w[(size_t)N * t] - vn;
        T sn = sv + dt * (wv + wn) / (T)2;
        if (dead_s<T>(sn)) { dead = true; break; }
        T e = sn - s[(size_t)N * t];
        sse += e * e;
        sv = sn; vv = vn;
    }
    q[(size_t)p * N + i] = dead ? (T)1e30 : sse / (T)Tn;
}

template <typename T>
static void body_gipps(const T* sp, const T* lv, const T* evs, const T* len, int L, int N,
                       const T* th, T dt, T* q, int p, int i)
{
    (void)L;
    const T* s = sp + (size_t)i;
    const T* w = lv + (size_t)i;
    const T* v0s = evs + (size_t)i;
    int Tn = (int)len[i];
    T sse = (T)0;
    bool dead = false;

    const cfm::GippsParams<T> par = cfm::gipps_params<T>(th);
    int lag = (int)std::ceil((double)par.tau / (double)dt);
    if (lag < 1) lag = 1;
    if (lag > Tn - 1) lag = Tn - 1;
    if (lag > LAGMAX) lag = LAGMAX;
    T hs[LAGMAX], hv[LAGMAX];
    for (int k = 0; k < lag; ++k) { hs[k] = s[(size_t)N * k]; hv[k] = v0s[(size_t)N * k]; }
    for (int t = lag; t < Tn; ++t) {
        int hidx = (t - lag) % lag;
        int pidx = (t - 1) % lag;
        T v_d = hv[hidx], s_d = hs[hidx], lv_d = w[(size_t)N * (t - lag)];
        T v_new = cfm::gipps_next_speed(par, s_d, v_d, lv_d);
        T rel_now = w[(size_t)N * (t - 1)] - hv[pidx];
        T s_new = hs[pidx] + dt * (rel_now + (w[(size_t)N * t] - v_new)) / (T)2;
        if (dead_s<T>(s_new)) { dead = true; break; }
        T e = s_new - s[(size_t)N * t];
        sse += e * e;
        hs[t % lag] = s_new;
        hv[t % lag] = v_new;
    }
    T produced = (Tn > lag) ? (T)(Tn - lag) : (T)1;
    q[(size_t)p * N + i] = dead ? (T)1e30 : sse / produced;
}

template <typename T>
static void run_batch(const T* sp, const T* lv, const T* evs, const T* len, int L, int N, int model,
                      const T* para, T dt, int P, int threads, int reduce, T* q, double* acc)
{
#ifdef _OPENMP
    omp_set_num_threads(threads < 1 ? 1 : threads);
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long pp = 0; pp < (long long)P; ++pp) {
        const int p = (int)pp;
        const T* th = para + (size_t)p * 6;
        for (int i = 0; i < N; ++i) {
            if (model == 0) body_idm<T>(sp, lv, evs, len, L, N, th, dt, q, p, i);
            else            body_gipps<T>(sp, lv, evs, len, L, N, th, dt, q, p, i);
        }
        if (reduce) {
            double se = 0.0, sf = 0.0, wf = 0.0, live = 0.0, dead = 0.0;
            for (int i = 0; i < N; ++i) {
                const T v = q[(size_t)p * N + i];
                if (std::isfinite((double)v) && v < (T)1e6) {
                    const double Lf = (double)len[i];
                    se += (double)v; sf += (double)v * Lf; wf += Lf; live += 1.0;
                } else {
                    dead += 1.0;
                }
            }
            // column major, as the P x 5 output is laid out
            acc[(size_t)p + (size_t)P * 0] = se;  acc[(size_t)p + (size_t)P * 1] = sf;
            acc[(size_t)p + (size_t)P * 2] = wf;  acc[(size_t)p + (size_t)P * 3] = live;
            acc[(size_t)p + (size_t)P * 4] = dead;
        }
    }
}

static double scalar(const mxArray* a, const char* name)
{
    if (!mxIsNumeric(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1)
        mexErrMsgIdAndTxt("batch:scalar", "%s must be a real scalar", name);
    return mxGetScalar(a);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 10)
        mexErrMsgIdAndTxt("batch:nrhs",
                          "ten inputs: theta, model, lvSpd, len, dt, space, evSpd, threads, reduce, single");

    const mxArray* th = prhs[0];
    if (mxGetM(th) != 6) mexErrMsgIdAndTxt("batch:theta", "theta must be 6 x P");
    const int P = (int)mxGetN(th);
    const int model = (int)scalar(prhs[1], "model");
    const int N = (int)mxGetM(prhs[2]);
    const int T = (int)mxGetN(prhs[2]);
    const int L = (int)mxGetM(prhs[3]);
    const double dt = scalar(prhs[4], "dt");
    const int threads = (int)scalar(prhs[7], "threads");
    const int reduce = (int)scalar(prhs[8], "reduce");
    const int single = (int)scalar(prhs[9], "single");

    // one staging block per array: the model headers take the storage type of the arithmetic
    std::vector<float>  fsp, flv, fev, flen, fth;
    std::vector<double> dsp, dlv, dev, dlen, dth;

    if (single) {
        fsp.resize((size_t)N * T); flv.resize((size_t)N * T); fev.resize((size_t)N * T);
        flen.resize(N); fth.resize((size_t)6 * P);
        const double* sp = mxGetPr(prhs[5]); const double* lv = mxGetPr(prhs[2]);
        const double* ev = mxGetPr(prhs[6]); const double* ln = mxGetPr(prhs[3]);
        const double* pa = mxGetPr(th);
        for (size_t k = 0; k < (size_t)N * T; ++k) { fsp[k] = (float)sp[k]; flv[k] = (float)lv[k]; fev[k] = (float)ev[k]; }
        for (int k = 0; k < N; ++k) flen[k] = (float)ln[k];
        for (size_t k = 0; k < (size_t)6 * P; ++k) fth[k] = (float)pa[k];
    } else {
        const double* sp = mxGetPr(prhs[5]);
        dsp.assign(sp, sp + (size_t)N * T);
        const double* lv = mxGetPr(prhs[2]); dlv.assign(lv, lv + (size_t)N * T);
        const double* ev = mxGetPr(prhs[6]); dev.assign(ev, ev + (size_t)N * T);
        const double* ln = mxGetPr(prhs[3]); dlen.assign(ln, ln + N);
        const double* pa = mxGetPr(th); dth.assign(pa, pa + (size_t)6 * P);
    }

    if (reduce) {
        plhs[0] = mxCreateDoubleMatrix(P, 5, mxREAL);
        double* acc = mxGetPr(plhs[0]);
        if (single) {
            std::vector<float> q((size_t)N * P);
            run_batch<float>(fsp.data(), flv.data(), fev.data(), flen.data(), L, N, model,
                             fth.data(), (float)dt, P, threads, 1, q.data(), acc);
        } else {
            std::vector<double> q((size_t)N * P);
            run_batch<double>(dsp.data(), dlv.data(), dev.data(), dlen.data(), L, N, model,
                              dth.data(), dt, P, threads, 1, q.data(), acc);
        }
    } else {
        plhs[0] = mxCreateDoubleMatrix(N, P, mxREAL);
        double* out = mxGetPr(plhs[0]);
        std::vector<double> acc((size_t)5 * P, 0.0);
        std::vector<double> q((size_t)N * P);
        if (single) {
            std::vector<float> qf((size_t)N * P);
            run_batch<float>(fsp.data(), flv.data(), fev.data(), flen.data(), L, N, model,
                             fth.data(), (float)dt, P, threads, 0, qf.data(), acc.data());
            for (size_t k = 0; k < (size_t)N * P; ++k) out[k] = (double)qf[k];
        } else {
            run_batch<double>(dsp.data(), dlv.data(), dev.data(), dlen.data(), L, N, model,
                              dth.data(), dt, P, threads, 0, q.data(), acc.data());
            std::memcpy(out, q.data(), sizeof(double) * (size_t)N * P);
        }
    }
}
