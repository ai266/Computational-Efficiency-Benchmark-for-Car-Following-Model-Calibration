// Double-precision build of the batched device kernel, used to price single precision on the device.
//   A = gpu_kernel_batch64(theta, model, lvSpd, len, dt, space, evSpd)
// theta is d x P; A is P x 5 = [sum of per-event mean squares, weighted sum, frames, live, dead].
// The state of a trajectory lives in the thread that simulates it, so no trajectory is written out.
#include "mex.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include "models/idm_step.h"
#include "models/gipps_step.h"

#define LAGMAX 32

__device__ __forceinline__ bool dead_s(double s) { return !isfinite(s) || s <= 0.f; }

__global__ __launch_bounds__(128, 8) void batch_idm(const double* sp, const double* lv, const double* evs,
                                                   const double* len, int L, int N,
                                                   const double* para, double dt, double* q)
{
    int p = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double* th = para + (size_t)p * 6;
    const double* s = sp + (size_t)i;
    const double* w = lv + (size_t)i;
    const double* v0s = evs + (size_t)i;
    int T = (int)len[i];
    double sse = 0.f;
    bool dead = false;

    {
        const cfm::IdmParams<double> par = cfm::idm_params<double>(th);
        double sv = s[0], vv = v0s[0];
        for (int t = 1; t < T; ++t) {
            double wv = w[(size_t)N * (t - 1)] - vv;
            double vn = cfm::idm_next_speed(par, sv, vv, w[(size_t)N * (t - 1)], dt);
            double wn = w[(size_t)N * t] - vn;
            double sn = sv + dt * (wv + wn) / 2.f;
            if (dead_s(sn)) { dead = true; break; }
            double e = sn - s[(size_t)N * t];
            sse += e * e;
            sv = sn; vv = vn;
        }
        q[(size_t)p * N + i] = dead ? 1e30 : sse / (double)T;
    }
}

__global__ __launch_bounds__(128, 8) void batch_gipps(const double* sp, const double* lv, const double* evs,
                                                     const double* len, int L, int N,
                                                     const double* para, double dt, double* q)
{
    int p = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double* th = para + (size_t)p * 6;
    const double* s = sp + (size_t)i;
    const double* w = lv + (size_t)i;
    const double* v0s = evs + (size_t)i;
    int T = (int)len[i];
    double sse = 0.f;
    bool dead = false;
    {
        const cfm::GippsParams<double> par = cfm::gipps_params<double>(th);
        const double an = par.an, bn = par.bn, Sn1 = par.Sn1, bl = par.bl, V0 = par.V0, tau = par.tau;
        int lag = (int)ceil(tau / dt);
        if (lag < 1) lag = 1;
        if (lag > T - 1) lag = T - 1;
        if (lag > LAGMAX) lag = LAGMAX;
        double hs[LAGMAX], hv[LAGMAX];      // the history this thread produces, newest last
        for (int k = 0; k < lag; ++k) { hs[k] = s[(size_t)N * k]; hv[k] = v0s[(size_t)N * k]; }
        for (int t = lag; t < T; ++t) {
            int hidx = (t - lag) % lag;
            int pidx = (t - 1) % lag;
            double v_d = hv[hidx], s_d = hs[hidx], lv_d = w[(size_t)N * (t - lag)];
            double v_new = cfm::gipps_next_speed(par, s_d, v_d, lv_d);
            double rel_now = w[(size_t)N * (t - 1)] - hv[pidx];
            double s_new = hs[pidx] + dt * (rel_now + (w[(size_t)N * t] - v_new)) / 2.f;
            if (dead_s(s_new)) { dead = true; break; }
            double e = s_new - s[(size_t)N * t];
            sse += e * e;
            hs[t % lag] = s_new;
            hv[t % lag] = v_new;
        }
        double produced = (T > lag) ? (double)(T - lag) : 1.f;
        q[(size_t)p * N + i] = dead ? 1e30 : sse / produced;
    }
}

// One block per candidate: the reduction of that candidate's events, five numbers out.
__global__ void reduce_p(const double* q, const double* len, int N, double* acc)
{
    __shared__ double sm[5][256];
    int p = blockIdx.x;
    int P = gridDim.x;
    int t = threadIdx.x;
    const double* qp = q + (size_t)p * N;
    double se = 0, sf = 0, wf = 0, live = 0, dead = 0;
    for (int i = t; i < N; i += blockDim.x) {
        double v = qp[i];
        if (isfinite(v) && v < 1e6f) {
            double L = (double)len[i];
            se += (double)v; sf += (double)v * L; wf += L; live += 1.0;
        } else {
            dead += 1.0;
        }
    }
    sm[0][t] = se; sm[1][t] = sf; sm[2][t] = wf; sm[3][t] = live; sm[4][t] = dead;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (t < step) for (int k = 0; k < 5; ++k) sm[k][t] += sm[k][t + step];
        __syncthreads();
    }
    if (t == 0) for (int k = 0; k < 5; ++k) acc[(size_t)k * P + p] = sm[k][0];
}

static void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) mexErrMsgIdAndTxt("gpu_kernel_batch64:cuda", "%s: %s", what, cudaGetErrorString(e));
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 7) mexErrMsgTxt("7 inputs required: theta, model, lvSpd, len, dt, space, evSpd");
    const int do_reduce = (nrhs >= 8) ? (int)mxGetScalar(prhs[7]) : 1;   // 0 leaves the pooling to the caller
    const int resident  = (nrhs >= 9) ? (int)mxGetScalar(prhs[8]) : 1;   // 0 re-uploads on every call
    const double* thd = mxGetPr(prhs[0]);
    int d = (int)mxGetM(prhs[0]);
    int P = (int)mxGetN(prhs[0]);
    if (d != 6) mexErrMsgTxt("theta must be 6 x P");
    int model = (int)mxGetScalar(prhs[1]);
    const double* lvSd = mxGetPr(prhs[2]);
    mwSize N = mxGetM(prhs[2]);
    mwSize L = mxGetN(prhs[2]);
    const double* lend = mxGetPr(prhs[3]);
    double dt = mxGetScalar(prhs[4]);
    const double* spaced = mxGetPr(prhs[5]);
    const double* evSpd = mxGetPr(prhs[6]);

    // Two resident sets, one per split of a run: the training and the test set stay on the device at the
    // same time, so scoring the test set after the clock does not evict the training set.
    struct Slot { const void* ptr; mwSize N, L; double* sp; double* lv; double* evs; double* len; };
    static Slot slot[2];
    static int slotReady = 0;
    static int slotUse = 0;
    static double* d_acc = nullptr;
    static double* d_para = nullptr; static int cD = 0, cP = 0;

    ck(cudaSetDevice(0), "setDevice");
    const void* hostptr = (const void*)mxGetData(prhs[5]);   // the spacing array identifies the split
    int use = -1;
    for (int k = 0; k < 2; ++k) {
        if (slot[k].sp && slot[k].ptr == hostptr && slot[k].N == N && slot[k].L == L) {
            bool alive = false;
            cudaPointerAttributes a;
            alive = (cudaPointerGetAttributes(&a, slot[k].sp) == cudaSuccess);
            if (alive) { use = k; break; }
            slot[k].sp = nullptr;
        }
    }
    if (use < 0 && !resident) {
        use = slotUse;                                   // the residency arm: the set is not kept
        slotUse = (slotUse + 1) % 2;
        if (slot[use].sp) {
            cudaFree(slot[use].sp); cudaFree(slot[use].lv); cudaFree(slot[use].evs); cudaFree(slot[use].len);
            slot[use].sp = nullptr;
        }
    } else if (use < 0) {
        use = (slotReady < 2) ? slotReady : 0;            // a new split takes the free slot, then the first
        if (slotReady < 2) ++slotReady;
        if (slot[use].sp) {
            cudaFree(slot[use].sp); cudaFree(slot[use].lv); cudaFree(slot[use].evs); cudaFree(slot[use].len);
            slot[use].sp = nullptr;
        }
    }
    if (!slot[use].sp) {
        size_t n = (size_t)N * (size_t)L;
        double* h1 = (double*)mxMalloc(sizeof(double) * n);
        double* h2 = (double*)mxMalloc(sizeof(double) * n);
        double* h3 = (double*)mxMalloc(sizeof(double) * n);
        for (size_t k = 0; k < n; ++k) { h1[k] = (double)spaced[k]; h2[k] = (double)lvSd[k]; h3[k] = (double)evSpd[k]; }
        ck(cudaMalloc(&slot[use].sp, sizeof(double) * n), "malloc space");
        ck(cudaMalloc(&slot[use].lv, sizeof(double) * n), "malloc lv");
        ck(cudaMalloc(&slot[use].evs, sizeof(double) * n), "malloc ev");
        ck(cudaMalloc(&slot[use].len, sizeof(double) * N), "malloc len");
        ck(cudaMemcpy(slot[use].sp, h1, sizeof(double) * n, cudaMemcpyHostToDevice), "copy space");
        ck(cudaMemcpy(slot[use].lv, h2, sizeof(double) * n, cudaMemcpyHostToDevice), "copy lv");
        ck(cudaMemcpy(slot[use].evs, h3, sizeof(double) * n, cudaMemcpyHostToDevice), "copy ev");
        double* hl = (double*)mxMalloc(sizeof(double) * N);
        for (mwSize i = 0; i < N; ++i) hl[i] = (double)lend[i];
        ck(cudaMemcpy(slot[use].len, hl, sizeof(double) * N, cudaMemcpyHostToDevice), "copy len");
        mxFree(h1); mxFree(h2); mxFree(h3); mxFree(hl);
        slot[use].ptr = hostptr; slot[use].N = N; slot[use].L = L;
    }
    double* d_sp = slot[use].sp; double* d_lv = slot[use].lv;
    double* d_evs = slot[use].evs; double* d_len = slot[use].len;

    static double* d_q = nullptr; static int cQN = 0, cQP = 0;
    if (cQN != (int)N || cQP < P) {
        if (d_q) cudaFree(d_q);
        ck(cudaMalloc(&d_q, sizeof(double) * (size_t)N * P), "malloc q");
        cQN = (int)N; cQP = P;
    }
    if (!d_para || cD != d || cP < P) {
        if (d_para) cudaFree(d_para);
        ck(cudaMalloc(&d_para, sizeof(double) * 6 * P), "malloc para");
        cD = d; cP = P;
    }
    static int cAccP = 0;
    if (!d_acc || cAccP < P) {
        if (d_acc) cudaFree(d_acc);
        ck(cudaMalloc(&d_acc, sizeof(double) * 5 * P), "malloc acc");
        cAccP = P;
    }

    double* hp = (double*)mxMalloc(sizeof(double) * 6 * P);
    for (int k = 0; k < 6 * P; ++k) hp[k] = (double)thd[k];
    ck(cudaMemcpy(d_para, hp, sizeof(double) * 6 * P, cudaMemcpyHostToDevice), "copy para");
    mxFree(hp);

    int threads = 128;
    dim3 blocks((unsigned)((N + threads - 1) / threads), (unsigned)P, 1);
    if (model == 0)
        batch_idm<<<blocks, threads>>>(d_sp, d_lv, d_evs, d_len, (int)L, (int)N, d_para, (double)dt, d_q);
    else
        batch_gipps<<<blocks, threads>>>(d_sp, d_lv, d_evs, d_len, (int)L, (int)N, d_para, (double)dt, d_q);
    ck(cudaGetLastError(), "batch kernel");
    reduce_p<<<(unsigned)P, 256>>>(d_q, d_len, (int)N, d_acc);
    ck(cudaGetLastError(), "reduce");

    if (do_reduce) {
        plhs[0] = mxCreateDoubleMatrix(P, 5, mxREAL);
        ck(cudaMemcpy(mxGetPr(plhs[0]), d_acc, sizeof(double) * 5 * P, cudaMemcpyDeviceToHost), "copy acc");
    } else {
        plhs[0] = mxCreateNumericMatrix((mwSize)N, (mwSize)P, mxDOUBLE_CLASS, mxREAL);
        ck(cudaMemcpy(mxGetData(plhs[0]), d_q, sizeof(double) * (size_t)N * P, cudaMemcpyDeviceToHost),
           "copy events");
    }
}
