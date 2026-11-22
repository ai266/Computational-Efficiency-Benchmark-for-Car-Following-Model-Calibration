// GPU kernel, single precision, device buffers resident across calls.
//   E = gpu_kernel_res(theta, model, lvSpd, len, dt, space, evSpd)
// The event set is uploaded on the first call and kept on the device afterwards, so a call transfers six
// numbers and returns one vector. The recursion is the one of fused_kernel.cpp, including the speed clamp,
// the live mask and the collision rule.
#include "mex.h"
#include <cuda_runtime.h>
#include <cmath>
#include "models/idm_step.h"
#include "models/gipps_step.h"

__device__ __forceinline__ bool dead(float s) { return !isfinite(s) || s <= 0.f; }

__global__ void idm_k(const float* sp, const float* lv, const float* evs, const float* len, int L, int N,
                      float v0, float Th, float am, float bb, float de, float s0, float dt, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* s = sp + (size_t)i;
    const float* w = lv + (size_t)i;
    const float* v0s = evs + (size_t)i;
    int T = (int)len[i];
    cfm::IdmParams<float> par;
    par.v0 = v0; par.Th = Th; par.a = am; par.b = bb; par.delta = de; par.s0 = s0;
    float sv = s[0], vv = v0s[0], sse = 0.f;
    for (int t = 1; t < T; ++t) {
        const float v_lead = w[(size_t)N * (t - 1)];
        const float wv = v_lead - vv;
        const float vn = cfm::idm_next_speed(par, sv, vv, v_lead, dt);
        float wn = w[(size_t)N * t] - vn;
        float sn = sv + dt * (wv + wn) / 2.f;
        if (dead(sn)) { out[i] = 1e6f; return; }
        float e = sn - s[(size_t)N * t];
        sse += e * e;
        sv = sn; vv = vn;
    }
    out[i] = sse / (float)T;
}

__global__ void gipps_k(const float* sp, const float* lv, const float* evs, const float* len, int L, int N,
                        float an, float bn, float Sn1, float bl, float V0, float tau, int lag,
                        float dt, float* sh, float* vh, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* s = sp + (size_t)i;
    const float* w = lv + (size_t)i;
    const float* v0s = evs + (size_t)i;
    int T = (int)len[i];
    float sse = 0.f;
    for (int t = lag; t < T; ++t) {
        float v_d = vh[i + (size_t)N * (t - lag)], s_d = sh[i + (size_t)N * (t - lag)];
        float lv_d = w[(size_t)N * (t - lag)];
        cfm::GippsParams<float> gpar;
        gpar.an = an; gpar.bn = bn; gpar.Sn1 = Sn1; gpar.bl = bl; gpar.V0 = V0; gpar.tau = tau;
        float v_new = cfm::gipps_next_speed(gpar, s_d, v_d, lv_d);
        float rel_now = w[(size_t)N * (t - 1)] - vh[i + (size_t)N * (t - 1)];
        float s_new = sh[i + (size_t)N * (t - 1)] + dt * (rel_now + (w[(size_t)N * t] - v_new)) / 2.f;
        if (dead(s_new)) { out[i] = 1e6f; return; }
        float e = s_new - s[(size_t)N * t];
        sse += e * e;
        sh[i + (size_t)N * t] = s_new;
        vh[i + (size_t)N * t] = v_new;
    }
    float produced = (T > lag) ? (float)(T - lag) : 1.f;
    out[i] = sse / produced;
}

__global__ void seed_hist(const float* sp, const float* evs, float* sh, float* vh, int N, int lag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * lag;
    if (idx >= total) return;
    int i = idx % N;            // event
    int t = idx / N;            // frame, column-major
    sh[i + (size_t)N * t] = sp[i + (size_t)N * t];
    vh[i + (size_t)N * t] = evs[i + (size_t)N * t];
}


// Single-block reduction of the per-event mean squares: deterministic, no atomics, five numbers back.
__global__ void reduce_one(const float* q, const float* len, int N, double* out)
{
    __shared__ double sm[5][256];
    int t = threadIdx.x;
    double se = 0, sf = 0, wf = 0, live = 0, dead = 0;
    for (int i = t; i < N; i += blockDim.x) {
        float v = q[i];
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
    if (t == 0) for (int k = 0; k < 5; ++k) out[k] = sm[k][0];
}

static void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) mexErrMsgIdAndTxt("gpu_kernel_res:cuda", "%s: %s", what, cudaGetErrorString(e));
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 7) mexErrMsgTxt("7 inputs required: theta, model, lvSpd, len, dt, space, evSpd");
    const double* th = mxGetPr(prhs[0]);
    int model = (int)mxGetScalar(prhs[1]);
    const double* lvSd = mxGetPr(prhs[2]);
    mwSize N = mxGetM(prhs[2]);
    mwSize L = mxGetN(prhs[2]);
    const double* lend = mxGetPr(prhs[3]);
    double dt = mxGetScalar(prhs[4]);
    const double* spaced = mxGetPr(prhs[5]);
    const double* evSpd  = mxGetPr(prhs[6]);

    static float* d_sp = nullptr; static float* d_lv = nullptr; static float* d_len = nullptr;
    static float* d_out = nullptr; static float* d_evs = nullptr;
    static float* d_sh = nullptr; static float* d_vh = nullptr;
    static mwSize cN = 0, cL = 0;
    static const void* cPtr = nullptr;

    ck(cudaSetDevice(0), "setDevice");
    // a pointer that the context no longer owns is rebuilt rather than reused
    bool valid = false;
    if (d_sp) {
        cudaPointerAttributes attr;
        valid = (cudaPointerGetAttributes(&attr, d_sp) == cudaSuccess);
    }
    if (!valid) { d_sp = nullptr; cN = 0; cL = 0; cPtr = nullptr; }
    const void* hostptr = (const void*)mxGetData(prhs[5]);   // the spacing array identifies the split
    if (cN != N || cL != L || cPtr != hostptr || d_sp == nullptr) {
        if (d_sp) { cudaFree(d_sp); cudaFree(d_lv); cudaFree(d_len); cudaFree(d_out); cudaFree(d_evs); cudaFree(d_sh); cudaFree(d_vh); }
        size_t n = (size_t)N * (size_t)L;
        float* h1 = (float*)mxMalloc(sizeof(float) * n);
        float* h2 = (float*)mxMalloc(sizeof(float) * n);
        float* h3 = (float*)mxMalloc(sizeof(float) * n);
        for (size_t k = 0; k < n; ++k) { h1[k] = (float)spaced[k]; h2[k] = (float)lvSd[k]; h3[k] = (float)evSpd[k]; }
        ck(cudaMalloc(&d_sp, sizeof(float) * n), "malloc space");
        ck(cudaMalloc(&d_lv, sizeof(float) * n), "malloc lv");
        ck(cudaMalloc(&d_evs, sizeof(float) * n), "malloc ev");
        ck(cudaMalloc(&d_len, sizeof(float) * N), "malloc len");
        ck(cudaMalloc(&d_out, sizeof(float) * N), "malloc out");
        ck(cudaMalloc(&d_sh, sizeof(float) * n), "malloc sh");
        ck(cudaMalloc(&d_vh, sizeof(float) * n), "malloc vh");
        ck(cudaMemcpy(d_sp, h1, sizeof(float) * n, cudaMemcpyHostToDevice), "copy space");
        ck(cudaMemcpy(d_lv, h2, sizeof(float) * n, cudaMemcpyHostToDevice), "copy lv");
        ck(cudaMemcpy(d_evs, h3, sizeof(float) * n, cudaMemcpyHostToDevice), "copy ev");
        // the initial speed of every event is its first frame; kept on the device as the seed state
        float* hl = (float*)mxMalloc(sizeof(float) * N);
        for (mwSize i = 0; i < N; ++i) hl[i] = (float)lend[i];
        ck(cudaMemcpy(d_len, hl, sizeof(float) * N, cudaMemcpyHostToDevice), "copy len");
        mxFree(h1); mxFree(h2); mxFree(h3); mxFree(hl);
        cN = N; cL = L; cPtr = hostptr;
    }

    static double* d_acc = nullptr;
    if (!d_acc) ck(cudaMalloc(&d_acc, sizeof(double) * 5), "malloc acc");

    int threads = 128;
    int blocks = (int)((N + threads - 1) / threads);
    float t0 = (float)th[0], t1 = (float)th[1], t2 = (float)th[2];
    float t3 = (float)th[3], t4 = (float)th[4], t5 = (float)th[5];
    if (model == 0) {
        idm_k<<<blocks, threads>>>(d_sp, d_lv, d_evs, d_len, (int)L, (int)N, t0, t1, t2, t3, t4, t5, (float)dt, d_out);
    } else {
        int lag = (int)std::ceil(t5 / dt);
        if (lag < 1) lag = 1;
        if (lag > (int)L - 1) lag = (int)L - 1;
        int total = (int)N * lag;
        int sblocks = (total + threads - 1) / threads;
        seed_hist<<<sblocks, threads>>>(d_sp, d_evs, d_sh, d_vh, (int)N, lag);
        gipps_k<<<blocks, threads>>>(d_sp, d_lv, d_evs, d_len, (int)L, (int)N, t0, t1, t2, t3, t4, t5, lag,
                                     (float)dt, d_sh, d_vh, d_out);
    }
    ck(cudaGetLastError(), "launch");
    reduce_one<<<1, 256>>>(d_out, d_len, (int)N, d_acc);
    ck(cudaGetLastError(), "reduce");
    plhs[0] = mxCreateDoubleMatrix(5, 1, mxREAL);
    ck(cudaMemcpy(mxGetPr(plhs[0]), d_acc, sizeof(double) * 5, cudaMemcpyDeviceToHost), "copy acc");
}
