// Single-precision transcription of fused_kernel.cpp: the same recursion, the same live mask, the same
// collision rule, the same per-event output. Only the arithmetic type differs.
//   q = fused_kernel_f32(theta, model, lvSpd, len, dt, space, evSpd)
#include "mex.h"
#include <cmath>
#include "models/idm_step.h"
#include "models/gipps_step.h"
#include <cstring>

static inline bool is_dead(float s) { return !std::isfinite(s) || s <= 0.f; }

static double sc(const mxArray* a, const char* name)
{
    if (!mxIsNumeric(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1)
        mexErrMsgIdAndTxt("fused_kernel_f32:type", "%s must be a real numeric scalar", name);
    return mxGetScalar(a);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 7) mexErrMsgTxt("7 inputs required");
    const double* para = mxGetPr(prhs[0]);
    const int model = (int)sc(prhs[1], "model");
    const double* lvSd = mxGetPr(prhs[2]);
    const mwSize N = mxGetM(prhs[2]);
    const mwSize T = mxGetN(prhs[2]);
    const double* lend = mxGetPr(prhs[3]);
    const double dt = sc(prhs[4], "dt");
    const double* spaced = mxGetPr(prhs[5]);
    const double* evSpd = mxGetPr(prhs[6]);

    plhs[0] = mxCreateDoubleMatrix(N, 1, mxREAL);
    double* q = mxGetPr(plhs[0]);

    // the arrays in single: taken as they come when the caller already holds them in single, converted
    // once per call otherwise, because the conversion of a large set costs more than the kernel
    const bool preconverted = mxIsSingle(prhs[2]) && mxIsSingle(prhs[5]) && mxIsSingle(prhs[6]);
    float* space; float* lvSpd; float* evS;
    if (preconverted) {
        space = (float*)mxGetData(prhs[5]);
        lvSpd = (float*)mxGetData(prhs[2]);
        evS   = (float*)mxGetData(prhs[6]);
    } else {
        space = (float*)mxMalloc(sizeof(float) * (size_t)N * T);
        lvSpd = (float*)mxMalloc(sizeof(float) * (size_t)N * T);
        evS   = (float*)mxMalloc(sizeof(float) * (size_t)N * T);
        for (size_t k = 0; k < (size_t)N * T; ++k) {
            space[k] = (float)spaced[k];
            lvSpd[k] = (float)lvSd[k];
            evS[k]   = (float)evSpd[k];
        }
    }

    if (model == 0) {
        const float v0 = (float)para[0], Th = (float)para[1], am = (float)para[2], bb = (float)para[3];
        const float de = (float)para[4], s0 = (float)para[5];
        const float sab = (float)std::sqrt((double)am * (double)bb);
        float* sb = (float*)mxMalloc(sizeof(float) * N);
        float* vb = (float*)mxMalloc(sizeof(float) * N);
        float* ap = (float*)mxCalloc(N, sizeof(float));
        char* collided = (char*)mxCalloc(N, sizeof(char));
        for (mwSize i = 0; i < N; ++i) { sb[i] = space[i]; vb[i] = evS[i]; }
        float* sp = sb; float* vp = vb;
        {
        for (mwSize t = 1; t < T; ++t) {
            const mwSize off = N * t;
            for (long long ii = 0; ii < (long long)N; ++ii) {
                const mwSize i = (mwSize)ii;
                const float live = ((float)(t + 1) <= (float)lend[i]) ? 1.f : 0.f;
                float sv = sp[i], vv = vp[i];
                float wv = lvSpd[i + N * (t - 1)] - vv;
                float u = Th * vv - vv * wv / (2.f * sab);
                float sstar = (u > 0.f ? u : 0.f) + s0;
                float acc = am * (1.f - (float)std::pow((double)(vv / v0), (double)de) - (sstar / sv) * (sstar / sv));
                float vn = vv + acc * dt;
                if (vn < 0.f) vn = 0.f;
                float wn = lvSpd[i + off] - vn;
                float sn = sv + dt * (wv + wn) / 2.f;
                if (live > 0.f && is_dead(sn)) collided[i] = 1;
                sp[i] = live * sn + (1.f - live) * sv;
                vp[i] = live * vn + (1.f - live) * vv;
                float dsn = sn - space[i + off];
                ap[i] += live * dsn * dsn;
            }
        }
        }
        for (mwSize i = 0; i < N; ++i) q[i] = collided[i] ? 1e6 : (double)(ap[i] / (float)lend[i]);
        mxFree(sb); mxFree(vb); mxFree(ap); mxFree(collided);
    } else {
        const float a_n = (float)para[0], b_n = (float)para[1], S_n1 = (float)para[2];
        const float bb = (float)para[3], V_0 = (float)para[4], tau = (float)para[5];
        mwSize lag = (mwSize)std::ceil(tau / dt);
        if (lag < 1) lag = 1;
        if (lag > T - 1) lag = T - 1;
        const size_t fr = (size_t)N * (size_t)T;
        float* sh = (float*)mxMalloc(sizeof(float) * fr);
        float* vh = (float*)mxMalloc(sizeof(float) * fr);
        float* ap = (float*)mxCalloc(N, sizeof(float));
        char* collided = (char*)mxCalloc(N, sizeof(char));
        for (mwSize t = 0; t < lag; ++t) {
            const mwSize off = N * t;
            for (mwSize i = 0; i < N; ++i) { sh[i + off] = space[i + off]; vh[i + off] = evS[i + off]; }
        }
        {
        for (mwSize t = lag; t < T; ++t) {
            const mwSize off = N * t, offp = N * (t - 1), hoff = N * (t - lag);
            for (long long ii = 0; ii < (long long)N; ++ii) {
                const mwSize i = (mwSize)ii;
                const float live = ((float)(t + 1) <= (float)lend[i]) ? 1.f : 0.f;
                float v_d = vh[i + hoff], s_d = sh[i + hoff], lv_d = lvSpd[i + hoff];
                float v_free = v_d + 2.5f * a_n * tau * (1.f - v_d / V_0) * (float)std::sqrt((double)(0.025f + v_d / V_0));
                float disc = b_n * b_n * tau * tau + b_n * (2.f * (s_d - S_n1) - v_d * tau + lv_d * lv_d / bb);
                if (disc < 0.f) disc = 0.f;
                float v_safe = -b_n * tau + (float)std::sqrt((double)disc);
                float v_new = (v_free < v_safe) ? v_free : v_safe;
                if (v_new < 0.f) v_new = 0.f;
                float v_lead = lvSpd[i + off];
                float rel_now = lvSpd[i + offp] - vh[i + offp];
                float s_new = sh[i + offp] + dt * (rel_now + (v_lead - v_new)) / 2.f;
                if (live > 0.f && is_dead(s_new)) collided[i] = 1;
                sh[i + off] = live * s_new + (1.f - live) * sh[i + offp];
                vh[i + off] = live * v_new + (1.f - live) * vh[i + offp];
                float dsn = s_new - space[i + off];
                ap[i] += live * dsn * dsn;
            }
        }
        }
        for (mwSize i = 0; i < N; ++i) {
            const float produced = (lend[i] > (double)lag) ? (float)(lend[i] - (double)lag) : 1.f;
            q[i] = collided[i] ? 1e6 : (double)(ap[i] / produced);
        }
        mxFree(sh); mxFree(vh); mxFree(ap); mxFree(collided);
    }
    if (!preconverted) { mxFree(space); mxFree(lvSpd); mxFree(evS); }
}
