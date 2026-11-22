// The Gipps model as a single templated function, next to the IDM's.
//
// The driver takes the lower of the speed that free flow allows and the speed that keeps a safe stop
// possible behind a leader braking at a fixed rate. The reaction time is applied by the caller, which
// reads the state of the earlier frame and writes the state of this one.
#ifndef CFM_GIPPS_STEP_H
#define CFM_GIPPS_STEP_H

#ifdef __CUDACC__
#define CFM_HD __host__ __device__
#else
#define CFM_HD
#endif

#include <cmath>

namespace cfm {

template <typename T>
struct GippsParams {
    T an;      // maximum acceleration
    T bn;      // maximum deceleration
    T Sn1;     // standstill gap
    T bl;      // estimated deceleration of the leader
    T V0;      // desired speed
    T tau;     // reaction time
};

template <typename T>
CFM_HD inline GippsParams<T> gipps_params(const T* th)
{
    GippsParams<T> p;
    p.an = th[0]; p.bn = th[1]; p.Sn1 = th[2]; p.bl = th[3]; p.V0 = th[4]; p.tau = th[5];
    return p;
}

template <typename T>
CFM_HD inline T gipps_next_speed(const GippsParams<T>& p, T s, T v, T v_lead)
{
    const T zero(0), one(1), two(2), c025(0.025), c25(2.5);
    const T free_flow = v + c25 * p.an * p.tau * (one - v / p.V0) * sqrt(c025 + v / p.V0);
    T disc = p.bn * p.bn * p.tau * p.tau + p.bn * (two * (s - p.Sn1) - v * p.tau + v_lead * v_lead / p.bl);
    if (disc < zero) disc = zero;
    const T safe = -p.bn * p.tau + sqrt(disc);
    T vn = (free_flow < safe) ? free_flow : safe;
    return (vn < zero) ? zero : vn;
}

}  // namespace cfm

#endif
