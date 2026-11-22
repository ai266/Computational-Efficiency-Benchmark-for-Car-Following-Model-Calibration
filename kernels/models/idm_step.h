// The Intelligent Driver Model as a single templated function.
//
// One implementation serves the CPU kernels (float and double) and the device kernels, so a change to
// the physics is a change in one file rather than in four. The caller supplies the state of the frame
// it is at and the speed of the leader, and receives the action of the frame, in the units of the run.
#ifndef CFM_IDM_STEP_H
#define CFM_IDM_STEP_H

#ifdef __CUDACC__
#define CFM_HD __host__ __device__
#else
#define CFM_HD
#endif

#include <cmath>

namespace cfm {

template <typename T>
struct IdmParams {
    T v0;      // desired speed
    T Th;      // desired time headway
    T a;       // maximum acceleration
    T b;       // comfortable deceleration
    T delta;   // free-flow exponent
    T s0;      // standstill gap
};

template <typename T>
CFM_HD inline IdmParams<T> idm_params(const T* th)
{
    IdmParams<T> p;
    p.v0 = th[0]; p.Th = th[1]; p.a = th[2]; p.b = th[3]; p.delta = th[4]; p.s0 = th[5];
    return p;
}

// The acceleration at one frame: the free-flow term relaxes toward the desired speed, the interaction
// term grows as the gap falls below the desired gap, and a leader faster than the follower enlarges that
// gap through the speed difference.
template <typename T>
CFM_HD inline T idm_acceleration(const IdmParams<T>& p, T s, T v, T v_lead)
{
    const T two(2), one(1), zero(0);
    const T sab = sqrt(p.a * p.b);
    const T dv = v_lead - v;
    T u = p.Th * v - v * dv / (two * sab);
    if (u < zero) u = zero;
    const T sstar = u + p.s0;
    const T ratio = v / p.v0;
    return p.a * (one - pow(ratio, p.delta) - (sstar / s) * (sstar / s));
}

// The speed of the next frame, never negative.
template <typename T>
CFM_HD inline T idm_next_speed(const IdmParams<T>& p, T s, T v, T v_lead, T dt)
{
    const T zero(0);
    T vn = v + idm_acceleration(p, s, v, v_lead) * dt;
    return (vn < zero) ? zero : vn;
}

}  // namespace cfm

#endif
