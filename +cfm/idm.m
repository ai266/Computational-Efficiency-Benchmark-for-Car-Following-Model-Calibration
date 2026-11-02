function [s_sim, v_sim] = idm(theta, D, dt)
%IDM  The Intelligent Driver Model, vectorised across events and looping over frames.
%
%   [s_sim, v_sim] = cfm.idm(theta, D, dt)
%
%   theta = [v0, T, a, b, delta, s0]
%
%       s* = s0 + max(0, v*T - v*dv/(2*sqrt(a*b)))
%       a  = a*(1 - (v/v0)^delta - (s*/s)^2)
%       v+ = max(0, v + a*dt)
%       s+ = s + dt*(dv + dv+)/2
%
%   sqrt(a*b) is hoisted out of the frame loop, as in the kernel. No mask is applied per frame: the
%   frame loop runs all T frames and the residual excludes the frames an event does not run.

    if nargin < 3 || isempty(dt), dt = 0.1; end

    N = D.N; T = D.T;
    v0 = theta(1); Th = theta(2); am = theta(3); bb = theta(4); de = theta(5); s0 = theta(6);

    s_sim = zeros(N, T, 'like', D.space);
    v_sim = zeros(N, T, 'like', D.space);

    s  = D.space(:, 1);
    v  = D.evSpd(:, 1);
    dv = D.lvSpd(:, 1) - v;

    s_sim(:, 1) = s;
    v_sim(:, 1) = v;

    sab = sqrt(am * bb);

    for t = 2:T
        sstar = Th .* v - v .* dv ./ (2 * sab);
        sstar = max(0, sstar) + s0;
        acc = am .* (1 - (v ./ v0) .^ de - (sstar ./ s) .^ 2);

        vn = v + acc * dt;
        vn = max(0, vn);

        dvn = D.lvSpd(:, t) - vn;
        sn = s + dt .* (dv + dvn) / 2;
        % No floor on the spacing, as in the kernel: a gap at or below zero produces a non-finite or
        % very large state, and the residual marks that event dead.

        s = sn; v = vn; dv = dvn;
        s_sim(:, t) = s;
        v_sim(:, t) = v;
    end
end
