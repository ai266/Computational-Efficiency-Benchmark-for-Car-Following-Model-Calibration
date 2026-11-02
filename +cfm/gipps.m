function [s_sim, v_sim] = gipps(theta, D, dt)
%GIPPS  The Gipps car-following model.
%
%   [s_sim, v_sim] = cfm.gipps(theta, D, dt)
%
%   theta = [a_n, b_n, S_n1, b, V_0, tau]
%     a_n   largest acceleration the driver accepts
%     b_n   largest deceleration the driver accepts
%     S_n1  effective gap at standstill
%     b     estimate of the leader's deceleration
%     V_0   desired speed
%     tau   reaction time
%
%   Two bounds per step, the lower one taken: the free bound rises toward V_0, the safe bound keeps the
%   follower able to stop behind a leader braking at b.
%
%   The reaction time is a delay of ceil(tau/dt) frames on every state the model reads. The first `lag`
%   frames are therefore the observed record, not a simulation: a driver at frame lag+1 is reacting to
%   what the two vehicles were doing at frame 1, and that is data. Simulation starts at frame lag+1, and
%   every simulated frame reads strictly earlier frames.

    if nargin < 3 || isempty(dt), dt = 0.1; end

    a_n = theta(1); b_n = theta(2); S_n1 = theta(3);
    b   = theta(4); V_0 = theta(5); tau   = theta(6);

    N = D.N; T = D.T;
    s_sim = zeros(N, T); v_sim = zeros(N, T);

    lag = max(1, ceil(tau / dt));
    lag = min(lag, T - 1);

    % Frames 1..lag are the observed record: the model has no history of its own yet.
    s_sim(:, 1:lag) = D.space(:, 1:lag);
    v_sim(:, 1:lag) = D.evSpd(:, 1:lag);

    s_hist = zeros(N, T); v_hist = zeros(N, T);
    s_hist(:, 1:lag) = D.space(:, 1:lag);
    v_hist(:, 1:lag) = D.evSpd(:, 1:lag);
    lv_hist = D.lvSpd;

    s = s_sim(:, lag); v = v_sim(:, lag);
    for t = lag+1:T
        h = t - lag;                % the frames the model reads are the delayed ones
        v_d = v_hist(:, h);
        s_d = s_hist(:, h);
        lv_d = lv_hist(:, h);

        v_free = v_d + 2.5 * a_n * tau .* (1 - v_d ./ V_0) .* sqrt(0.025 + v_d ./ V_0);
        disc = b_n^2 * tau^2 + b_n .* (2 * (s_d - S_n1) - v_d * tau + lv_d .^ 2 / b);
        disc = max(disc, 0);
        v_safe = -b_n * tau + sqrt(disc);

        v_new = min(v_free, v_safe);
        v_new = max(0, v_new);

        v_lead = D.lvSpd(:, t);
        % Only the speeds the model reads are delayed; the integration runs on the live state.
        rel_now = D.lvSpd(:, t - 1) - v;
        s_new = s + dt * (rel_now + (v_lead - v_new)) / 2;

        v = v_new; s = s_new;
        v_sim(:, t) = v; s_sim(:, t) = s;
        v_hist(:, t) = v; s_hist(:, t) = s;
    end
end
