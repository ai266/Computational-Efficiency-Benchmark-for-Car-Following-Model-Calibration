function f = objective(theta, D, o)
%OBJECTIVE  One evaluation of the calibration objective.
%
%   f = cfm.objective(theta, D)
%   f = cfm.objective(theta, D, o)      o from cfm.config
%
%   Root mean squared spacing error over the pooled event set: per event the mean squared error over
%   its own frames, averaged over the events that produced a finite trajectory, one square root on that
%   mean.
%
%       e_i = (1 / L_i) * sum_t (s_sim - s_obs)^2        [m^2]
%       f   = pool_i ( e_i ) + 1e6 * n_dead              [m]
%
%   `o.pool` decides the pooling: 'event' gives each event the same weight and takes the square root of
%   their mean; 'frame' weights each event by the frames it produced.
%
%   An event whose simulated spacing is not finite, or has reached zero, is a failure: it leaves the
%   RMSE, whose square root would turn it into a large spacing error, and re-enters as a counted
%   penalty. A run of collisions therefore scores in millions, not in hundreds of metres.

    if nargin < 3 || isempty(o), o = cfm.config(); end
    o = fill(o, D);
    theta = theta(:);

    lag = frames_given(o, theta, D);
    if o.block >= D.N
        f = score(one_call(theta, D, o), D.len, lag, o.pool);
        return
    end

    E = zeros(D.N, 1);
    for i1 = 1:o.block:D.N
        i2 = min(i1 + o.block - 1, D.N);
        E(i1:i2) = one_call(theta, slice(D, i1:i2), o);
    end
    f = score(E, D.len, lag, o.pool);
end

function f = score(E, len, lag, pool)
%SCORE  The objective over the events that produced a finite trajectory.
%
%   A failed event, whether a collision or a numerical failure, is not part of the RMSE: its square root
%   would masquerade as a large error and swamp the events that behave. The failures are counted and
%   charged separately, so one collision cannot be confused with a bad fit.
%
%   `pool` decides how the remaining per-event errors are averaged. 'event' gives every event the same
%   weight, which is the rule the paper reports; 'frame' weights an event by the frames it produced, so a
%   long event counts for more. The two differ only in how much a long event is worth, never in which
%   events are scored.
    dead = E >= 1e6;
    live = ~dead;
    if any(live)
        switch lower(pool)
            case 'event'
                f = sqrt(mean(E(live)));
            case 'frame'
                L = max(len(:) - lag, 1);
                f = sqrt(sum(E(live) .* L(live)) / sum(L(live)));
            otherwise
                error('objective:pool', 'pool must be ''event'' or ''frame''');
        end
    else
        f = 0;
    end
    f = f + 1e6 * sum(dead);
end

function e = one_call(theta, D, o)
    if o.compile
        if isnan(o.kernel)
            error('objective:compiled', ...
                  'model %s has no compiled branch; run it with compile=false', o.model);
        end
        persistent kernel_checked      % one build check per session, not one per evaluation
        if isempty(kernel_checked)
            build_kernel('ensure');
            kernel_checked = true;
        end
        e = fused_kernel(theta, o.kernel, D.lvSpd, D.len, D.dt, D.space, o.threads, D.evSpd);
        return
    end
    [s_sim, v_sim] = o.rollout(theta, D, D.dt);
    e = residual(D, s_sim, v_sim, o, frames_given(o, theta, D));
end

function e = residual(D, s_sim, v_sim, o, lag)
%RESIDUAL  Per-event mean squared spacing error, over the frames the model produced.
%
%   Gipps spends its first ceil(tau/dt) frames reading the observed record, so those frames are not
%   predictions and are not scored; the denominator is the number of frames it produced. IDM reacts at
%   the current frame, so lag is zero and it is scored over the whole event.
    live  = ((1:D.T) > lag) & ((1:D.T) <= D.len(:));
    dead  = ~isfinite(s_sim) | (s_sim <= 0);     % the same test as is_dead() in the kernel
    d2    = (s_sim - D.space) .^ 2;
    scored = max(D.len(:) - lag, 1);
    e     = sum(d2 .* live, 2) ./ scored;
    e(any(dead & live, 2)) = 1e6;
end

function lag = frames_given(o, theta, D)
%FRAMES_GIVEN  How many leading frames are the observed record rather than a prediction.
    lag = 0;
    if strcmpi(o.model, 'gipps')
        lag = max(1, ceil(theta(6) / D.dt));
        lag = min(lag, D.T - 1);
    end
end

function Dk = slice(D, idx)
%SLICE  The data contract over a subset of the events.
    Dk = struct('space', D.space(idx, :), 'evSpd', D.evSpd(idx, :), ...
                'lvSpd', D.lvSpd(idx, :), 'len', D.len(idx), ...
                'N', numel(idx), 'T', D.T, 'dt', D.dt, 'name', D.name);
end

function o = fill(o, D)
%FILL  Fields the caller left out take their default, and the two that depend on the data are set.
    base = cfm.config();
    f = fieldnames(base);
    for k = 1:numel(f)
        if ~isfield(o, f{k}) || isempty(o.(f{k}))
            o.(f{k}) = base.(f{k});
        end
    end
    if isempty(o.block), o.block = D.N; end
end
