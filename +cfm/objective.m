function f = objective(theta, D, o)
%OBJECTIVE  One evaluation of the calibration objective.
%
%   f = cfm.objective(theta, D)
%   f = cfm.objective(theta, D, o)      o from cfm.config
%
%   Root mean squared spacing error over the pooled event set: per event the mean squared error over
%   its own frames, averaged over events, one square root on that mean.
%
%       e_i = (1 / L_i) * sum_t (s_sim - s_obs)^2        [m^2]
%       f   = sqrt( (1 / N) * sum_i e_i )                [m]
%
%   An event whose simulated spacing is not finite, or has reached zero, is scored 1e5 rather than
%   dropped, so the denominator stays the event count.

    if nargin < 3 || isempty(o), o = cfm.config(); end
    o = fill(o, D);
    theta = theta(:);

    if o.block >= D.N
        f = sqrt(mean(one_call(theta, D, o)));
        return
    end

    E = zeros(D.N, 1);
    for i1 = 1:o.block:D.N
        i2 = min(i1 + o.block - 1, D.N);
        E(i1:i2) = one_call(theta, slice(D, i1:i2), o);
    end
    f = sqrt(mean(E));
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
    e = residual(D, s_sim, v_sim, o);
end

function e = residual(D, s_sim, v_sim, o)
%RESIDUAL  Per-event mean squared spacing error, over the frames each event runs.
%
%   The same formula for both models, over every frame of the event: a frame the model was given rather
%   than simulated carries no error, so it neither helps nor hurts a ranking and the average stays
%   comparable between a model with a reaction time and one without.
    live  = (1:D.T) <= D.len(:);
    dead  = ~isfinite(s_sim) | (s_sim <= 0);     % the same test as is_dead() in the kernel
    d2    = (s_sim - D.space) .^ 2;
    e     = sum(d2 .* live, 2) ./ D.len(:);
    e(any(dead & live, 2)) = 1e5;
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
