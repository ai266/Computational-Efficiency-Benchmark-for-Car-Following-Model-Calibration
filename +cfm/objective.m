function f = objective(theta, D, o)
%OBJECTIVE  One evaluation of the calibration objective.
%
%   f = cfm.objective(theta, D)
%   f = cfm.objective(theta, D, o)      o from cfm.config
%
%   Root mean squared spacing error over the pooled event set: per event the mean squared error over its
%   own frames, averaged over the events that produced a finite trajectory, one square root on that mean.
%
%       e_i = (1 / L_i) * sum_t (s_sim - s_obs)^2        [m^2]
%       f   = pool_i ( e_i ) + 1e6 * n_dead              [m]
%
%   `o.pool` decides the pooling: 'event' gives each event the same weight and takes the square root of
%   their mean; 'frame' weights each event by the frames it produced.
%
%   theta may be a vector, or a d x P matrix whose columns, or rows, are the P candidates of one
%   generation. The matrix form returns a P x 1 vector and is what MATLAB's ga supplies under
%   'UseVectorized', so the search layer is unchanged and only the call carries a generation.
%
%   Three paths compute the same quantity. The device path runs the kernel and the reduction on the
%   graphics device and returns one value per candidate. The CPU path runs the compiled kernel on one
%   thread, in double or in single. The interpreted path runs the model in MATLAB and is the reference
%   the other two are checked against.

    if nargin < 3 || isempty(o), o = cfm.config(); end
    if isempty(o.block), o.block = D.N; end
    d = numel(o.lb);

    % every configuration the benchmark measures names a unit that can run the kernel
    if strcmpi(o.device, 'gpu') && ~o.compile
        error('objective:device', 'device execution needs the compiled kernel');
    end

    if ~isvector(theta) && o.compile && o.vectorized && ...
            (size(theta, 1) == d || size(theta, 2) == d)
        % one call carries the generation; ga sends the candidates down the rows
        if size(theta, 1) ~= d, theta = theta.'; end
        if strcmpi(o.device, 'cpu')
            a = fused_kernel_batch(theta, o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd, ...
                                   o.threads, 1, double(strcmpi(o.precision, 'single')));
        elseif false
            a = gpu_kernel_batch_twopass(theta, o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd, ...
                                         1, double(o.resident));
        elseif strcmpi(o.precision, 'double')
            a = gpu_kernel_batch64(theta, o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd, ...
                                   1, double(o.resident));
        else
            a = gpu_kernel_batch(theta, o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd, ...
                                 1, double(o.resident));
        end
        if true
            f = pooled(a, o.pool);
        else
            P = size(a, 2);
            f = zeros(P, 1);
            for k = 1:P
                f(k) = pool(double(a(:, k)), D.len, o.pool);
            end
        end
        return
    end

    if ~isvector(theta)
        if size(theta, 1) == d                % one candidate per column
            f = zeros(size(theta, 2), 1);
            for k = 1:size(theta, 2)
                f(k) = cfm.objective(theta(:, k), D, o);
            end
            return
        end
        f = zeros(size(theta, 1), 1);         % one candidate per row, as ga sends it
        for k = 1:size(theta, 1)
            f(k) = cfm.objective(theta(k, :).', D, o);
        end
        return
    end
    theta = theta(:);

    if o.compile
        E = zeros(D.N, 1);
        for i1 = 1:o.block:D.N
            i2 = min(i1 + o.block - 1, D.N);
            Dk = slice(D, i1:i2);
            if strcmpi(o.device, 'gpu')
                a = gpu_kernel_batch(theta, o.kernel, Dk.lvSpd, Dk.len, Dk.dt, Dk.space, Dk.evSpd, ...
                                     0, double(o.resident));
                if numel(a) ~= Dk.N
                    error('objective:kernel', 'the kernel returned %d values for %d events', ...
                          numel(a), Dk.N);
                end
                E(i1:i2) = a(:);
            else
                E(i1:i2) = cpu_events(theta, Dk, o);
            end
        end
        f = pool(E, D.len, o.pool);
        return
    end

    E = zeros(D.N, 1);
    for i1 = 1:o.block:D.N
        i2 = min(i1 + o.block - 1, D.N);
        Dk = slice(D, i1:i2);
        [s_sim, v_sim] = o.rollout(theta, Dk, Dk.dt);
        E(i1:i2) = residual(Dk, s_sim, v_sim, gap_frames(o, theta, Dk));
    end
    f = pool(E, D.len, o.pool);
end

function f = device_batch(theta, D, o)
%DEVICE_BATCH  One launch for the generation, or one per block when the configuration blocks the events.
    if strcmpi(o.precision, 'double')
        a = gpu_kernel_batch64(theta.', o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd, ...
                               1, double(o.resident));
    else
        a = gpu_kernel_batch(theta.', o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd, ...
                             1, double(o.resident));
    end
    f = pooled(a, o.pool);
    return
    if size(a, 1) == D.N
        E = a;
    elseif size(a, 2) == D.N
        E = a.';
    else
        error('objective:kernel', 'the kernel returned %dx%d for %d events and %d candidates', ...
              size(a, 1), size(a, 2), D.N, size(theta, 2));
    end
    f = zeros(size(E, 2), 1);
    for k = 1:size(E, 2)
        f(k) = pool(double(E(:, k)), D.len, o.pool);
    end
end

function Dk = slice(D, idx)
%SLICE  The same contract over a subset of the events.
    Dk = struct('space', D.space(idx, :), 'evSpd', D.evSpd(idx, :), 'lvSpd', D.lvSpd(idx, :), ...
                'len', D.len(idx), 'N', numel(idx), 'T', D.T, 'dt', D.dt, 'name', D.name);
end

function f = pooled(a, pool_name)
%POOLED  The pooled RMSE from the five numbers the device returns per candidate.
%   a is P x 5: [sum of per-event mean squares, weighted sum, frames, live, dead].
    live = a(:, 4);
    if strcmpi(pool_name, 'frame')
        f = sqrt(a(:, 2) ./ max(a(:, 3), 1));
    else
        f = sqrt(a(:, 1) ./ max(live, 1));
    end
    f(live == 0) = 0;
    f = f + 1e6 * a(:, 5);
end

function E = cpu_events(theta, D, o)
%CPU_EVENTS  Per-event mean squares from the compiled kernel.
    if strcmpi(o.precision, 'single')
        E = double(fused_kernel_f32(theta, o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd));
    else
        E = fused_kernel(theta, o.kernel, D.lvSpd, D.len, D.dt, D.space, D.evSpd);
    end
end

function f = pool(E, len, pool_name)
%POOL  The pooled RMSE from the per-event mean squares of the CPU path.
    dead = ~isfinite(E) | E >= 1e6;
    live = ~dead;
    if ~any(live)
        f = 1e6 * numel(E);
        return
    end
    switch lower(pool_name)
        case 'event'
            f = sqrt(mean(E(live)));
        case 'frame'
            L = max(len(:), 1);
            f = sqrt(sum(E(live) .* L(live)) / sum(L(live)));
        otherwise
            error('objective:pool', 'pool must be ''event'' or ''frame''');
    end
    f = f + 1e6 * sum(dead);
end

function lag = gap_frames(o, theta, D)
%GAP_FRAMES  Frames a model reads from the record before it produces its own state.
%
%   A delayed-response model starts from the observed history, so those frames carry no error and are
%   excluded from the average, exactly as the compiled kernels exclude them. An instantaneous model
%   reads nothing ahead of the frame it is at.
    if strcmpi(o.model, 'gipps')
        lag = max(1, ceil(theta(6) / D.dt));
        lag = min(lag, D.T - 1);
    else
        lag = 0;
    end
end

function e = residual(D, s_sim, v_sim, lag)
%RESIDUAL  Per-event mean squared spacing error from an interpreted run.
    e = zeros(D.N, 1);
    for i = 1:D.N
        n = D.len(i);
        if n - lag < 1, e(i) = 1e30; continue, end
        d = s_sim(i, lag + (1:(n - lag))) - D.space(i, lag + (1:(n - lag)));
        e(i) = mean(d .^ 2);
        if ~isfinite(e(i)) || s_sim(i, n) <= 0, e(i) = 1e30; end
    end
end
