function out = run_calibration(train_set, test_set, model, cfg)
%RUN_CALIBRATION  Calibrate one model on one split and report the fit and the timings.
%
%   out = run_calibration('ngsim_train', 'ngsim_test', 'idm', struct(...))
%
%   cfg fields, all optional, defaulting to cfm.config: pop, gens, threads, workers, seed,
%   population (d x pop starting generation), block, D (an event set already loaded), override (a
%   whole settings struct), objective (a handle to replace cfm.objective), keep_pool.
%
%   out fields: model, train, test, theta, train_rmse, train_mse, test_rmse, test_mse, t_eval, t_gen,
%   t_calib, n_eval, seed, check.
%
%   The timings cover the search only; the pool is started before the clock.

    if nargin < 4, cfg = struct(); end

    here = fileparts(mfilename('fullpath'));
    addpath(here); addpath(fullfile(here, 'kernels'));
    % One BLAS thread in the client; the kernel's own threads are set separately. Restored on return.
    nThreadsBefore = maxNumCompThreads(1);
    restoreThreads = onCleanup(@() maxNumCompThreads(nThreadsBefore));

    % A field the caller names beats the override struct, which beats the default; the five fields the
    % search reads are copied back into cfg, so the pool and the clock come from the same settings.
    named = struct();
    for f = {'pop', 'gens', 'seed', 'threads', 'workers'}
        if isfield(cfg, f{1}) && ~isempty(cfg.(f{1}))
            named.(f{1}) = cfg.(f{1});
        end
    end

    o = cfm.config('model', model);
    if isfield(cfg,'override') && ~isempty(cfg.override)
        o = cfg.override;
        o.model = model;
    end
    if isfield(cfg,'block'), o.block = cfg.block; end
    for f = fieldnames(named).'
        o.(f{1}) = named.(f{1});
    end
    cfg.pop = o.pop;  cfg.gens = o.gens;  cfg.seed = o.seed;
    cfg.threads = o.threads;  cfg.workers = o.workers;  cfg.block = o.block;

    if isfield(cfg,'D') && ~isempty(cfg.D)
        Dtr = cfg.D;
    else
        Dtr = cfm.load_trajectories(train_set);
    end

    if nargin >= 2 && ~isempty(test_set)
        Dte = cfm.load_trajectories(test_set);
    else
        Dte = [];
    end

    lb = o.lb; ub = o.ub;
    d  = numel(lb);

    f = @(x) cfm.objective(x, Dtr, o);
    custom = isfield(cfg,'objective') && ~isempty(cfg.objective);
    if custom, f = cfg.objective; end

    use_pool = cfg.workers > 0 && ~custom;
    if ~isfield(cfg,'keep_pool'), cfg.keep_pool = false; end
    if use_pool
        cur = gcp('nocreate');
        if isempty(cur) || cur.NumWorkers ~= cfg.workers
            if ~isempty(cur), delete(cur); end
            parpool('Processes', cfg.workers);
        end
        cD = parallel.pool.Constant(Dtr);
        f = @(x) cfm.objective(x, cD.Value, o);
        parfor i = 1:cfg.workers                       % every worker evaluates once, untimed
            maxNumCompThreads(1);
            cfm.objective(0.5*ones(d,1), cD.Value, o);
        end
    else
        cfm.objective(0.5*ones(d,1), Dtr, o);          % warm-up, untimed
    end

    t1 = inf;
    for r = 1:5
        x0 = lb + rand(d,1).*(ub-lb);
        t0 = tic; f(x0); t1 = min(t1, toc(t0));
    end

    % MATLAB's ga, every operator at its default. The two stopping rules are disabled so that a run
    % is exactly pop x gens evaluations rather than ending on a stall or a tolerance.
    rng(cfg.seed, 'twister');
    np = cfg.pop;
    opts = optimoptions('ga', ...
        'PopulationSize', np, ...
        'MaxGenerations', cfg.gens, ...
        'MaxStallGenerations', Inf, ...
        'FunctionTolerance', 0, ...
        'UseParallel', use_pool, ...
        'Display', 'off');
    if isfield(cfg,'population') && ~isempty(cfg.population)
        P = cfg.population;
        if ~isequal(size(P), [d np])
            error('run_calibration:population', 'population must be %d x %d, got %dx%d', ...
                  d, np, size(P,1), size(P,2));
        end
        opts = optimoptions(opts, 'InitialPopulationMatrix', P.');
    end

    t0 = tic;
    [xbest, fbest] = ga(f, d, [], [], [], [], lb, ub, [], opts);
    t_calib = toc(t0);

    if use_pool && ~cfg.keep_pool
        delete(gcp('nocreate'));
    end
    if o.compile
        fused_kernel('release');    % the buffers of this data set are not needed once it is scored
    end

    % The reported RMSE must be reproducible at the reported parameters.
    recheck = cfm.objective(xbest, Dtr, o);
    rel = abs(recheck - fbest) / max(abs(fbest), eps);

    if isempty(Dte)
        test_mse = NaN;
    else
        test_mse = cfm.objective(xbest, Dte, o);
    end

    out = struct('model', model, 'train', train_set, 'test', test_set, ...
                 'theta', xbest, 'train_rmse', fbest, 'train_mse', fbest^2, ...
                 'test_rmse', test_mse, 'test_mse', test_mse^2, ...
                 'check', rel, 't_eval', t1, 't_gen', t1*cfg.pop, 't_calib', t_calib, ...
                 'n_eval', np*cfg.gens, 'seed', cfg.seed);

    fprintf('\n%s / %s\n', train_set, model);
    fprintf('  calib  RMSE %8.4f m   MSE %9.4f m^2\n', fbest, fbest^2);
    fprintf('  test   RMSE %8.4f m   MSE %9.4f m^2\n', test_mse, test_mse^2);
    fprintf('  theta  %s\n', mat2str(xbest.', 6));
    fprintf('  timing one evaluation %.3f ms | one generation %.3f s | calibration %.1f s\n', ...
            t1*1e3, t1*cfg.pop, t_calib);
    fprintf('  budget %d evaluations (%d x %d)\n', np*cfg.gens, np, cfg.gens);
end
