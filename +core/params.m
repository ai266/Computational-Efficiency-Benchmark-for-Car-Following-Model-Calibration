function cfg = params(name, cell, model)
%PARAMS  One row of the experimental design, as a configuration for run_calibration.
%
%   cfg = core.params('opt', 'ngsim_train', 'idm')
%
%   Every experiment in the paper is a name in this table, so a name identifies one configuration and
%   one row of results/campaign.csv. A run that is not in the table is an error rather than an
%   improvisation, which is what keeps two experiments from measuring the same configuration twice.
%
%   Names are grouped by the table they feed.
%
%     baseline   the operating point, also the baseline of the reverse drop-one
%     drop-*     one technique removed from the operating point
%     events-*   fraction of the event set, workload sensitivity
%     pop-*      population at a fixed budget, budget sensitivity
%     pool-*     process pool instead of the device, configuration sensitivity
%
%   The base configuration is the shipped default, so any field a name does not change stays at the
%   value the package documents.

    if nargin < 2, cell = 'ngsim_train'; end
    if nargin < 3, model = 'idm'; end

    cfg = struct('model', model);

    switch name
        case 'baseline'
            % nothing
        case 'drop-device'
            cfg.device = 'cpu';
        case 'drop-event-batching'
            cfg.block = 1;            % one call per event: batching off
        case 'drop-candidate-batching'
            cfg.vectorized = false;

        case 'drop-fusion'
            cfg.fusion = false;
        case 'drop-reduction'
            cfg.reduce = false;
        case 'drop-residency'
            cfg.resident = false;
        case 'drop-precision'
            cfg.precision = 'double';
        case 'drop-compilation'
            cfg.compile = false;      % the same budget, at the cost of the interpreted path
            cfg.device  = 'cpu';      % the interpreter runs where the model runs

        case 'events-75'
            cfg.D = core.dataset(cell, 0.75);
        case 'events-50'
            cfg.D = core.dataset(cell, 0.50);
        case 'events-25'
            cfg.D = core.dataset(cell, 0.25);
        case 'pop-50'
            cfg.pop = 50; cfg.gens = 200;
        case 'pop-200'
            cfg.pop = 200; cfg.gens = 50;
        case 'pool-4'
            cfg.device = 'cpu'; cfg.workers = 4;
        case 'pool-8'
            cfg.device = 'cpu'; cfg.workers = 8;
        otherwise
            error('params:unknown', 'unknown configuration %s', name);
    end
end
