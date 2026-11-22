function D = dataset(cell_name, frac, repeat)
%DATASET  One cell as an event set, or a prefix of it for the workload sensitivity.
%
%   D = core.dataset('ngsim_train')        the whole set
%   D = core.dataset('ngsim_train', 0.5)   the first half of the events
%
%   The fraction is taken from the front of the set rather than by sampling, because the events of the
%   two corpora are extracted in recording order and a prefix keeps the order a reader of the data set
%   would expect. The subset keeps the contract of the full set, so every kernel accepts it unchanged.

    if nargin < 2, frac = 1; end
    if nargin < 3, repeat = 1; end
    D = cfm.load_trajectories(cell_name);
    if frac < 1
    n = max(2, round(frac * D.N));
    D.space = D.space(1:n, :);
    D.evSpd = D.evSpd(1:n, :);
    D.lvSpd = D.lvSpd(1:n, :);
    D.len   = D.len(1:n);
    D.N     = n;
    D.name  = sprintf('%s-%d', D.name, round(frac * 100));
    end
    if repeat > 1
        % the number of events is the number of trajectories, so the workload can be widened past the
        % corpus by stacking copies of the same set, which keeps the arithmetic identical per event
        D.space = repmat(D.space, repeat, 1);
        D.evSpd = repmat(D.evSpd, repeat, 1);
        D.lvSpd = repmat(D.lvSpd, repeat, 1);
        D.len   = repmat(D.len, repeat, 1);
        D.N     = D.N * repeat;
        D.name  = sprintf('%s-x%d', D.name, repeat);
    end
end
