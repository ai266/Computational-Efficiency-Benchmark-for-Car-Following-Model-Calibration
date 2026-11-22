function record(experiment, cell_name, model, name, t, t_calib, rmse, n_eval, note)
%RECORD  Append one measurement, with its repetitions, to results/campaign.csv.
%
%   core.record('ablation', 'ngsim_train', 'idm', 'drop-residency', t, t_calib, rmse, 10000, '')
%
%   t is the vector of repetitions. The row keeps the median, the quartiles and a bootstrap confidence
%   interval on the median, so a ratio in the paper can be read against the spread that produced it
%   rather than against a single number. The raw repetitions go to results/raw/ under the same key, so a
%   reader can recompute the interval.
%
%   A key that already exists is replaced rather than duplicated, which makes a rerun safe.

    if nargin < 9, note = ''; end
    if isscalar(t), t = t; end
    reps = numel(t);
    med = median(t);
    lo  = prctile(t, 25);
    hi  = prctile(t, 75);
    [ci_lo, ci_hi] = boot_median_ci(t);

    here = fileparts(mfilename('fullpath'));
    res  = fullfile(here, '..', 'results');
    if ~exist(fullfile(res, 'raw'), 'dir'), mkdir(fullfile(res, 'raw')); end

    key = sprintf('%s|%s|%s', cell_name, model, name);
    row = table(string(key), string(experiment), string(cell_name), string(model), string(name), ...
                reps, med, lo, hi, ci_lo, ci_hi, min(t), max(t), t_calib, rmse, n_eval, string(note), ...
                'VariableNames', {'key','experiment','cell','model','name','reps','t_median', ...
                                  'q25','q75','ci_lo','ci_hi','t_min','t_max','t_calib','rmse', ...
                                  'n_eval','note'});

    csv = fullfile(res, 'campaign.csv');
    if isfile(csv)
        T = readtable(csv, 'TextType', 'string');
        T = T(T.key ~= key, :);
        T = [T; row];
    else
        T = row;
    end
    writetable(T, csv);
    writematrix(t(:), fullfile(res, 'raw', [strrep(key, '|', '_') '.csv']));
end

function [lo, hi] = boot_median_ci(t)
%BOOT_MEDIAN_CI  Percentile bootstrap interval on the median, ten per cent tails.
    t = t(:);
    n = numel(t);
    if n < 4, lo = min(t); hi = max(t); return, end
    rng(7, 'twister');
    B = 1000;
    meds = zeros(B, 1);
    for b = 1:B
        meds(b) = median(t(randi(n, n, 1)));
    end
    lo = prctile(meds, 5);
    hi = prctile(meds, 95);
end
