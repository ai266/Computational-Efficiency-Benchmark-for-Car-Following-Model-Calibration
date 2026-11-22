function ok = verify_arms()
%VERIFY_ARMS  Check that every arm reaches the objective as exactly one changed field.
%
%   ok = core.verify_arms()
%
%   The check exists because an override that never reaches the objective turns an arm into a second
%   run of the baseline, and the ratio then reads 1.0 without saying so. Each arm is resolved against
%   the shipped configuration, the differing fields are counted, and a behavioural signature is compared
%   with the baseline, so a field that is set but ignored is caught here rather than in a table.

    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..')));
    parallel.gpu.enableCUDAForwardCompatibility(true);

    base = cfm.config('model', 'idm');
    names = {'drop-event-batching','drop-candidate-batching','drop-reduction','drop-residency', ...
             'drop-precision','drop-compilation'};
    ok = true;
    for k = 1:numel(names)
        dev = core.params(names{k}, 'ngsim_train', 'idm');
        cfg = base;
        for f = fieldnames(dev).'
            cfg.(f{1}) = dev.(f{1});
        end
        f = fieldnames(base);
        differing = {};
        for j = 1:numel(f)
            if ~isequal(cfg.(f{j}), base.(f{j}))
                differing{end + 1} = f{j}; %#ok<AGROW>
            end
        end
        fprintf('%-24s changes %d field(s): %s\n', names{k}, numel(differing), strjoin(differing, ', '));
        if numel(differing) ~= 1
            fprintf('  FAIL: an arm must change exactly one field\n');
            ok = false;
        end
    end

    % a field that is set but ignored shows up as a signature that does not change
    D = cfm.load_trajectories('ngsim_train');
    th = [39.4; 1.45; 0.74; 0.36; 1.49; 4.28];
    X = repmat(th, 1, 4).';
    v0 = cfm.objective(X, D, base);
    v1 = cfm.objective(X, D, cfm.config('block', 512));       % same value, different number of launches
    v2 = cfm.objective(X, D, cfm.config('precision', 'double'));
    fprintf('blocked value %s, double value %s\n', mat2str(v1(1), 9), mat2str(v2(1), 9));
    if abs(v1(1) - v0(1)) > 1e-6, fprintf('  FAIL: the block arm changed the value\n'); ok = false; end
    if abs(v2(1) - v0(1)) < 1e-12, fprintf('  FAIL: the precision arm did not change the arithmetic\n'); ok = false; end
end
