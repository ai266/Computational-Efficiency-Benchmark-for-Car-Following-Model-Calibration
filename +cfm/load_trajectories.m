function D = load_trajectories(name)
%LOAD_TRAJECTORIES  Read one converted split into the data contract.
%
%   D = cfm.load_trajectories('ngsim_train')
%
%   Fields
%     space   N x T  observed spacing, m
%     evSpd   N x T  observed follower speed, m/s
%     lvSpd   N x T  leader speed, m/s
%     len     N x 1  frames each event runs
%     N, T, dt, name
%
%   Every event is cut to its first `window` frames at `frameStep` seconds each, both from
%   cfm.config. A shorter event keeps its own length. The loader does not resample.

    if nargin < 1 || isempty(name), name = 'ngsim_train'; end

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    file = fullfile(root, 'data', [lower(name) '.mat']);
    if ~isfile(file)
        error('dataset:missing', 'no such data set: %s', file);
    end

    % One field is read; any other field in the file is ignored.
    S = load(file, 'events');
    if ~isfield(S, 'events')
        error('dataset:field', '%s has no events array', file);
    end
    raw = S.events;

    o = cfm.config();
    nWindow = o.window;
    if size(raw, 2) > nWindow
        raw = raw(:, 1:nWindow, :);
    end

    % Channels: 1 spacing, 2 follower speed, 3 relative speed, 4 leader speed.
    D.name  = lower(name);
    D.dt    = o.frameStep;
    D.space = double(raw(:, :, 1));
    D.evSpd = double(raw(:, :, 2));
    D.lvSpd = double(raw(:, :, 4));
    D.N     = size(D.space, 1);
    D.T     = size(D.space, 2);

    % Zero spacing is padding, so an event ends at its last positive frame, and at least two frames
    % are needed for a step.
    pos = D.space > 0;
    [~, fromEnd] = max(flip(pos, 2), [], 2);
    D.len = D.T + 1 - fromEnd;
    D.len(~any(pos, 2)) = 2;
end
