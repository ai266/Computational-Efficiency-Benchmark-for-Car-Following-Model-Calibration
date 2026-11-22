function o = config(varargin)
%CONFIG  Every number a run takes, in one place, overridable by name.
%
%   o = cfm.config()                 the shipped configuration
%   o = cfm.config('block', 100)     one field overridden
%
%   Fields
%     compile    true   the compiled kernel
%     block      []     events per call; empty means the whole set
%     workers    0      processes over candidates; 0 runs in this process
%     threads    8      OpenMP threads of the processor kernel
%     vectorized true   one call carries the generation (ga 'UseVectorized')
%     resident   true   the event set stays on the device between calls
%     device     'gpu'  'cpu' evaluates in this process, 'gpu' on the graphics device
%     precision  'single' 'single', or 'double' on the processor path
%     pool       'event' how the per-event errors are pooled: 'event' equal weight, 'frame' by frames
%     window     150    frames per event
%     frameStep  0.1    seconds per frame
%     pop        100    candidates per generation
%     gens       100    generations
%     seed       7      seed of the draw
%     model      'idm'  'idm' or 'gipps'
%
%   `lb`, `ub`, `kernel` and `rollout` are filled from model_table below and cannot be overridden.
%   An unknown or protected name is an error.

    o = struct('compile', true, 'block', [], ...
               'workers', 0, 'threads', 8, 'vectorized', true, 'resident', true, ...
               'device', 'gpu', 'precision', 'single', ...
               'pool', 'event', ...
               'window', 150, 'frameStep', 0.1, ...
               'pop', 100, 'gens', 100, 'seed', 7, ...
               'model', 'idm');

    protected = {'lb', 'ub', 'kernel', 'rollout'};
    for k = 1:2:numel(varargin)
        name = varargin{k};
        if ~ischar(name) || ~isfield(o, name) || any(strcmp(name, protected))
            error('opts:unknown', 'unknown or protected option %s', num2str(name));
        end
        o.(name) = varargin{k + 1};
    end

    l = model_table();
    k = find(strcmpi({l.name}, o.model), 1);
    if isempty(k)
        error('opts:model', 'unknown model %s', o.model);
    end
    o.model   = l(k).name;
    o.lb      = l(k).lb;
    o.ub      = l(k).ub;
    o.kernel  = l(k).id;
    o.rollout = l(k).rollout;

    if ~isempty(o.block) && o.block < 1
        error('opts:block', 'block must be at least 1');
    end
    if ~any(strcmpi(o.device, {'cpu', 'gpu'}))
        error('opts:device', 'device must be ''cpu'' or ''gpu''');
    end
    if ~any(strcmpi(o.precision, {'double', 'single'}))
        error('opts:precision', 'precision must be ''double'' or ''single''');
    end
end

function L = model_table()
%MODEL_TABLE  One row per model: name, compiled-kernel branch, sampling box, rollout handle.
    L = struct('name', {}, 'id', {}, 'lb', {}, 'ub', {}, 'rollout', {});
    L(end + 1) = struct('name', 'idm', 'id', 0, ...
        'lb', [ 0.3; 0.1; 0.1; 0.1; 1.0;  0.1], ...   % v0, T, a, b, delta, s0
        'ub', [41.7; 5.0; 5.0; 5.0; 40.0; 10.0], ...
        'rollout', @cfm.idm);
    L(end + 1) = struct('name', 'gipps', 'id', 1, ...
        'lb', [0.1; 0.1; 0.5; 0.1;  0.3; 0.3], ...     % a_n, b_n, S_n1, b, V_0, tau
        'ub', [5.0; 5.0; 3.0; 5.0; 41.7; 3.0], ...
        'rollout', @cfm.gipps);
end
