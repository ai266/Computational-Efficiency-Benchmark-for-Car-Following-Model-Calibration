function build_kernel(action)
%BUILD_KERNEL  Build the compiled kernel.
%
%   build_kernel            build, then compare the objective with the interpreted path
%   build_kernel('quiet')   build without the comparison
%   build_kernel('ensure')  build only if the binary is missing or older than its source
%
%   cfm.objective calls the 'ensure' form once per session.
%
%   OpenMP is enabled through COMPFLAGS because mex does not expose it. MSVC reads /openmp and
%   ignores -fopenmp without a word; without the flag every #pragma omp is skipped and the thread
%   count has no effect.

    if nargin < 1 || isempty(action), action = 'check'; end
    action = validatestring(action, {'check', 'quiet', 'ensure'});

    here = fileparts(mfilename('fullpath'));
    kdir = fullfile(here, 'kernels');
    src  = fullfile(kdir, 'fused_kernel.cpp');
    if ~isfile(src)
        error('build_kernel:missing', 'no %s', src);
    end
    bin = fullfile(kdir, ['fused_kernel.' mexext]);

    if strcmp(action, 'ensure') && isfile(bin)
        d = dir(bin); s = dir(src);
        if d.datenum >= s.datenum, return, end
    end

    old = pwd;
    cleaner = onCleanup(@() cd(old));
    cd(kdir);

    cc = mex.getCompilerConfigurations('C++', 'Selected');
    if isempty(cc)
        error('build_kernel:nocompiler', 'no C++ compiler selected; run mex -setup C++');
    end
    fprintf('compiler %s %s\n', cc.Name, cc.Version);
    mex('COMPFLAGS=$COMPFLAGS /openmp -O2', 'fused_kernel.cpp', '-output', 'fused_kernel');

    d = dir(['fused_kernel.' mexext]);
    fprintf('built    fused_kernel.%s  %.1f kB  %s\n', mexext, d(1).bytes / 1e3, d(1).date);

    if ~strcmp(action, 'quiet')
        % The two paths are algebraically identical and agree bit for bit where the rollout is
        % smooth; at an exponent near the top of the box the trajectory is chaotic and the last bits
        % of the power can grow to about 1e-5 relative.
        addpath(here, kdir);
        D  = cfm.load_trajectories('ngsim_test');
        th = [33; 1.2; 1.5; 1.0; 3; 2];
        base = {'block', [], 'threads', 1, 'model', 'idm'};
        vc = cfm.objective(th, D, cfm.config('compile', true,  base{:}));
        vi = cfm.objective(th, D, cfm.config('compile', false, base{:}));
        rel = abs(vc - vi) / max(abs(vi), eps);
        fprintf('check    compiled %.10g  interpreted %.10g  relative %.2e\n', vc, vi, rel);
        if ~isfinite(vc) || rel > 1e-6
            error('build_kernel:gate', 'the built kernel disagrees with the model: relative %g', rel);
        end
    end
end
