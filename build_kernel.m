function build_kernel(action)
%BUILD_KERNEL  Build every compiled kernel from the shared model headers.
%
%   build_kernel            build, then check that the paths agree
%   build_kernel('quiet')   build without the check
%   build_kernel('ensure')  build only what is missing or older than a source
%
%   Four binaries are produced from the same model sources in kernels/models.
%
%     fused_kernel        the CPU kernel in double precision
%     fused_kernel_f32    the CPU kernel in single precision
%     gpu_kernel          the device kernel, one candidate per call
%     gpu_kernel_batch    the device kernel, one generation per call, single precision
%     gpu_kernel_batch64  the same in double precision
%
%   A model is one header in kernels/models, reached by every kernel through the same two functions, so a
%   change to the physics is a change in one place. The CPU kernels are built with mex and the device
%   kernels with nvcc; a machine without CUDA keeps the CPU kernels and says so.
%
%   The check at the end scores a collision-free parameter vector on every available path. A difference
%   beyond the rounding of single precision means a kernel has drifted from its model.

    if nargin < 1 || isempty(action), action = 'check'; end
    action = validatestring(action, {'check', 'quiet', 'ensure'});

    here = fileparts(mfilename('fullpath'));
    kdir = fullfile(here, 'kernels');
    cpu = { 'fused_kernel.cpp',     'fused_kernel'; ...
            'fused_kernel_f32.cpp', 'fused_kernel_f32' };
    gpu = { 'gpu_kernel.cu',        'gpu_kernel'; ...
            'gpu_kernel_batch.cu',  'gpu_kernel_batch'; ...
            'gpu_kernel_batch64.cu','gpu_kernel_batch64' };
    srcs = [cpu(:, 1); gpu(:, 1); {'models/idm_step.h'; 'models/gipps_step.h'}];
    for k = 1:numel(srcs)
        if ~isfile(fullfile(kdir, srcs{k}))
            error('build_kernel:missing', 'no %s', fullfile(kdir, srcs{k}));
        end
    end

    bins = [cpu(:, 2); gpu(:, 2)];
    if strcmp(action, 'ensure')
        newest = 0;
        for k = 1:numel(srcs)
            newest = max(newest, dir(fullfile(kdir, srcs{k})).datenum);
        end
        stale = false;
        for k = 1:numel(bins)
            bin = fullfile(kdir, [bins{k} '.' mexext]);
            if ~isfile(bin) || dir(bin).datenum < newest, stale = true; end
        end
        if ~stale, return, end
    end

    old = pwd;
    cleaner = onCleanup(@() cd(old));
    cd(kdir);

    cc = mex.getCompilerConfigurations('C++', 'Selected');
    if isempty(cc)
        error('build_kernel:nocompiler', 'no C++ compiler selected; run mex -setup C++');
    end
    fprintf('compiler %s %s\n', cc.Name, cc.Version);

    for k = 1:size(cpu, 1)
        mex('COMPFLAGS=$COMPFLAGS -O2', '-I.', cpu{k, 1}, '-output', cpu{k, 2});
        d = dir([cpu{k, 2} '.' mexext]);
        fprintf('built    %-24s %7.1f kB  %s\n', [cpu{k, 2} '.' mexext], d(1).bytes / 1e3, d(1).date);
    end

    nvcc = find_nvcc();
    ccbin = find_cl();
    if isempty(nvcc)
        fprintf('skipped  the device kernels: nvcc not found; the CPU kernels are in use\n');
    elseif isempty(ccbin)
        fprintf('skipped  the device kernels: no cl.exe found for nvcc\n');
    else
        mexroot = fullfile(matlabroot, 'extern');
        for k = 1:size(gpu, 1)
            cmd = sprintf(['"%s" -ccbin "%s" -shared -allow-unsupported-compiler -O3 ' ...
                           '-cudart static -Xcompiler /MD -o "%s" "%s" -I"%s" -I. ' ...
                           '-L"%s" -llibmex -llibmx -Xlinker /EXPORT:mexFunction'], ...
                          nvcc, ccbin, [gpu{k, 2} '.' mexext], gpu{k, 1}, ...
                          fullfile(mexroot, 'include'), fullfile(mexroot, 'lib', 'win64', 'microsoft'));
            [st, out] = system(cmd);
            if st ~= 0
                fprintf('failed   %s\n%s\n', gpu{k, 2}, out);
                error('build_kernel:cuda', 'nvcc failed on %s', gpu{k, 1});
            end
            d = dir([gpu{k, 2} '.' mexext]);
            fprintf('built    %-24s %7.1f kB  %s\n', [gpu{k, 2} '.' mexext], d(1).bytes / 1e3, d(1).date);
        end
    end

    if ~strcmp(action, 'quiet'), check_paths(here, kdir); end
end

function nvcc = find_nvcc()
%FIND_NVCC  The compiler that built the device kernels, if this machine has CUDA.
    nvcc = '';
    roots = { getenv('CUDA_PATH'), 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' };
    for k = 1:numel(roots)
        if isempty(roots{k}), continue, end
        d = dir(fullfile(roots{k}, 'v*'));
        for j = 1:numel(d)
            exe = fullfile(roots{k}, d(j).name, 'bin', 'nvcc.exe');
            if isfile(exe), nvcc = exe; return, end
        end
        exe = fullfile(roots{k}, 'bin', 'nvcc.exe');
        if isfile(exe), nvcc = exe; return, end
    end
    [st, out] = system('where nvcc');
    if st == 0
        lines = strsplit(strtrim(out), char(10));
        if ~isempty(lines) && isfile(lines{1}), nvcc = lines{1}; end
    end
end

function cl = find_cl()
%FIND_CL  The directory holding cl.exe, which nvcc takes as its host compiler.
    cl = '';
    roots = { 'C:\Program Files\Microsoft Visual Studio', ...
              'C:\Program Files (x86)\Microsoft Visual Studio' };
    for k = 1:numel(roots)
        d = dir(fullfile(roots{k}, '*', '*', 'VC', 'Tools', 'MSVC', '*', 'bin', 'Hostx64', 'x64', 'cl.exe'));
        if ~isempty(d)
            cl = d(1).folder;
            return
        end
    end
    [st, out] = system('where cl');
    if st == 0
        lines = strsplit(strtrim(out), char(10));
        if ~isempty(lines), cl = fileparts(lines{1}); end
    end
end

function check_paths(here, kdir)
%CHECK_PATHS  Score one collision-free vector on every path that was built.
    addpath(here, kdir);
    models = {'idm', 'gipps'};
    for c = 1:numel(models)
        D = cfm.load_trajectories('ngsim_test');
        base = {'block', [], 'model', models{c}};
        o = cfm.config('compile', false, 'device', 'cpu', base{:});
        rng(11);
        th = []; ref = [];
        for k = 1:400
            x = o.lb + rand(numel(o.lb), 1) .* (o.ub - o.lb);
            v = cfm.objective(x, D, o);
            if isfinite(v) && v < 100, th = x; ref = v; break, end
        end
        if isempty(th)
            fprintf('check    %-5s skipped, no collision-free vector in the box\n', models{c});
            continue
        end
        fprintf('check    %-5s interpreted %12.6f\n', models{c}, ref);
        list = { 'cpu', 'double'; 'cpu', 'single'; 'gpu', 'single' };
        for k = 1:size(list, 1)
            try
                v = cfm.objective(th, D, cfm.config('compile', true, 'device', list{k, 1}, ...
                                                   'precision', list{k, 2}, base{:}));
                rel = abs(v - ref) / max(abs(ref), eps);
                fprintf('         %-4s %-7s %12.6f  relative %8.1e\n', list{k, 1}, list{k, 2}, v, rel);
                if rel > 1e-4
                    error('build_kernel:gate', '%s %s disagrees with the model: relative %g', ...
                          list{k, 1}, list{k, 2}, rel);
                end
            catch err
                if strcmp(err.identifier, 'build_kernel:gate'), rethrow(err); end
                fprintf('         %-4s %-7s skipped, %s\n', list{k, 1}, list{k, 2}, err.message);
            end
        end
    end
end
