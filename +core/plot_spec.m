function s = plot_spec(what)
%PLOT_SPEC  The one place where the figures of the paper take their typography and colour.
%
%   s = core.plot_spec('line')      a curve
%   s = core.plot_spec('bar')       a bar chart
%   s = core.plot_spec('map')       a matrix of ratios
%
%   Every figure is drawn from a data file in results/ with the settings returned here, so the figures
%   of the paper share a type size, a line weight and a palette without each script repeating them.

    s.font = 'Times New Roman';
    s.size = 8;
    s.line = 1.0;
    s.axes = 0.5;
    s.palette = [0.00 0.00 0.00; 0.35 0.35 0.35; 0.62 0.62 0.62; 0.85 0.85 0.85];
    switch what
        case 'line'
            s.marker = {'o','s','^','d'};
            s.colour = [0 0 0; 0.45 0.45 0.45; 0.20 0.20 0.20; 0.65 0.65 0.65];
        case 'bar'
            s.width = 0.62;
            s.edge = 0.4;
        case 'map'
            s.ramp = gray(64);
            s.text = 7;
        otherwise
            error('plot_spec:unknown', 'unknown figure class %s', what);
    end
end
