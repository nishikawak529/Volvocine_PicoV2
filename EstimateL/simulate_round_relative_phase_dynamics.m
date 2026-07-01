function out = simulate_round_relative_phase_dynamics(round_dir, M, N, varargin)
% Simulate a round oscillator network using pairwise couplings estimated from CSV data.
%
% For each pair folder such as "7-8", this function calls
% plot_phase_dynamics_from_csv() to estimate the pairwise coupling functions,
% fits the resulting Gamma curves with a first harmonic, and then simulates
% the phase network
%   dphi_j/dt = omega_j + sigma * sum_k Gamma_kj(phi_j - phi_k)
% with omega_j fixed to 2.5*pi by default.
%
% The simulation is returned both as absolute phases and as relative phases
% with respect to a chosen reference oscillator.
%
% Examples:
%   out = simulate_round_relative_phase_dynamics();
%   out = simulate_round_relative_phase_dynamics(fullfile('EstimateL','Round'));
%   out = simulate_round_relative_phase_dynamics(fullfile('EstimateL','Round'), 10, 10, ...
%       'simulation_duration_sec', 120, 'simulation_dt', 0.01);

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'Round');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end
    if nargin < 3 || isempty(N)
        N = M;
    end

    default_sigma = 7;

    opts = parse_options(default_sigma, varargin{:});
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');
    validateattributes(N, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'N');

    pair_infos = list_pair_folders(round_dir);
    if isempty(pair_infos)
        error('No pair folders like "7-8" were found under %s.', round_dir);
    end

    pair_results = [];
    for k = 1:numel(pair_infos)
        info = pair_infos(k);
        csv_pattern = fullfile(info.folder, '*.csv');
        fprintf('[INFO] Pair %d/%d: %d-%d\n', k, numel(pair_infos), info.agent_ids(1), info.agent_ids(2));

        try
            pair_out = plot_phase_dynamics_from_csv(csv_pattern, info.agent_ids, M, N, ...
                'analysis_start_sec', opts.analysis_start_sec, ...
                'analysis_duration_sec', opts.analysis_duration_sec, ...
                'sample_dt', opts.sample_dt, ...
                'sigma', opts.sigma, ...
                'n_psi', opts.n_psi, ...
                'n_theta', opts.n_theta, ...
                'signal_column', opts.signal_column, ...
                'normalize_signal', opts.normalize_signal, ...
                'tail_percent', opts.tail_percent, ...
                'clip_normalized_signal', opts.clip_normalized_signal, ...
                'clip_limit', opts.clip_limit, ...
                'plot_surfaces', false, ...
                'plot_gamma', false, ...
                'save_output', false, ...
                'use_cache', opts.use_cache, ...
                'cache_dir', opts.cache_dir, ...
                'file_indices', opts.file_indices);
        catch ME
            warning('Skipping pair folder %s: %s', info.folder, ME.message);
            continue;
        end

        pair_model = build_pair_model(info, pair_out);
        if isempty(pair_results)
            pair_results = pair_model;
        else
            pair_results(end + 1, 1) = pair_model; %#ok<AGROW>
        end

        if ~opts.keep_pair_figures
            close_pair_figures(pair_out);
        end
    end

    if isempty(pair_results)
        error('No valid pair analyses were completed.');
    end

    node_ids = unique(reshape([pair_results.agent_ids], 2, []).');
    node_ids = sort(node_ids(:));
    node_index_map = containers.Map(num2cell(node_ids), num2cell(1:numel(node_ids)));

    for k = 1:numel(pair_results)
        pair_results(k).source_idx = node_index_map(pair_results(k).source_agent_id);
        pair_results(k).target_idx = node_index_map(pair_results(k).target_agent_id);
    end

    if isempty(opts.initial_phases)
        phi0 = 2*pi*rand(numel(node_ids), 1);
    else
        phi0 = opts.initial_phases(:);
        if numel(phi0) ~= numel(node_ids)
            error('initial_phases must contain exactly %d values.', numel(node_ids));
        end
    end

    omega_rad_s = opts.omega_rad_s * ones(numel(node_ids), 1);
    time = (0:opts.simulation_dt:opts.simulation_duration_sec).';
    phase = nan(numel(time), numel(node_ids));
    phase(1, :) = phi0.';

    for t_idx = 1:(numel(time) - 1)
        dphi = compute_phase_velocity(phase(t_idx, :).', omega_rad_s, pair_results, opts.sigma);
        phase(t_idx + 1, :) = phase(t_idx, :) + opts.simulation_dt * dphi.';
    end

    reference_agent_id = opts.reference_agent_id;
    if isempty(reference_agent_id)
        reference_agent_id = node_ids(1);
    end
    reference_idx = find(node_ids == reference_agent_id, 1, 'first');
    if isempty(reference_idx)
        error('reference_agent_id %d is not present in the simulated node set.', reference_agent_id);
    end

    relative_phase = wrap_to_pi(phase - phase(:, reference_idx));

    figures = struct();
    figures.relative_phase = plot_relative_phase_trajectories(time, relative_phase, node_ids, reference_agent_id);
    figures.gamma_functions = plot_all_gamma_functions(pair_results);
    if opts.plot_absolute_phases
        figures.absolute_phase = plot_absolute_phases(time, phase, node_ids);
    else
        figures.absolute_phase = [];
    end

    out = struct();
    out.round_dir = round_dir;
    out.M = M;
    out.N = N;
    out.options = opts;
    out.node_ids = node_ids;
    out.reference_agent_id = reference_agent_id;
    out.omega_rad_s = omega_rad_s;
    out.time = time;
    out.phase = phase;
    out.relative_phase = relative_phase;
    out.pair_results = pair_results;
    out.figures = figures;

    if opts.save_output
        out.export = save_outputs(out, opts);
    end
end

function opts = parse_options(default_sigma, varargin)
    p = inputParser;
    addParameter(p, 'analysis_start_sec', 10, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
    addParameter(p, 'analysis_duration_sec', 80, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'sample_dt', 0.01, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'sigma', default_sigma, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
    addParameter(p, 'n_psi', 501, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 3);
    addParameter(p, 'n_theta', 2001, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 3);
    addParameter(p, 'signal_column', 'a2', @(x) ischar(x) || isstring(x));
    addParameter(p, 'normalize_signal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'tail_percent', 10, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x < 50);
    addParameter(p, 'clip_normalized_signal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'clip_limit', 0.5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'simulation_duration_sec', 120, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'simulation_dt', 0.01, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'omega_rad_s', 2.5*pi, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
    addParameter(p, 'initial_phases', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'reference_agent_id', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isfinite(x)));
    addParameter(p, 'plot_absolute_phases', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'keep_pair_figures', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'save_output', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'use_cache', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_indices', [], @(x) isempty(x) || isnumeric(x));
    parse(p, varargin{:});

    opts = p.Results;
    opts.signal_column = char(opts.signal_column);
    opts.normalize_signal = logical(opts.normalize_signal);
    opts.clip_normalized_signal = logical(opts.clip_normalized_signal);
    opts.plot_absolute_phases = logical(opts.plot_absolute_phases);
    opts.keep_pair_figures = logical(opts.keep_pair_figures);
    opts.save_output = logical(opts.save_output);
    opts.use_cache = logical(opts.use_cache);
    opts.output_dir = char(opts.output_dir);
    opts.cache_dir = char(opts.cache_dir);
    opts.file_indices = double(opts.file_indices);
end

function pair_infos = list_pair_folders(round_dir)
    if isstring(round_dir)
        round_dir = char(round_dir);
    end
    if ~isfolder(round_dir)
        candidate = fullfile(pwd, round_dir);
        if isfolder(candidate)
            round_dir = candidate;
        else
            error('round_dir was not found: %s', round_dir);
        end
    end

    dirs = dir(round_dir);
    dirs = dirs([dirs.isdir]);
    pair_infos = struct('name', {}, 'folder', {}, 'agent_ids', {});
    for k = 1:numel(dirs)
        name = dirs(k).name;
        tokens = regexp(name, '^(\d+)-(\d+)$', 'tokens', 'once');
        if isempty(tokens)
            continue;
        end
        folder = fullfile(dirs(k).folder, name);
        if isempty(dir(fullfile(folder, '*.csv')))
            continue;
        end

        pair_infos(end + 1, 1) = struct( ...
            'name', name, ...
            'folder', folder, ...
            'agent_ids', [str2double(tokens{1}), str2double(tokens{2})]); %#ok<AGROW>
    end

    if ~isempty(pair_infos)
        [~, order] = sort({pair_infos.name});
        pair_infos = pair_infos(order);
    end
end

function pair_model = build_pair_model(info, pair_out)
    gamma1_fit = fit_first_harmonic(pair_out.psi, pair_out.gamma1);
    gamma2_fit = fit_first_harmonic(pair_out.psi, pair_out.gamma2_minus_psi);

    pair_model = struct();
    pair_model.pair_name = info.name;
    pair_model.pair_folder = info.folder;
    pair_model.agent_ids = info.agent_ids;
    pair_model.source_agent_id = info.agent_ids(2);
    pair_model.target_agent_id = info.agent_ids(1);
    pair_model.psi = pair_out.psi(:);
    pair_model.gamma1 = pair_out.gamma1(:);
    pair_model.gamma2_minus_psi = pair_out.gamma2_minus_psi(:);
    pair_model.gamma1_fit = gamma1_fit;
    pair_model.gamma2_fit = gamma2_fit;
    pair_model.delta_omega = pair_out.delta_omega;
    pair_model.sigma = pair_out.sigma;
    pair_model.n_points = numel(pair_out.point_cloud.phi1);
    pair_model.fit_s1_rmse = pair_out.fit_s1.rmse;
    pair_model.fit_s2_rmse = pair_out.fit_s2.rmse;
    pair_model.first_pair = struct( ...
        'source_agent_id', info.agent_ids(2), ...
        'target_agent_id', info.agent_ids(1), ...
        'fit', gamma1_fit, ...
        'name', 'Gamma_1');
    pair_model.second_pair = struct( ...
        'source_agent_id', info.agent_ids(1), ...
        'target_agent_id', info.agent_ids(2), ...
        'fit', gamma2_fit, ...
        'name', 'Gamma_2_minus_psi');
end

function dphi = compute_phase_velocity(phi, omega_rad_s, pair_results, sigma)
    dphi = omega_rad_s(:);
    for k = 1:numel(pair_results)
        pair = pair_results(k);
        psi = wrap_to_pi(phi(pair.target_idx) - phi(pair.source_idx));
        gamma1_value = interp1(pair.psi, pair.gamma1, psi, 'linear', 'extrap');
        gamma2_value = interp1(pair.psi, pair.gamma2_minus_psi, psi, 'linear', 'extrap');
        dphi(pair.target_idx) = dphi(pair.target_idx) + sigma * gamma1_value;
        dphi(pair.source_idx) = dphi(pair.source_idx) + sigma * gamma2_value;
    end
end

function value = evaluate_first_harmonic(fit, psi)
    value = fit.bias + fit.sin_coefficient * sin(psi) + fit.cos_coefficient * cos(psi);
end

function fit = fit_first_harmonic(psi, gamma_values)
    psi = psi(:);
    gamma_values = gamma_values(:);
    valid = isfinite(psi) & isfinite(gamma_values);
    psi = psi(valid);
    gamma_values = gamma_values(valid);
    if numel(psi) < 3
        error('At least three finite samples are required for sine fitting.');
    end

    X = [ones(size(psi)), sin(psi), cos(psi)];
    coeff = X \ gamma_values;
    y_fit = X * coeff;
    residual = gamma_values - y_fit;
    centered = gamma_values - mean(gamma_values);
    ss_res = sum(residual .^ 2);
    ss_tot = sum(centered .^ 2);

    fit = struct();
    fit.bias = coeff(1);
    fit.sin_coefficient = coeff(2);
    fit.cos_coefficient = coeff(3);
    fit.amplitude = hypot(coeff(2), coeff(3));
    fit.phase_rad = atan2(coeff(3), coeff(2));
    fit.rmse = sqrt(mean(residual .^ 2));
    fit.r2 = 1 - ss_res / max(ss_tot, eps);
end

function fig = plot_relative_phase_trajectories(time, relative_phase, node_ids, reference_agent_id)
    fig = figure('Color', 'w', 'Name', 'Relative phase simulation');
    ax = axes('Parent', fig);
    hold(ax, 'on');

    colors = lines(numel(node_ids));
    for k = 1:numel(node_ids)
        plot(ax, time, relative_phase(:, k), 'LineWidth', 1.5, 'Color', colors(k, :), ...
            'DisplayName', sprintf('ID %d', node_ids(k)));
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlabel(ax, 'Time (s)');
    ylabel(ax, sprintf('phi_j - phi_%d', reference_agent_id));
    title(ax, 'Simulated relative phase differences');
    legend(ax, 'Location', 'best');
end

function fig = plot_absolute_phases(time, phase, node_ids)
    fig = figure('Color', 'w', 'Name', 'Absolute phase simulation');
    ax = axes('Parent', fig);
    hold(ax, 'on');

    colors = lines(numel(node_ids));
    for k = 1:numel(node_ids)
        plot(ax, time, phase(:, k), 'LineWidth', 1.2, 'Color', colors(k, :), ...
            'DisplayName', sprintf('ID %d: \phi', node_ids(k)));
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlabel(ax, 'Time (s)');
    ylabel(ax, '\phi_j');
    title(ax, 'Simulated oscillator phases');
    legend(ax, 'Location', 'best');
end

function fig = plot_all_gamma_functions(pair_results)
    fig = figure('Color', 'w', 'Name', 'All Gamma functions used in simulation');
    n_pairs = numel(pair_results);
    tiledlayout(fig, n_pairs, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for k = 1:n_pairs
        pair = pair_results(k);
        ax = nexttile;
        hold(ax, 'on');
        plot(ax, pair.psi, pair.gamma1, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('\\Gamma_{%d<-%d}(\\psi)', pair.target_agent_id, pair.source_agent_id));
        plot(ax, pair.psi, pair.gamma2_minus_psi, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('\\Gamma_{%d<-%d}(-\\psi)', pair.source_agent_id, pair.target_agent_id));
        grid(ax, 'on');
        box(ax, 'on');
        xlim(ax, [-pi, pi]);
        xticks(ax, [-pi, -pi/2, 0, pi/2, pi]);
        xticklabels(ax, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
        ylabel(ax, '\Gamma');
        title(ax, sprintf('Pair %d-%d', pair.agent_ids(1), pair.agent_ids(2)), 'Interpreter', 'none');
        if k == 1
            legend(ax, 'Location', 'best');
        end
    end
    xlabel(ax, '\psi');
end

function phase_wrapped = wrap_to_pi(phase)
    phase_wrapped = atan2(sin(phase), cos(phase));
end

function export = save_outputs(out, opts)
    output_dir = opts.output_dir;
    if isempty(output_dir)
        output_dir = fullfile(out.round_dir, 'relative_phase_sim_exports');
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    csv_path = fullfile(output_dir, 'relative_phase_sim.csv');

    final_relative_phase = out.relative_phase(end, :);
    relative_table = array2table(final_relative_phase, ...
        'VariableNames', arrayfun(@(id) sprintf('phi_%d_minus_ref', id), out.node_ids, 'UniformOutput', false));
    writetable(relative_table, csv_path);

    export = struct('output_dir', output_dir, 'csv_path', csv_path);
end

function close_pair_figures(pair_out)
    if ~isfield(pair_out, 'figures') || ~isstruct(pair_out.figures)
        return;
    end

    fields = fieldnames(pair_out.figures);
    for k = 1:numel(fields)
        h = pair_out.figures.(fields{k});
        if ~isempty(h) && all(ishandle(h))
            close(h);
        end
    end
end
