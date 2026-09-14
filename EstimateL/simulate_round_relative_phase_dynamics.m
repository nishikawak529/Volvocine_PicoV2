function out = simulate_round_relative_phase_dynamics(round_dir, M, varargin)
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
%   out = simulate_round_relative_phase_dynamics(fullfile('EstimateL','Round'), 10, ...
%       'simulation_duration_sec', 120, 'simulation_dt', 0.01);

    % --- Display-only agent ID offset for publication plots ---
    % Set to 0 to show raw ids, or -6 to display 7->1, 8->2, ..., 10->4.
    agent_display_offset = -6;

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'SStick');
    end
    round_dir = resolve_dataset_directory(round_dir);

    if nargin < 2 || isempty(M)
        M = 10;
    end

    default_sigma = -5;
    default_remove_gamma_bias = true; % Set to true to subtract the mean (bias) from Gamma functions
    default_subtract_self_profile = true; % Set to true to subtract mean self-profile before Gamma calculation
    default_add_self_feedback = true; % Set to true to add 1 copy of self-profile feedback in simulation when subtract_self_profile is true
    default_use_first_harmonic = false; % Set to true to approximate Gamma with constant + 1st sin wave
    default_use_original_system = true; % Set to true to simulate the original 2D dynamics instead of phase-averaged Gamma dynamics
    default_plot_gamma = false; % Set to true to plot the Gamma functions used in simulation
    default_plot_relative_phase = true; % Set to true to plot relative phase trajectories
    default_plot_mode_amplitudes = true; % Set to true to plot SVD/PMD mode order parameter amplitudes |Z_l(t)|
    default_plot_agent_input_amp = false; % Set to true to plot agent total weighted input amplitude |G_i(t)|
    default_decomposition = 'svd'; % 'svd' or 'sparse_pmd'
    default_agent_display_offset = -6; % Display ID offset (e.g., -6 to display 7->1, 8->2, ..., 10->4)

    opts = parse_options(default_sigma, default_remove_gamma_bias, default_subtract_self_profile, ...
        default_add_self_feedback, default_use_first_harmonic, default_use_original_system, ...
        default_plot_gamma, default_plot_relative_phase, default_plot_mode_amplitudes, ...
        default_plot_agent_input_amp, default_decomposition, default_agent_display_offset, varargin{:});
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');

    agent_display_offset = opts.agent_display_offset;

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
            pair_out = plot_phase_dynamics_from_csv(csv_pattern, info.agent_ids, M, ...
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
                'file_indices', opts.file_indices, ...
                'subtract_self_profile', opts.subtract_self_profile, ...
                'self_profile_dir', opts.self_profile_dir, ...
                'use_first_harmonic', opts.use_first_harmonic);
        catch ME
            warning('Skipping pair folder %s: %s', info.folder, ME.message);
            continue;
        end

        pair_model = build_pair_model(info, pair_out, opts);
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

    % Load self profiles if subtract_self_profile is true
    node_self_profiles = cell(numel(node_ids), 1);
    if opts.subtract_self_profile
        self_dir = opts.self_profile_dir;
        if isempty(self_dir)
            candidate_dirs = {
                fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), 'agent_self_profiles'), ...
                fullfile(round_dir, 'low_rank_analysis', 'M10', 'agent_self_profiles'), ...
                fullfile('EstimateL', 'Round', 'low_rank_analysis', 'M10', 'agent_self_profiles')
            };
            for c = 1:numel(candidate_dirs)
                if exist(candidate_dirs{c}, 'dir')
                    self_dir = candidate_dirs{c};
                    break;
                end
            end
        end
        for i = 1:numel(node_ids)
            aid = node_ids(i);
            csv_path = fullfile(self_dir, sprintf('agent%d_self_profile_data.csv', aid));
            if isfile(csv_path)
                t = readtable(csv_path);
                node_self_profiles{i} = struct('phi', t.phi, 'val', t.mean_self_profile);
                fprintf('[INFO] Loaded mean self-profile for Agent %d to add back to dynamics.\n', aid);
            else
                warning('Self-profile file not found for Agent %d: %s. Using zero self-profile.', aid, csv_path);
            end
        end
    end

    for t_idx = 1:(numel(time) - 1)
        dphi = compute_phase_velocity(phase(t_idx, :).', omega_rad_s, pair_results, opts.sigma, ...
            opts.subtract_self_profile, opts.add_self_feedback, node_self_profiles, opts.use_original_system);
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

    % --- Mode Decomposition Time-Series Analysis (like simulate_cp_rank1_relative_phase_dynamics.m) ---
    mode_model = load_mode_analysis_model(round_dir, M, node_ids, opts);
    mode_analysis = compute_mode_amplitudes_time_series(phase, mode_model);

    [~, dataset_name] = fileparts(round_dir);
    sim_title = sprintf('Pairwise Phase Dynamics (%s), \\sigma = %.2f', dataset_name, opts.sigma);

    figures = struct();
    if opts.plot_relative_phase
        figures.relative_phase = plot_relative_phase_trajectories(time, relative_phase, node_ids, reference_agent_id, agent_display_offset, sim_title);
    else
        figures.relative_phase = [];
    end

    if ~opts.use_original_system && opts.plot_gamma
        figures.gamma_functions = plot_all_gamma_functions(pair_results, agent_display_offset);
    else
        figures.gamma_functions = [];
    end

    if opts.plot_absolute_phases
        figures.absolute_phase = plot_absolute_phases(time, phase, node_ids, agent_display_offset);
    else
        figures.absolute_phase = [];
    end

    if opts.plot_mode_amplitudes && mode_analysis.available
        figures.mode_amplitudes = plot_mode_order_parameters( ...
            time, mode_analysis, sim_title);
    else
        figures.mode_amplitudes = [];
    end

    if opts.plot_agent_input_amp && mode_analysis.available
        figures.agent_input_amp = plot_agent_input_amplitudes( ...
            time, mode_analysis, node_ids, agent_display_offset, sim_title);
    else
        figures.agent_input_amp = [];
    end

    out = struct();
    out.round_dir = round_dir;
    out.M = M;
    out.options = opts;
    out.node_ids = node_ids;
    out.reference_agent_id = reference_agent_id;
    out.omega_rad_s = omega_rad_s;
    out.time = time;
    out.phase = phase;
    out.relative_phase = relative_phase;
    out.pair_results = pair_results;
    out.mode_model = mode_model;
    out.mode_analysis = mode_analysis;
    out.figures = figures;

    if opts.save_output
        out.export = save_outputs(out, opts);
    end
end

function opts = parse_options(default_sigma, default_remove_gamma_bias, default_subtract_self_profile, ...
        default_add_self_feedback, default_use_first_harmonic, default_use_original_system, ...
        default_plot_gamma, default_plot_relative_phase, default_plot_mode_amplitudes, ...
        default_plot_agent_input_amp, default_decomposition, default_agent_display_offset, varargin)
    p = inputParser;
    addParameter(p, 'analysis_start_sec', 6.5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
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
    addParameter(p, 'simulation_duration_sec', 100, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'simulation_dt', 0.01, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'omega_rad_s', 2.5*pi, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
    addParameter(p, 'initial_phases', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'reference_agent_id', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isfinite(x)));
    addParameter(p, 'plot_relative_phase', default_plot_relative_phase, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_absolute_phases', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_mode_amplitudes', default_plot_mode_amplitudes, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_agent_input_amp', default_plot_agent_input_amp, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'decomposition', default_decomposition, @(x) ischar(x) || isstring(x));
    addParameter(p, 'mode_model_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'agent_display_offset', default_agent_display_offset, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'keep_pair_figures', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'save_output', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'use_cache', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_indices', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'remove_gamma_bias', default_remove_gamma_bias, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'subtract_self_profile', default_subtract_self_profile, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'self_profile_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'add_self_feedback', default_add_self_feedback, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'use_first_harmonic', default_use_first_harmonic, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'use_original_system', default_use_original_system, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_gamma', default_plot_gamma, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});

    opts = p.Results;
    opts.signal_column = char(opts.signal_column);
    opts.normalize_signal = logical(opts.normalize_signal);
    opts.clip_normalized_signal = logical(opts.clip_normalized_signal);
    opts.plot_relative_phase = logical(opts.plot_relative_phase);
    opts.plot_absolute_phases = logical(opts.plot_absolute_phases);
    opts.plot_mode_amplitudes = logical(opts.plot_mode_amplitudes);
    opts.plot_agent_input_amp = logical(opts.plot_agent_input_amp);
    opts.decomposition = char(opts.decomposition);
    opts.mode_model_dir = char(opts.mode_model_dir);
    opts.keep_pair_figures = logical(opts.keep_pair_figures);
    opts.save_output = logical(opts.save_output);
    opts.use_cache = logical(opts.use_cache);
    opts.output_dir = char(opts.output_dir);
    opts.cache_dir = char(opts.cache_dir);
    opts.file_indices = double(opts.file_indices);
    opts.remove_gamma_bias = logical(opts.remove_gamma_bias);
    opts.subtract_self_profile = logical(opts.subtract_self_profile);
    opts.self_profile_dir = char(opts.self_profile_dir);
    opts.add_self_feedback = logical(opts.add_self_feedback);
    opts.use_first_harmonic = logical(opts.use_first_harmonic);
    opts.use_original_system = logical(opts.use_original_system);
    opts.plot_gamma = logical(opts.plot_gamma);
end

function pair_infos = list_pair_folders(round_dir)
    if isstring(round_dir)
        round_dir = char(round_dir);
    end
    if ~isfolder(round_dir)
        round_dir = resolve_dataset_directory(round_dir);
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

function pair_model = build_pair_model(info, pair_out, opts)
    gamma1 = pair_out.gamma1(:);
    gamma2_minus_psi = pair_out.gamma2_minus_psi(:);
    if opts.remove_gamma_bias
        gamma1 = gamma1 - mean(gamma1, 'omitnan');
        gamma2_minus_psi = gamma2_minus_psi - mean(gamma2_minus_psi, 'omitnan');
    end

    gamma1_fit = fit_first_harmonic(pair_out.psi, gamma1);
    gamma2_fit = fit_first_harmonic(pair_out.psi, gamma2_minus_psi);

    pair_model = struct();
    pair_model.pair_name = info.name;
    pair_model.pair_folder = info.folder;
    pair_model.agent_ids = info.agent_ids;
    pair_model.source_agent_id = info.agent_ids(2);
    pair_model.target_agent_id = info.agent_ids(1);
    pair_model.psi = pair_out.psi(:);
    pair_model.gamma1 = gamma1;
    pair_model.gamma2_minus_psi = gamma2_minus_psi;
    pair_model.gamma1_fit = gamma1_fit;
    pair_model.gamma2_fit = gamma2_fit;
    pair_model.fit_s1 = pair_out.fit_s1; % Save original 2D Fourier fit for target s1
    pair_model.fit_s2 = pair_out.fit_s2; % Save original 2D Fourier fit for target s2
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

function dphi = compute_phase_velocity(phi, omega_rad_s, pair_results, sigma, subtract_self_profile, add_self_feedback, node_self_profiles, use_original_system)
    if nargin < 8
        use_original_system = false;
    end
    if nargin < 6
        add_self_feedback = false;
    end
    if nargin < 5
        subtract_self_profile = false;
    end
    dphi = omega_rad_s(:);
    
    if use_original_system
        % -- Original System Dynamics --
        % dphi_j/dt = omega_j + sigma * ( q_j(phi_j)*z(phi_j) + sum_k (s_kj(phi_j, phi_k) - q_j(phi_j))*z(phi_j) )
        % Note: self-profile feedback q_j(phi_j)*z(phi_j) is added at the end of this function if add_self_feedback is true.
        % Here we compute the sum_k (s_kj(phi_j, phi_k) - q_j(phi_j))*z(phi_j) term.
        coupling_sum = zeros(size(phi));
        
        for k = 1:numel(pair_results)
            pair = pair_results(k);
            t_idx = pair.target_idx; % receiver j
            s_idx = pair.source_idx; % sender k
            
            phi_t = phi(t_idx);
            phi_s = phi(s_idx);
            
            % Compute s_kj(phi_j, phi_k) for both directions
            s_value_1 = evaluate_fit_surface_local(phi_t, phi_s, pair.fit_s1);
            s_value_2 = evaluate_fit_surface_local(phi_s, phi_t, pair.fit_s2);
            
            % Get self profiles q_j(phi_j)
            q_t = 0; q_s = 0;
            if subtract_self_profile && ~isempty(node_self_profiles)
                if ~isempty(node_self_profiles{t_idx})
                    q_t = interp1(node_self_profiles{t_idx}.phi, node_self_profiles{t_idx}.val, mod(phi_t, 2*pi), 'linear', 'extrap');
                end
                if ~isempty(node_self_profiles{s_idx})
                    q_s = interp1(node_self_profiles{s_idx}.phi, node_self_profiles{s_idx}.val, mod(phi_s, 2*pi), 'linear', 'extrap');
                end
            end
            
            % Multiply by PRC z(phi) = -sin(phi) and accumulate
            z_t = -sin(phi_t);
            z_s = -sin(phi_s);
            coupling_sum(t_idx) = coupling_sum(t_idx) + (s_value_1 - q_t) * z_t;
            coupling_sum(s_idx) = coupling_sum(s_idx) + (s_value_2 - q_s) * z_s;
        end
        
        dphi = dphi + sigma * coupling_sum;
    else
        % -- Phase-averaged Gamma Coupling System --
        for k = 1:numel(pair_results)
            pair = pair_results(k);
            psi = wrap_to_pi(phi(pair.target_idx) - phi(pair.source_idx));
            gamma1_value = interp1(pair.psi, pair.gamma1, psi, 'linear', 'extrap');
            gamma2_value = interp1(pair.psi, pair.gamma2_minus_psi, psi, 'linear', 'extrap');
            dphi(pair.target_idx) = dphi(pair.target_idx) + sigma * gamma1_value;
            dphi(pair.source_idx) = dphi(pair.source_idx) + sigma * gamma2_value;
        end
    end
    
    % Add self-profile * z(phi) term (1 copy per agent) if add_self_feedback is active
    if nargin >= 6 && add_self_feedback && ~isempty(node_self_profiles)
        for i = 1:numel(phi)
            sp = node_self_profiles{i};
            if ~isempty(sp)
                % wrap phi to [0, 2*pi] for profile interpolation
                phi_wrapped = mod(phi(i), 2*pi);
                q_value = interp1(sp.phi, sp.val, phi_wrapped, 'linear', 'extrap');
                z_value = -sin(phi_wrapped); % PRC: z(theta) = -sin(theta)
                dphi(i) = dphi(i) + sigma * q_value * z_value;
            end
        end
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

function fig = plot_relative_phase_trajectories(time, relative_phase, node_ids, reference_agent_id, agent_display_offset, sim_title)
    if nargin < 5 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end
    if nargin < 6 || isempty(sim_title)
        sim_title = 'Pairwise Phase Dynamics';
    end

    fig = figure('Color', 'w', 'Name', sprintf('Relative phase simulation [%s]', sim_title));
    ax = axes('Parent', fig);
    hold(ax, 'on');

    colors = lines(numel(node_ids));
    for k = 1:numel(node_ids)
        y_val = relative_phase(:, k);
        
        % Insert NaNs where phase jumps across wrapping boundaries (> pi gap)
        for j = 2:numel(y_val)
            if isnan(y_val(j)) || isnan(y_val(j-1))
                continue;
            end
            if abs(y_val(j) - y_val(j-1)) > pi
                y_val(j) = NaN;
            end
        end
        
        plot(ax, time, y_val, 'LineWidth', 1.5, 'Color', colors(k, :), ...
            'DisplayName', sprintf('$$j = %d$$', displayed_agent_id(node_ids(k), agent_display_offset)));
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    ylim(ax, [-pi, pi]);
    yticks(ax, [-pi, -pi/2, 0, pi/2, pi]);
    yticklabels(ax, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
    xlabel(ax, 'Time (s)');
    ylabel(ax, sprintf('$$\\phi_j - \\phi_{%d}$$', displayed_agent_id(reference_agent_id, agent_display_offset)), 'Interpreter', 'latex');
    title(ax, sprintf('Relative Phase Dynamics [%s]', sim_title), 'FontSize', 11);
    legend(ax, 'Location', 'eastoutside', 'Interpreter', 'latex');

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_absolute_phases(time, phase, node_ids, agent_display_offset)
    if nargin < 4 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end

    fig = figure('Color', 'w', 'Name', 'Absolute phase simulation');
    ax = axes('Parent', fig);
    hold(ax, 'on');

    colors = lines(numel(node_ids));
    for k = 1:numel(node_ids)
        plot(ax, time, phase(:, k), 'LineWidth', 1.2, 'Color', colors(k, :), ...
            'DisplayName', sprintf('$$j = %d$$', displayed_agent_id(node_ids(k), agent_display_offset)));
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    xlabel(ax, 'Time (s)');
    ylabel(ax, '$$\\phi_j$$', 'Interpreter', 'latex');
    legend(ax, 'Location', 'best', 'Interpreter', 'latex');

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_all_gamma_functions(pair_results, agent_display_offset)
    if nargin < 2 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end

    n_pairs = numel(pair_results);
    % Taller window for multiple rows of side-by-side plots
    fig = figure('Color', 'w', 'Position', [100, 100, 800, min(1000, 250*n_pairs)], ...
        'Name', 'All Gamma functions used in simulation');
    
    tiledlayout(fig, n_pairs, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for k = 1:n_pairs
        pair = pair_results(k);
        
        % Left subplot: i <- j
        ax1 = nexttile;
        hold(ax1, 'on');
        plot(ax1, pair.psi, pair.gamma1, 'LineWidth', 1.5);
        grid(ax1, 'on');
        box(ax1, 'on');
        xlim(ax1, [-pi, pi]);
        xticks(ax1, [-pi, -pi/2, 0, pi/2, pi]);
        xticklabels(ax1, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
        xlabel(ax1, '$$\\psi$$', 'Interpreter', 'latex');
        ylabel(ax1, sprintf('$$\\Gamma_{%d\\leftarrow%d}(\\psi)$$', displayed_agent_id(pair.target_agent_id, agent_display_offset), displayed_agent_id(pair.source_agent_id, agent_display_offset)), 'Interpreter', 'latex');
        
        % Right subplot: j <- i
        ax2 = nexttile;
        hold(ax2, 'on');
        plot(ax2, pair.psi, pair.gamma2_minus_psi, 'LineWidth', 1.5);
        grid(ax2, 'on');
        box(ax2, 'on');
        xlim(ax2, [-pi, pi]);
        xticks(ax2, [-pi, -pi/2, 0, pi/2, pi]);
        xticklabels(ax2, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
        xlabel(ax2, '$$\\psi$$', 'Interpreter', 'latex');
        ylabel(ax2, sprintf('$$\\Gamma_{%d\\leftarrow%d}(-\\psi)$$', displayed_agent_id(pair.source_agent_id, agent_display_offset), displayed_agent_id(pair.target_agent_id, agent_display_offset)), 'Interpreter', 'latex');
    end
    
    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function display_id = displayed_agent_id(agent_id, offset)
    display_id = agent_id + offset;
end

function phase_wrapped = wrap_to_pi(phase)
    phase_wrapped = atan2(sin(phase), cos(phase));
end

function fig = plot_mode_order_parameters(time, mode_analysis, net_title)
    if nargin < 3 || isempty(net_title)
        net_title = 'Network Model';
    end

    fig = figure('Color', 'w', 'Name', sprintf('Weighted Order Parameter |Z_l(t)| (%s)', net_title));
    ax = axes('Parent', fig);
    hold(ax, 'on');

    K = mode_analysis.num_modes;
    d = mode_analysis.d;
    method = upper(mode_analysis.method);
    colors = lines(K);
    line_handles = gobjects(K, 1);
    mode_legend = cell(K, 1);

    for m = 1:K
        line_handles(m) = plot(ax, time, mode_analysis.Z_abs(:, m), ...
            'LineWidth', 1.6, 'Color', colors(m, :));
        if strcmpi(method, 'SPARSE_PMD')
            mode_legend{m} = sprintf('Sender Mode %d (d_{%d} = %.3f)', m, m, d(m));
        else
            mode_legend{m} = sprintf('Sender Mode %d (\\sigma_{%d} = %.3f)', m, m, d(m));
        end
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    xlabel(ax, 'Time (s)', 'FontSize', 11);
    ylabel(ax, '$$|Z_l(t)| = \left|\sum_j v_{jl} e^{i \phi_j(t)}\right|$$', ...
        'Interpreter', 'latex', 'FontSize', 12);
    title(ax, sprintf('Weighted Order Parameter |Z_l(t)| [%s]', net_title), 'FontSize', 12);
    legend(ax, line_handles, mode_legend, 'Location', 'eastoutside', 'FontSize', 10);

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_agent_input_amplitudes(time, mode_analysis, node_ids, agent_display_offset, net_title)
    if nargin < 4 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end
    if nargin < 5 || isempty(net_title)
        net_title = 'Network Model';
    end

    fig = figure('Color', 'w', 'Name', sprintf('Agent Weighted Input Amplitude |G_i(t)| (%s)', net_title));
    ax = axes('Parent', fig);
    hold(ax, 'on');

    N = numel(node_ids);
    colors = lines(N);
    line_handles = gobjects(N, 1);
    agent_legend = cell(N, 1);

    for i = 1:N
        line_handles(i) = plot(ax, time, mode_analysis.G_abs(:, i), ...
            'LineWidth', 1.5, 'Color', colors(i, :));
        agent_legend{i} = sprintf('ID %d', node_ids(i) + agent_display_offset);
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    xlabel(ax, 'Time (s)', 'FontSize', 11);
    ylabel(ax, '$$|G_i(t)| = \left|\sum_{\ell}\sigma_{\ell}u_{i\ell}Z_{\ell}(t)\right|$$', ...
        'Interpreter', 'latex', 'FontSize', 12);
    title(ax, sprintf('Agent Total Input Amplitude |G_i(t)| [%s]', net_title), 'FontSize', 12);
    legend(ax, line_handles, agent_legend, 'Location', 'eastoutside', 'FontSize', 10);

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function mode_analysis = compute_mode_amplitudes_time_series(phase, mode_model)
    mode_analysis = struct();
    mode_analysis.available = false;

    if isempty(mode_model) || ~isfield(mode_model, 'available') || ~mode_model.available
        return;
    end

    modes = mode_model.network_modes;
    if isempty(modes) || ~isfield(modes, 'P') || ~isfield(modes, 'Q') || ~isfield(modes, 'd')
        return;
    end

    U = modes.P;
    V = modes.Q;
    d = modes.d(:);
    K = numel(d);

    % Complex phasors: exp(1i * phase) (T x N)
    phasors = exp(1i * phase);

    % Weighted order parameter: Z_l(t) = sum_j v_{jl} * exp(1i * phi_j(t))
    Z_complex = phasors * V;     % T x K
    Z_abs = abs(Z_complex);     % T x K

    % Receiver total mode input amplitude: G_i(t) = sum_l sigma_l * u_{il} * Z_l(t)
    G_complex = (Z_complex .* d.') * U.'; % T x N
    G_abs = abs(G_complex);               % T x N

    % Real collective sender signal: X_real (T x K)
    if isfield(mode_model, 'delta') && ~isempty(mode_model.delta)
        B_mat = sqrt(2) * cos(phase - mode_model.delta);
        X_real = B_mat * V;
    else
        X_real = [];
    end

    mode_analysis.available = true;
    mode_analysis.method = modes.method;
    mode_analysis.num_modes = K;
    mode_analysis.d = d;
    mode_analysis.U = U;
    mode_analysis.V = V;
    mode_analysis.Z_complex = Z_complex;
    mode_analysis.Z_abs = Z_abs;
    mode_analysis.G_complex = G_complex;
    mode_analysis.G_abs = G_abs;
    mode_analysis.X_real = X_real;
end

function mode_model = load_mode_analysis_model(round_dir, M, node_ids, opts)
% Load network mode decomposition (SVD / Sparse PMD) corresponding to round_dir.
    mode_model = struct('available', false);

    decomp = opts.decomposition;
    
    % Candidate directories to search for saved low-rank analysis outputs
    candidate_dirs = {};
    if ~isempty(opts.mode_model_dir)
        candidate_dirs{end + 1} = opts.mode_model_dir;
    end

    candidate_subdirs = {
        sprintf('global_joint_cp_rank1_profile_free_network_%s', decomp), ...
        'global_joint_cp_rank1_profile_free_network_svd', ...
        'global_joint_cp_multirank_agentwise_sender_free_network_svd', ...
        'global_joint_cp_rank1'
    };

    m_dirs = {sprintf('M%d', M), 'M10', 'M5'};
    for m_idx = 1:numel(m_dirs)
        base_low_rank = fullfile(round_dir, 'low_rank_analysis', m_dirs{m_idx});
        for s_idx = 1:numel(candidate_subdirs)
            candidate_dirs{end + 1} = fullfile(base_low_rank, candidate_subdirs{s_idx}); %#ok<AGROW>
        end
        candidate_dirs{end + 1} = base_low_rank; %#ok<AGROW>
    end

    % Filter existing directories
    valid_dirs = {};
    for c = 1:numel(candidate_dirs)
        if exist(candidate_dirs{c}, 'dir')
            valid_dirs{end + 1} = candidate_dirs{c}; %#ok<AGROW>
        end
    end
    candidate_dirs = unique(valid_dirs, 'stable');

    if isempty(candidate_dirs)
        warning('No low_rank_analysis directory found under %s. Mode analysis will be skipped.', round_dir);
        return;
    end

    % Search through candidate directories
    for c = 1:numel(candidate_dirs)
        target_dir = candidate_dirs{c};
        
        % 1. Try MAT file
        mat_files = {
            sprintf('rank1_profile_free_network_%s_results.mat', decomp), ...
            'rank1_profile_free_network_svd_results.mat'
        };
        for m_file = 1:numel(mat_files)
            mat_path = fullfile(target_dir, mat_files{m_file});
            if exist(mat_path, 'file')
                try
                    loaded = load(mat_path);
                    model = struct();
                    model.agent_ids = loaded.agent_ids(:).';
                    if isfield(loaded, 'W')
                        model.W = loaded.W;
                    else
                        model.W = [];
                    end
                    if isfield(loaded, 'delta')
                        model.delta = loaded.delta;
                    else
                        model.delta = [];
                    end
                    model.network_modes = struct('P', loaded.P, 'Q', loaded.Q, 'd', loaded.d(:), 'method', loaded.method);
                    
                    aligned = align_mode_model_to_nodes(model, node_ids);
                    if aligned.available
                        mode_model = aligned;
                        fprintf('[INFO] Loaded mode analysis model from MAT: %s\n', mat_path);
                        return;
                    end
                catch
                end
            end
        end

        % 2. Try CSV files
        w_csv = fullfile(target_dir, 'network_coupling_matrix_W.csv');
        delta_csv = fullfile(target_dir, 'sender_phase_shift_delta.csv');
        contrib_csv = fullfile(target_dir, 'agent_svd_contributions.csv');
        modes_csv = fullfile(target_dir, 'network_svd_modes_summary.csv');

        delta_val = [];
        if exist(delta_csv, 'file')
            try
                t_delta = readtable(delta_csv);
                delta_val = t_delta.delta_rad(1);
            catch
            end
        end

        % Option A: Read W matrix
        if exist(w_csv, 'file')
            try
                t_w = readtable(w_csv);
                var_names = t_w.Properties.VariableNames;
                sender_cols = var_names(startsWith(var_names, 'sender_agent_'));
                agent_ids = zeros(1, numel(sender_cols));
                for k = 1:numel(sender_cols)
                    agent_ids(k) = str2double(regexprep(sender_cols{k}, '^sender_agent_', ''));
                end
                W_mat = table2array(t_w(:, sender_cols));

                [U, S_mat, V] = svd(W_mat);
                model = struct();
                model.agent_ids = agent_ids;
                model.W = W_mat;
                model.delta = delta_val;
                model.network_modes = struct('P', U, 'Q', V, 'd', diag(S_mat), 'method', 'svd');

                aligned = align_mode_model_to_nodes(model, node_ids);
                if aligned.available
                    mode_model = aligned;
                    fprintf('[INFO] Loaded mode analysis model from W CSV in %s\n', target_dir);
                    return;
                end
            catch ME
                warning('Failed reading W CSV: %s', ME.message);
            end
        end

        % Option B: Read agent_svd_contributions.csv and network_svd_modes_summary.csv
        if exist(contrib_csv, 'file')
            try
                t_contrib = readtable(contrib_csv);
                agent_ids = t_contrib.agent_id(:).';
                var_names = t_contrib.Properties.VariableNames;
                u_cols = var_names(startsWith(var_names, 'receiver_u_mode'));
                v_cols = var_names(startsWith(var_names, 'sender_v_mode'));
                
                U = table2array(t_contrib(:, u_cols));
                V = table2array(t_contrib(:, v_cols));
                
                d = [];
                if exist(modes_csv, 'file')
                    t_modes = readtable(modes_csv);
                    if ismember('singular_value_sigma', t_modes.Properties.VariableNames)
                        d = t_modes.singular_value_sigma(:);
                    end
                end
                if isempty(d)
                    d = ones(numel(v_cols), 1);
                end

                model = struct();
                model.agent_ids = agent_ids;
                model.W = [];
                model.delta = delta_val;
                model.network_modes = struct('P', U, 'Q', V, 'd', d, 'method', 'svd');

                aligned = align_mode_model_to_nodes(model, node_ids);
                if aligned.available
                    mode_model = aligned;
                    fprintf('[INFO] Loaded mode analysis model from agent_svd_contributions.csv in %s\n', target_dir);
                    return;
                end
            catch ME
                warning('Failed reading agent_svd_contributions.csv: %s', ME.message);
            end
        end
    end

    warning('Could not find or load valid mode decomposition model under %s.', round_dir);
end

function aligned = align_mode_model_to_nodes(model, node_ids)
    aligned = struct('available', false);
    if isempty(model) || ~isfield(model, 'network_modes')
        return;
    end

    model_agent_ids = model.agent_ids(:).';
    node_ids = node_ids(:).';

    [tf, loc] = ismember(node_ids, model_agent_ids);
    if ~all(tf)
        % Not all simulation nodes are present in the mode model
        missing_ids = node_ids(~tf);
        warning('Simulation node IDs %s not found in mode model agent IDs %s.', ...
            mat2str(missing_ids), mat2str(model_agent_ids));
        return;
    end

    % Reorder U, V, and W to match node_ids
    aligned.available = true;
    aligned.agent_ids = node_ids;
    aligned.delta = model.delta;
    
    modes = model.network_modes;
    aligned_P = modes.P(loc, :);
    aligned_Q = modes.Q(loc, :);
    
    aligned.network_modes = struct( ...
        'P', aligned_P, ...
        'Q', aligned_Q, ...
        'd', modes.d, ...
        'method', modes.method);

    if ~isempty(model.W)
        aligned.W = model.W(loc, loc);
    else
        aligned.W = [];
    end
end

function resolved_dir = resolve_dataset_directory(d)
    if ischar(d) || isstring(d)
        d_str = char(d);
    else
        resolved_dir = d;
        return;
    end

    if isfolder(d_str)
        resolved_dir = d_str;
        return;
    end

    this_file = which('simulate_round_relative_phase_dynamics');
    if ~isempty(this_file)
        estimate_dir = fileparts(this_file);
        base_root = fileparts(estimate_dir);
    else
        estimate_dir = fullfile(pwd, 'EstimateL');
        base_root = pwd;
    end

    candidates = {
        fullfile(estimate_dir, d_str), ...
        fullfile(base_root, 'EstimateL', d_str), ...
        fullfile('EstimateL', d_str), ...
        fullfile(pwd, 'EstimateL', d_str), ...
        fullfile(pwd, d_str), ...
        fullfile(base_root, d_str)
    };
    for c = 1:numel(candidates)
        if isfolder(candidates{c})
            resolved_dir = candidates{c};
            return;
        end
    end
    resolved_dir = d_str;
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

    % Also save mode order parameter amplitudes if available
    if isfield(out, 'mode_analysis') && out.mode_analysis.available
        mode_csv_path = fullfile(output_dir, 'mode_order_parameters.csv');
        Z_table = array2table([out.time, out.mode_analysis.Z_abs], ...
            'VariableNames', [{'time_sec'}, arrayfun(@(m) sprintf('mode_%d_abs', m), 1:out.mode_analysis.num_modes, 'UniformOutput', false)]);
        writetable(Z_table, mode_csv_path);
    end

    mat_path = fullfile(output_dir, 'relative_phase_sim_results.mat');
    save(mat_path, 'out', '-v7.3');

    export = struct('output_dir', output_dir, 'csv_path', csv_path, 'mat_path', mat_path);
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

function s_values = evaluate_fit_surface_local(phi1, phi2, fit_result)
    M = fit_result.M;
    coeff = fit_result.coeff;
    z_mean = fit_result.z_mean;
    
    n_samples = numel(phi1);
    n_basis = 1 + 4*M + 4*M*M;
    A = zeros(n_samples, n_basis);
    
    col = 1;
    A(:, col) = 1;
    col = col + 1;
    
    for m = 1:M
        A(:, col) = cos(m * phi1);
        col = col + 1;
        A(:, col) = sin(m * phi1);
        col = col + 1;
    end
    
    for n = 1:M
        A(:, col) = cos(n * phi2);
        col = col + 1;
        A(:, col) = sin(n * phi2);
        col = col + 1;
    end
    
    for m = 1:M
        c1 = cos(m * phi1);
        s1 = sin(m * phi1);
        for n = 1:M
            c2 = cos(n * phi2);
            s2 = sin(n * phi2);
            
            A(:, col) = c1 .* c2;
            col = col + 1;
            A(:, col) = c1 .* s2;
            col = col + 1;
            A(:, col) = s1 .* c2;
            col = col + 1;
            A(:, col) = s1 .* s2;
            col = col + 1;
        end
    end
    
    s_values = A * coeff + z_mean;
end
