function out = plot_phase_dynamics_from_csv(input_path, phase_agent_ids, M, N, varargin)
% Fit s_1(phi1,phi2), s_2(phi1,phi2) from CSV data and plot psi dynamics.
%
% Defaults target the current Round CSV:
%   out = plot_phase_dynamics_from_csv()
%
% Examples:
%   out = plot_phase_dynamics_from_csv('EstimateL\Round\merged_20260701_115558.csv');
%   out = plot_phase_dynamics_from_csv('EstimateL\Round', [7 8], 10, 10);
%   out = plot_phase_dynamics_from_csv('EstimateL\Round', [], 8, 8, 'file_indices', 1);
%
% The phase convention is psi = phi1 - phi2, so
%   Delta omega = mean(omega_1) - mean(omega_2).
%
% The plotted dynamics are
%   dpsi/dt = sigma * (Gamma_1(psi) - Gamma_2(-psi))
%   Gamma_1(psi) = 1/(2*pi) int_0^(2*pi) z(theta) s_1(theta,theta-psi) dtheta
%   Gamma_2(psi) = 1/(2*pi) int_0^(2*pi) z(theta) s_2(theta-psi,theta) dtheta
% with z(theta) = -sin(theta).

    if nargin < 1 || isempty(input_path)
        input_path = fullfile('EstimateL', 'Round', '7-10', '*.csv');
    end
    if nargin < 2
        phase_agent_ids = [];
    end
    if nargin < 3 || isempty(M)
        M = 10;
    end
    if nargin < 4 || isempty(N)
        N = M;
    end

    default_sigma = 7;

    opts = parse_options(default_sigma, varargin{:});
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');
    validateattributes(N, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'N');

    csv_paths = list_csv_paths(input_path, opts.file_indices);
    if isempty(csv_paths)
        error('No CSV files were found for input_path: %s', input_path);
    end

    if isempty(phase_agent_ids)
        phase_agent_ids = detect_default_phase_agents(csv_paths);
    else
        phase_agent_ids = phase_agent_ids(:).';
        if numel(phase_agent_ids) ~= 2 || phase_agent_ids(1) == phase_agent_ids(2)
            error('phase_agent_ids must contain exactly two distinct agent IDs.');
        end
    end

    [cache_path, cache_key] = resolve_cache_path(input_path, csv_paths, phase_agent_ids, M, N, opts);
    cache_data = [];
    if opts.use_cache
        cache_data = load_cache(cache_path);
        if ~isempty(cache_data) && (~isfield(cache_data, 'cache_key') || ~strcmp(cache_data.cache_key, cache_key))
            cache_data = [];
        elseif ~isempty(cache_data)
            fprintf('[INFO] Loaded cached dynamics from %s\n', cache_path);
        end
    end

    if isempty(cache_data)
        point_cloud = collect_point_cloud(csv_paths, phase_agent_ids, opts);
        fit_s1 = fit_double_fourier_scatter( ...
            point_cloud.phi1, point_cloud.phi2, point_cloud.s1, M, N, ...
            sprintf('s_1, agent %d', phase_agent_ids(1)));
        fit_s2 = fit_double_fourier_scatter( ...
            point_cloud.phi1, point_cloud.phi2, point_cloud.s2, M, N, ...
            sprintf('s_2, agent %d', phase_agent_ids(2)));

        psi = linspace(-pi, pi, opts.n_psi).';
        theta = linspace(0, 2*pi, opts.n_theta).';
        z_theta = -sin(theta);

        [gamma1, gamma2_minus_psi] = compute_gamma_for_dynamics(psi, theta, z_theta, fit_s1, fit_s2);
        omega_rad_s = mean(point_cloud.omega_rad_s, 1, 'omitnan');
        delta_omega = 0;
        psi_dot = compute_psi_dot(opts.sigma, gamma1, gamma2_minus_psi);

        cache_data = struct();
        cache_data.cache_version = 2;
        cache_data.cache_key = cache_key;
        cache_data.cache_path = cache_path;
        cache_data.input_path = input_path;
        cache_data.csv_paths = csv_paths;
        cache_data.phase_agent_ids = phase_agent_ids;
        cache_data.M = M;
        cache_data.N = N;
        cache_data.omega_rad_s = omega_rad_s;
        cache_data.delta_omega = delta_omega;
        cache_data.psi = psi;
        cache_data.gamma1 = gamma1;
        cache_data.gamma2_minus_psi = gamma2_minus_psi;
        cache_data.fit_s1 = fit_s1;
        cache_data.fit_s2 = fit_s2;
        cache_data.point_cloud = point_cloud;
        if opts.use_cache
            save_cache(cache_path, cache_data);
            fprintf('[INFO] Saved cached dynamics to %s\n', cache_path);
        end
    else
        point_cloud = cache_data.point_cloud;
        fit_s1 = cache_data.fit_s1;
        fit_s2 = cache_data.fit_s2;
        psi = cache_data.psi;
        gamma1 = cache_data.gamma1;
        gamma2_minus_psi = cache_data.gamma2_minus_psi;
        omega_rad_s = cache_data.omega_rad_s;
        delta_omega = 0;
        psi_dot = compute_psi_dot(opts.sigma, gamma1, gamma2_minus_psi);
    end

    figures = struct();
    if opts.plot_gamma
        figures.gamma = plot_gamma_components(psi, gamma1, gamma2_minus_psi, psi_dot, opts.sigma, phase_agent_ids);
    else
        figures.gamma = [];
    end
    figures.psi_dynamics = [];
    if opts.plot_surfaces
        figures.surfaces = plot_fitted_surfaces(fit_s1, fit_s2, phase_agent_ids);
    else
        figures.surfaces = [];
    end

    out = struct();
    out.input_path = input_path;
    out.csv_paths = csv_paths;
    out.phase_agent_ids = phase_agent_ids;
    out.signal_column = opts.signal_column;
    out.M = M;
    out.N = N;
    out.sigma = opts.sigma;
    out.phase_sensitivity = 'z(phi) = -sin(phi)';
    out.psi_definition = 'psi = phi1 - phi2';
    out.omega_rad_s = omega_rad_s;
    out.delta_omega = delta_omega;
    out.psi = psi;
    out.gamma1 = gamma1;
    out.gamma2_minus_psi = gamma2_minus_psi;
    out.psi_dot = psi_dot;
    out.fit_s1 = fit_s1;
    out.fit_s2 = fit_s2;
    out.point_cloud = point_cloud;
    out.cache = struct('path', cache_path, 'key', cache_key, 'used', ~isempty(cache_data) && opts.use_cache);
    out.figures = figures;

    if opts.save_output
        out.export = save_outputs(out, opts);
    end

    fprintf('[INFO] Fit completed with %d samples from %d file(s).\n', ...
        numel(point_cloud.phi1), numel(csv_paths));
    fprintf('[INFO] Agent IDs: s1 -> %d, s2 -> %d\n', phase_agent_ids(1), phase_agent_ids(2));
    fprintf('[INFO] omega1 = %.12g rad/s, omega2 = %.12g rad/s, Delta omega = %.12g rad/s\n', ...
        omega_rad_s(1), omega_rad_s(2), delta_omega);
    fprintf('[INFO] RMSE: s1 = %.6g, s2 = %.6g\n', fit_s1.rmse, fit_s2.rmse);
end

function opts = parse_options(default_sigma, varargin)
    p = inputParser;
    addParameter(p, 'sigma', default_sigma, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
    addParameter(p, 'sample_dt', 0.01, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'analysis_start_sec', 10, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
    addParameter(p, 'analysis_duration_sec', 80, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'n_psi', 501, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 3);
    addParameter(p, 'n_theta', 2001, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 3);
    addParameter(p, 'signal_column', 'a2', @(x) ischar(x) || isstring(x));
    addParameter(p, 'normalize_signal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'tail_percent', 10, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x < 50);
    addParameter(p, 'clip_normalized_signal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'clip_limit', 0.5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'plot_surfaces', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_gamma', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'save_output', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'use_cache', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_indices', [], @(x) isempty(x) || isnumeric(x));
    parse(p, varargin{:});

    opts = p.Results;
    opts.n_psi = round(opts.n_psi);
    opts.n_theta = round(opts.n_theta);
    opts.signal_column = char(opts.signal_column);
    opts.normalize_signal = logical(opts.normalize_signal);
    opts.clip_normalized_signal = logical(opts.clip_normalized_signal);
    opts.plot_surfaces = logical(opts.plot_surfaces);
    opts.plot_gamma = logical(opts.plot_gamma);
    opts.save_output = logical(opts.save_output);
    opts.output_dir = char(opts.output_dir);
    opts.use_cache = logical(opts.use_cache);
    opts.cache_dir = char(opts.cache_dir);
end

function csv_paths = list_csv_paths(input_path, file_indices)
    if isstring(input_path)
        input_path = char(input_path);
    end

    if has_wildcard(input_path)
        files = dir(input_path);
        files = files(~[files.isdir]);
        csv_paths = sort(cellfun(@(folder, name) fullfile(folder, name), ...
            {files.folder}, {files.name}, 'UniformOutput', false));
        csv_paths = apply_file_indices(csv_paths, file_indices);
        return;
    end

    if isfile(input_path)
        csv_paths = {input_path};
        return;
    end

    if ~isfolder(input_path)
        candidate = fullfile(pwd, input_path);
        if isfile(candidate)
            csv_paths = {candidate};
            return;
        elseif isfolder(candidate)
            input_path = candidate;
        else
            error('input_path is neither a file nor a directory: %s', input_path);
        end
    end

    files = dir(fullfile(input_path, '*.csv'));
    files = files(~[files.isdir]);
    csv_paths = sort(cellfun(@(folder, name) fullfile(folder, name), ...
        {files.folder}, {files.name}, 'UniformOutput', false));
    csv_paths = apply_file_indices(csv_paths, file_indices);
end

function tf = has_wildcard(path_text)
    tf = ~isempty(regexp(path_text, '[\*\?]', 'once'));
end

function csv_paths = apply_file_indices(csv_paths, file_indices)
    if isempty(csv_paths) || isempty(file_indices)
        return;
    end

    file_indices = file_indices(:).';
    if any(file_indices < 1) || any(file_indices > numel(csv_paths)) || any(file_indices ~= round(file_indices))
        error('file_indices must contain integers in 1..%d.', numel(csv_paths));
    end
    csv_paths = csv_paths(file_indices);
end

function phase_agent_ids = detect_default_phase_agents(csv_paths)
    for i = 1:numel(csv_paths)
        T = readtable(csv_paths{i});
        if ~ismember('agent_id', T.Properties.VariableNames)
            continue;
        end

        agents = unique(T.agent_id, 'sorted').';
        agents = agents(agents ~= 99);
        if numel(agents) >= 2
            phase_agent_ids = agents(1:2);
            return;
        end
    end

    error('Could not detect two non-99 agent IDs from the selected CSV files.');
end

function point_cloud = collect_point_cloud(csv_paths, phase_agent_ids, opts)
    phi1_all = [];
    phi2_all = [];
    s1_all = [];
    s2_all = [];
    time_all = [];
    file_index_all = [];
    omega_rad_s = nan(numel(csv_paths), 2);
    per_file = struct('file_path', {}, 'window_start_abs', {}, 'window_end_abs', {}, ...
        'n_points', {}, 'omega_rad_s', {});
    skipped_files = struct('file_path', {}, 'reason', {});

    for file_idx = 1:numel(csv_paths)
        csv_path = csv_paths{file_idx};
        try
            [points, meta] = compute_points_for_csv(csv_path, phase_agent_ids, opts);
        catch ME
            warning('Skipping %s: %s', csv_path, ME.message);
            skipped_files(end + 1) = struct('file_path', csv_path, 'reason', ME.message); %#ok<AGROW>
            continue;
        end

        phi1_all = [phi1_all; points.phi1(:)]; %#ok<AGROW>
        phi2_all = [phi2_all; points.phi2(:)]; %#ok<AGROW>
        s1_all = [s1_all; points.s1(:)]; %#ok<AGROW>
        s2_all = [s2_all; points.s2(:)]; %#ok<AGROW>
        time_all = [time_all; points.time(:)]; %#ok<AGROW>
        file_index_all = [file_index_all; file_idx * ones(numel(points.time), 1)]; %#ok<AGROW>
        omega_rad_s(file_idx, :) = meta.omega_rad_s;
        per_file(end + 1) = meta; %#ok<AGROW>
    end

    if isempty(phi1_all)
        error('No valid overlapping samples were found in the selected CSV file(s).');
    end

    point_cloud = struct();
    point_cloud.phi1 = phi1_all;
    point_cloud.phi2 = phi2_all;
    point_cloud.s1 = s1_all;
    point_cloud.s2 = s2_all;
    point_cloud.time = time_all;
    point_cloud.file_index = file_index_all;
    point_cloud.omega_rad_s = omega_rad_s;
    point_cloud.per_file = per_file;
    point_cloud.skipped_files = skipped_files;
end

function [cache_path, cache_key] = resolve_cache_path(input_path, csv_paths, phase_agent_ids, M, N, opts)
    cache_dir = opts.cache_dir;
    if isempty(cache_dir)
        if isfolder(input_path)
            cache_dir = input_path;
        else
            [cache_dir, ~, ~] = fileparts(csv_paths{1});
        end
    end

    if isempty(cache_dir)
        cache_dir = pwd;
    end

    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end

    cache_key = compute_cache_key(csv_paths, phase_agent_ids, M, N, opts);
    cache_path = fullfile(cache_dir, sprintf('phase_dynamics_cache_%s.mat', cache_key));
end

function cache_key = compute_cache_key(csv_paths, phase_agent_ids, M, N, opts)
    parts = {};
    parts{end + 1} = sprintf('phase_agents=%s', mat2str(phase_agent_ids));
    parts{end + 1} = sprintf('M=%d|N=%d', M, N);
    parts{end + 1} = sprintf('sample_dt=%.15g', opts.sample_dt);
    parts{end + 1} = sprintf('analysis_start_sec=%.15g', opts.analysis_start_sec);
    parts{end + 1} = sprintf('analysis_duration_sec=%.15g', opts.analysis_duration_sec);
    parts{end + 1} = sprintf('signal_column=%s', opts.signal_column);
    parts{end + 1} = sprintf('normalize_signal=%d', opts.normalize_signal);
    parts{end + 1} = sprintf('tail_percent=%.15g', opts.tail_percent);
    parts{end + 1} = sprintf('clip_normalized_signal=%d', opts.clip_normalized_signal);
    parts{end + 1} = sprintf('clip_limit=%.15g', opts.clip_limit);
    for i = 1:numel(csv_paths)
        info = dir(csv_paths{i});
        if isempty(info)
            parts{end + 1} = sprintf('missing=%s', csv_paths{i}); %#ok<AGROW>
        else
            parts{end + 1} = sprintf('%s|%d|%.15g', csv_paths{i}, info.bytes, info.datenum);
        end
    end
    cache_key = md5_hex(strjoin(parts, '\n'));
end

function psi_dot = compute_psi_dot(sigma, gamma1, gamma2_minus_psi)
    psi_dot = sigma * (gamma1 - gamma2_minus_psi);
end

function hex_text = md5_hex(text_value)
    md = java.security.MessageDigest.getInstance('MD5');
    md.update(uint8(text_value(:)));
    hash = typecast(md.digest(), 'uint8');
    hex_text = lower(reshape(dec2hex(hash, 2).', 1, []));
end

function cache_data = load_cache(cache_path)
    if ~isfile(cache_path)
        cache_data = [];
        return;
    end

    cached = load(cache_path, 'cache_data');
    if isfield(cached, 'cache_data')
        cache_data = cached.cache_data;
    else
        cache_data = [];
    end
end

function save_cache(cache_path, cache_data)
    save(cache_path, 'cache_data', '-v7.3');
end

function [points, meta] = compute_points_for_csv(csv_path, phase_agent_ids, opts)
    series_by_agent = load_agent_series_from_csv(csv_path, phase_agent_ids, opts.signal_column);

    overlap_start = -inf;
    overlap_end = inf;
    for k = 1:numel(phase_agent_ids)
        aid = phase_agent_ids(k);
        series = series_by_agent(aid);
        if isempty(series.time)
            error('No valid samples found for agent %d.', aid);
        end
        overlap_start = max(overlap_start, min(series.time));
        overlap_end = min(overlap_end, max(series.time));
    end
    if overlap_end <= overlap_start
        error('Selected agents do not share an overlapping time range.');
    end

    window_start_abs = overlap_start + opts.analysis_start_sec;
    if isfinite(opts.analysis_duration_sec)
        window_end_abs = min(overlap_end, window_start_abs + opts.analysis_duration_sec);
    else
        window_end_abs = overlap_end;
    end
    if window_end_abs <= window_start_abs
        error('Requested analysis window is outside the common overlap.');
    end

    n_steps = floor((window_end_abs - window_start_abs) / opts.sample_dt);
    time_abs = window_start_abs + (0:n_steps).' * opts.sample_dt;
    if numel(time_abs) < 2
        error('Analysis window is too short after applying sample_dt.');
    end

    a0_1 = interp1(series_by_agent(phase_agent_ids(1)).time, ...
        series_by_agent(phase_agent_ids(1)).a0_corr, time_abs, 'linear', NaN);
    a0_2 = interp1(series_by_agent(phase_agent_ids(2)).time, ...
        series_by_agent(phase_agent_ids(2)).a0_corr, time_abs, 'linear', NaN);
    signal_1 = interp1(series_by_agent(phase_agent_ids(1)).time, ...
        series_by_agent(phase_agent_ids(1)).signal, time_abs, 'linear', NaN);
    signal_2 = interp1(series_by_agent(phase_agent_ids(2)).time, ...
        series_by_agent(phase_agent_ids(2)).signal, time_abs, 'linear', NaN);

    valid = isfinite(a0_1) & isfinite(a0_2) & isfinite(signal_1) & isfinite(signal_2);
    time_abs = time_abs(valid);
    a0_1 = a0_1(valid);
    a0_2 = a0_2(valid);
    signal_1 = signal_1(valid);
    signal_2 = signal_2(valid);
    if numel(time_abs) < 2
        error('No overlapping interpolated samples remained after removing NaN/Inf.');
    end

    if opts.normalize_signal
        signal_1 = normalize_by_agent_percentile_span( ...
            signal_1, series_by_agent(phase_agent_ids(1)).signal, opts.tail_percent);
        signal_2 = normalize_by_agent_percentile_span( ...
            signal_2, series_by_agent(phase_agent_ids(2)).signal, opts.tail_percent);
        if opts.clip_normalized_signal
            signal_1 = clip_values(signal_1, -opts.clip_limit, opts.clip_limit);
            signal_2 = clip_values(signal_2, -opts.clip_limit, opts.clip_limit);
        end
    end

    omega_rad_s = nan(1, 2);
    duration_used_s = window_end_abs - window_start_abs;
    for k = 1:numel(phase_agent_ids)
        aid = phase_agent_ids(k);
        theta_vals = double(series_by_agent(aid).a0_corr(:)) * (2*pi/256);
        theta_start = interp1(series_by_agent(aid).time, theta_vals, window_start_abs, 'linear', 'extrap');
        theta_end = interp1(series_by_agent(aid).time, theta_vals, window_end_abs, 'linear', 'extrap');
        omega_rad_s(k) = (theta_end - theta_start) / duration_used_s;
    end

    points = struct();
    points.time = time_abs - window_start_abs;
    points.phi1 = mod(a0_1, 256) * (2*pi/256);
    points.phi2 = mod(a0_2, 256) * (2*pi/256);
    points.s1 = signal_1;
    points.s2 = signal_2;

    meta = struct();
    meta.file_path = csv_path;
    meta.window_start_abs = window_start_abs;
    meta.window_end_abs = window_end_abs;
    meta.n_points = numel(points.time);
    meta.omega_rad_s = omega_rad_s;
end

function series_by_agent = load_agent_series_from_csv(csv_path, requested_agents, signal_column)
    T = readtable(csv_path);
    required_cols = {'time_pc_sec_abs', 'a0', signal_column};
    if ~all(ismember(required_cols, T.Properties.VariableNames))
        error('CSV missing required columns. Required: %s', strjoin(required_cols, ', '));
    end
    if ~ismember('agent_id', T.Properties.VariableNames)
        T.agent_id = ones(height(T), 1);
    end

    T.time_pc_sec_abs = double(T.time_pc_sec_abs);
    if ismember('chunk_id', T.Properties.VariableNames)
        t_overflow = 2^32 / 1e6;
        T = correct_large_jump_matlab(T, t_overflow - 5.0, t_overflow);
        T = correct_chunk_start_times_matlab(T, 4000.0, t_overflow);
    end

    all_agents = unique(T.agent_id, 'sorted').';
    requested_agents = requested_agents(:).';
    if ~all(ismember(requested_agents, all_agents))
        error('Requested agents %s are not all present. Available agents: %s', ...
            mat2str(requested_agents), mat2str(all_agents));
    end

    template = struct('time', [], 'a0_corr', [], 'signal', []);
    series_by_agent = repmat(template, 1, max(requested_agents));
    for k = 1:numel(requested_agents)
        aid = requested_agents(k);
        sub = T(T.agent_id == aid, :);
        sub = sortrows(sub, 'time_pc_sec_abs');

        time_vals = double(sub.time_pc_sec_abs(:));
        a0_vals = correct_phase_discontinuity(double(sub.a0(:)));
        signal_vals = double(sub.(signal_column)(:));
        valid = isfinite(time_vals) & isfinite(a0_vals) & isfinite(signal_vals);

        time_vals = time_vals(valid);
        a0_vals = a0_vals(valid);
        signal_vals = signal_vals(valid);
        [time_vals, ia] = unique(time_vals, 'stable');

        series_by_agent(aid).time = time_vals;
        series_by_agent(aid).a0_corr = a0_vals(ia);
        series_by_agent(aid).signal = signal_vals(ia);
    end
end

function corrected_phase = correct_phase_discontinuity(phase_data)
    corrected_phase = phase_data(:);
    for i = 2:numel(corrected_phase)
        diffv = corrected_phase(i) - corrected_phase(i - 1);
        if diffv < -128
            corrected_phase(i:end) = corrected_phase(i:end) + 256;
        elseif diffv > 128
            corrected_phase(i:end) = corrected_phase(i:end) - 256;
        end
    end
end

function T = correct_large_jump_matlab(T, threshold_sec, jump_sec)
    [G, ~] = findgroups(T.agent_id, T.chunk_id);
    for i = 1:max(G)
        idx = find(G == i);
        if isempty(idx)
            continue;
        end

        [~, rel] = sort(T.time_pc_sec_abs(idx));
        idx = idx(rel);
        time_diff = [0; diff(T.time_pc_sec_abs(idx))];
        jump_idx = find(time_diff > threshold_sec);
        for j = 1:numel(jump_idx)
            fix_range = jump_idx(j):numel(idx);
            T.time_pc_sec_abs(idx(fix_range)) = T.time_pc_sec_abs(idx(fix_range)) - jump_sec;
        end
    end
end

function T = correct_chunk_start_times_matlab(T, threshold_sec, jump_sec)
    [G, ~] = findgroups(T.agent_id, T.chunk_id);
    chunk_start = splitapply(@(x) min(x), T.time_pc_sec_abs, G);
    median_start = median(chunk_start, 'omitnan');
    for i = 1:max(G)
        idx = find(G == i);
        if isempty(idx)
            continue;
        end

        start_time = min(T.time_pc_sec_abs(idx));
        if (start_time - median_start) > threshold_sec
            T.time_pc_sec_abs(idx) = T.time_pc_sec_abs(idx) - jump_sec;
        end
    end
end

function x_norm = normalize_by_agent_percentile_span(x, reference_x, tail_percent)
    x = double(x(:));
    reference_x = double(reference_x(:));
    valid = isfinite(x);
    ref_valid = isfinite(reference_x);
    x_norm = nan(size(x));

    if nnz(valid) < 1
        return;
    end
    if nnz(ref_valid) < 5
        x_norm(valid) = x(valid);
        return;
    end

    ref_values = reference_x(ref_valid);
    low_value = prctile(ref_values, tail_percent);
    high_value = prctile(ref_values, 100 - tail_percent);
    center_value = 0.5 * (low_value + high_value);
    span_value = high_value - low_value;

    if ~isfinite(span_value) || span_value <= 0
        x_norm(valid) = x(valid) - center_value;
        return;
    end

    x_norm(valid) = (x(valid) - center_value) / span_value;
end

function x_clipped = clip_values(x, lower_bound, upper_bound)
    x_clipped = min(max(x, lower_bound), upper_bound);
end

function fit_result = fit_double_fourier_scatter(phi1, phi2, z, M, N, label)
    phi1 = mod(phi1(:), 2*pi);
    phi2 = mod(phi2(:), 2*pi);
    z = z(:);
    valid = isfinite(phi1) & isfinite(phi2) & isfinite(z);
    phi1 = phi1(valid);
    phi2 = phi2(valid);
    z = z(valid);
    if isempty(z)
        error('No valid samples remain for %s.', label);
    end

    z_mean = mean(z);
    z_centered = z - z_mean;
    [A, basis_names, basis_groups, basis_m, basis_n, basis_types] = build_double_fourier_design_matrix(phi1, phi2, M, N);
    coeff = A \ z_centered;
    z_hat_centered = A * coeff;
    z_hat = z_hat_centered + z_mean;
    residual = z - z_hat;
    rmse = sqrt(mean(residual .^ 2));
    z_var = sum((z - mean(z)) .^ 2);
    r2 = 1 - sum(residual .^ 2) / max(z_var, eps);

    fit_result = struct();
    fit_result.label = label;
    fit_result.M = M;
    fit_result.N = N;
    fit_result.coeff = coeff;
    fit_result.z_mean = z_mean;
    fit_result.rmse = rmse;
    fit_result.r2 = r2;
    fit_result.phi1 = phi1;
    fit_result.phi2 = phi2;
    fit_result.z = z;
    fit_result.z_hat = z_hat;
    fit_result.residual = residual;
    fit_result.basis_names = basis_names;
    fit_result.basis_groups = basis_groups;
    fit_result.basis_m = basis_m;
    fit_result.basis_n = basis_n;
    fit_result.basis_types = basis_types;
    fit_result.rankA = rank(A);
    fit_result.condA = cond(A);
    fit_result.contribution_table = table((1:numel(coeff)).', basis_names, basis_groups, ...
        basis_m, basis_n, basis_types, coeff, ...
        'VariableNames', {'term_index', 'basis_name', 'basis_group', ...
        'phi1_order', 'phi2_order', 'basis_type', 'coefficient'});
end

function [A, basis_names, basis_groups, basis_m, basis_n, basis_types] = build_double_fourier_design_matrix(phi1, phi2, M, N)
    n_samples = numel(phi1);
    n_basis = 1 + 2*M + 2*N + 4*M*N;
    A = zeros(n_samples, n_basis);
    basis_names = cell(n_basis, 1);
    basis_groups = cell(n_basis, 1);
    basis_m = zeros(n_basis, 1);
    basis_n = zeros(n_basis, 1);
    basis_types = cell(n_basis, 1);

    col = 1;
    A(:, col) = 1;
    basis_names{col} = '1';
    basis_groups{col} = 'constant';
    basis_types{col} = 'constant';
    col = col + 1;

    for m = 1:M
        A(:, col) = cos(m * phi1);
        basis_names{col} = sprintf('cos(%d*phi1)', m);
        basis_groups{col} = 'phi1_only';
        basis_m(col) = m;
        basis_types{col} = 'phi1_cos';
        col = col + 1;

        A(:, col) = sin(m * phi1);
        basis_names{col} = sprintf('sin(%d*phi1)', m);
        basis_groups{col} = 'phi1_only';
        basis_m(col) = m;
        basis_types{col} = 'phi1_sin';
        col = col + 1;
    end

    for n = 1:N
        A(:, col) = cos(n * phi2);
        basis_names{col} = sprintf('cos(%d*phi2)', n);
        basis_groups{col} = 'phi2_only';
        basis_n(col) = n;
        basis_types{col} = 'phi2_cos';
        col = col + 1;

        A(:, col) = sin(n * phi2);
        basis_names{col} = sprintf('sin(%d*phi2)', n);
        basis_groups{col} = 'phi2_only';
        basis_n(col) = n;
        basis_types{col} = 'phi2_sin';
        col = col + 1;
    end

    for m = 1:M
        c1 = cos(m * phi1);
        s1 = sin(m * phi1);
        for n = 1:N
            c2 = cos(n * phi2);
            s2 = sin(n * phi2);

            A(:, col) = c1 .* c2;
            basis_names{col} = sprintf('cos(%d*phi1)cos(%d*phi2)', m, n);
            basis_groups{col} = 'mixed';
            basis_m(col) = m;
            basis_n(col) = n;
            basis_types{col} = 'mixed_cc';
            col = col + 1;

            A(:, col) = c1 .* s2;
            basis_names{col} = sprintf('cos(%d*phi1)sin(%d*phi2)', m, n);
            basis_groups{col} = 'mixed';
            basis_m(col) = m;
            basis_n(col) = n;
            basis_types{col} = 'mixed_cs';
            col = col + 1;

            A(:, col) = s1 .* c2;
            basis_names{col} = sprintf('sin(%d*phi1)cos(%d*phi2)', m, n);
            basis_groups{col} = 'mixed';
            basis_m(col) = m;
            basis_n(col) = n;
            basis_types{col} = 'mixed_sc';
            col = col + 1;

            A(:, col) = s1 .* s2;
            basis_names{col} = sprintf('sin(%d*phi1)sin(%d*phi2)', m, n);
            basis_groups{col} = 'mixed';
            basis_m(col) = m;
            basis_n(col) = n;
            basis_types{col} = 'mixed_ss';
            col = col + 1;
        end
    end
end

function s_values = evaluate_fit_surface(phi1, phi2, fit_result)
    original_size = size(phi1);
    phi1_col = phi1(:);
    phi2_col = phi2(:);
    A = build_double_fourier_design_matrix(phi1_col, phi2_col, fit_result.M, fit_result.N);
    s_values = reshape(A * fit_result.coeff + fit_result.z_mean, original_size);
end

function [gamma1, gamma2_minus_psi] = compute_gamma_for_dynamics(psi, theta, z_theta, fit_s1, fit_s2)
    gamma1 = nan(size(psi));
    gamma2_minus_psi = nan(size(psi));
    for i = 1:numel(psi)
        psi_i = psi(i);
        s1_shifted = evaluate_fit_surface(theta, theta - psi_i, fit_s1);
        s2_shifted_minus = evaluate_fit_surface(theta + psi_i, theta, fit_s2);
        gamma1(i) = trapz(theta, z_theta .* s1_shifted) / (2*pi);
        gamma2_minus_psi(i) = trapz(theta, z_theta .* s2_shifted_minus) / (2*pi);
    end
end

function fig = plot_gamma_components(psi, gamma1, gamma2_minus_psi, psi_dot, sigma, phase_agent_ids)
    fig = figure('Color', 'w', 'Name', 'Gamma components');
    ax1 = subplot(2, 1, 1, 'Parent', fig);
    hold(ax1, 'on');
    plot(ax1, psi, gamma1, 'LineWidth', 1.6, 'DisplayName', '\Gamma_1(\psi)');
    plot(ax1, psi, gamma2_minus_psi, 'LineWidth', 1.6, 'DisplayName', '\Gamma_2(-\psi)');
    grid(ax1, 'on');
    box(ax1, 'on');
    xlim(ax1, [-pi, pi]);
    ylabel(ax1, '\Gamma');
    legend(ax1, 'Location', 'best');

    ax2 = subplot(2, 1, 2, 'Parent', fig);
    hold(ax2, 'on');
    plot(ax2, psi, psi_dot, 'LineWidth', 1.6, 'Color', [0.8500, 0.3250, 0.0980]);
    plot(ax2, [min(psi), max(psi)], [0, 0], ':', 'LineWidth', 1.0, 'Color', [0.45, 0.45, 0.45]);
    grid(ax2, 'on');
    box(ax2, 'on');
    xlim(ax2, [-pi, pi]);
    xlabel(ax2, '\psi');
    ylabel(ax2, 'd\psi/dt');
    title(ax2, sprintf('agents %d,%d | sigma = %.6g | Delta omega = 0', ...
        phase_agent_ids(1), phase_agent_ids(2), sigma), 'Interpreter', 'none');
end

function fig = plot_fitted_surfaces(fit_s1, fit_s2, phase_agent_ids)
    phi_values = linspace(0, 2*pi, 121);
    [Phi1, Phi2] = meshgrid(phi_values, phi_values);
    S1 = evaluate_fit_surface(Phi1, Phi2, fit_s1);
    S2 = evaluate_fit_surface(Phi1, Phi2, fit_s2);

    fig = figure('Color', 'w', 'Name', 'fitted s surfaces');
    ax1 = subplot(1, 2, 1, 'Parent', fig);
    imagesc(ax1, phi_values, phi_values, S1);
    set(ax1, 'YDir', 'normal');
    axis(ax1, 'square');
    colorbar(ax1);
    xlabel(ax1, '\phi_1');
    ylabel(ax1, '\phi_2');
    title(ax1, sprintf('s_1, agent %d', phase_agent_ids(1)));

    ax2 = subplot(1, 2, 2, 'Parent', fig);
    imagesc(ax2, phi_values, phi_values, S2);
    set(ax2, 'YDir', 'normal');
    axis(ax2, 'square');
    colorbar(ax2);
    xlabel(ax2, '\phi_1');
    ylabel(ax2, '\phi_2');
    title(ax2, sprintf('s_2, agent %d', phase_agent_ids(2)));
end

function export = save_outputs(out, opts)
    output_dir = opts.output_dir;
    if isempty(output_dir)
        output_dir = fullfile('EstimateL', 'Round', 'phase_dynamics_exports');
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    csv_path = fullfile(output_dir, 'phase_dynamics.csv');

    dynamics_table = table(out.psi(:), out.gamma1(:), out.gamma2_minus_psi(:), out.psi_dot(:), ...
        'VariableNames', {'psi', 'gamma1', 'gamma2_minus_psi', 'psi_dot'});
    writetable(dynamics_table, csv_path);

    export = struct('output_dir', output_dir, 'csv_path', csv_path);
end
