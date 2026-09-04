function varargout = plot_relative_phase(dirpath, csv_rank_from_latest, n_seconds_to_cut, plot_duration, apply_filter, filter_window_size, do_save_figure, n_sync, m_sync, sample_window, overlay_mode, plot_order_parameter)
% Phase-relationship time evolution from one selected CSV in a directory
%
% Usage:
%   plot_relative_phase()
%   plot_relative_phase(dirpath, csv_rank_from_latest)
%   plot_relative_phase(dirpath, csv_rank_from_latest, n_seconds_to_cut, plot_duration, [], [], [], [], [], [], overlay_mode, plot_order_parameter)
%
% Defaults:
%   dirpath = 'merged_chunks_organized/2026-07-13'
%   csv_rank_from_latest = 1
%   n_seconds_to_cut = 10
%   plot_duration = 1200
%   apply_filter = true
%   filter_window_size = 1
%   do_save_figure = false
%   sample_window = [50, 60]
%   overlay_mode = false
%   plot_order_parameter = true

    % --- Display-only agent ID offset for publication plots ---
    % Set to 0 to show raw ids, or -6 to display 7->1, 8->2, ..., 10->4.
    agent_display_offset = -0;

    % --- SVD Mode weights & Agent ID mapping settings ---
    % File or directory path containing agent SVD contributions (sender_v_mode1, sender_v_mode2, ...)
    svd_weights_file = fullfile('EstimateL', 'SStick', 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd', 'agent_svd_contributions.csv');

    % Mapping from real robot agent_id (in experiment CSV) to agent_id used in SVD mode calculation.
    % Format options:
    %   - 2-column matrix: [real_id1, mode_id1; real_id2, mode_id2; ...] (e.g., [1, 7; 2, 8; 3, 9; 4, 10])
    %   - containers.Map:  containers.Map([7, 8, 9, 10], [7, 8, 9, 10])
    %   - empty []:        direct 1-to-1 matching (real_id == mode_id)
    agent_id_map = [9,8; 8,10; 11,7; 12,9]; % Example mapping for 4 agents (7->7, 8->8, 9->9, 10->10)

    if nargin < 1 || isempty(dirpath)
        dirpath = fullfile('merged_chunks_organized','2026-09-04');
        %dirpath = fullfile('EstimateF','Spring5/250');
        %dirpath = fullfile('EstimateQ','VerifyZopt/Spring3/w1/250');
    end
    if nargin < 2 || isempty(csv_rank_from_latest)
        csv_rank_from_latest = 1;
    end
    if nargin < 3 || isempty(n_seconds_to_cut)
        n_seconds_to_cut = 0.1;
    end
    if nargin < 4 || isempty(plot_duration)
        plot_duration = 105.1;
    end
    if nargin < 5 || isempty(apply_filter)
        apply_filter = true;
    end
    if nargin < 6 || isempty(filter_window_size)
        filter_window_size = 1;
    end
    if nargin < 7 || isempty(do_save_figure)
        do_save_figure = false;
    end
    if nargin < 8 || isempty(n_sync)
        n_sync = 1; % default: check 2:1 synchronization
    end
    if nargin < 9 || isempty(m_sync)
        m_sync = 1;
    end
    if nargin < 10 || isempty(sample_window)
        sample_window = [50, 60];
    end
    if nargin < 11 || isempty(overlay_mode)
        overlay_mode = false;
    end
    if nargin < 12 || isempty(plot_order_parameter)
        plot_order_parameter = true;
    end

    if ~isnumeric(csv_rank_from_latest) || ~isscalar(csv_rank_from_latest) || ...
            ~isfinite(csv_rank_from_latest) || csv_rank_from_latest < 1 || ...
            fix(csv_rank_from_latest) ~= csv_rank_from_latest
        error('csv_rank_from_latest must be a positive integer.');
    end

    if numel(sample_window) ~= 2
        error('sample_window must contain exactly two elements [t_start, t_end].');
    end
    sample_window = sort(sample_window(:)).';
    if sample_window(2) <= sample_window(1)
        error('sample_window end must be greater than start.');
    end

    if ~isfolder(dirpath)
        error('Directory not found: %s', dirpath);
    end

    csvs = dir(fullfile(dirpath, '*.csv'));
    if isempty(csvs)
        error('No CSV files found in %s', dirpath);
    end

    csv_table = struct2table(csvs);
    csv_table = sortrows(csv_table, {'datenum','name'}, {'descend','descend'});
    csvs = table2struct(csv_table);

    if csv_rank_from_latest > numel(csvs)
        error('Requested CSV #%d from latest, but only %d CSV file(s) found in %s.', ...
            csv_rank_from_latest, numel(csvs), dirpath);
    end

    selected_csv = csvs(csv_rank_from_latest);
    file_list = {fullfile(dirpath, selected_csv.name)};
    fprintf('[INFO] Selected CSV #%d from latest: %s (modified %s).\n', ...
        csv_rank_from_latest, selected_csv.name, selected_csv.date);

    % --- overflow / chunk jump correction constants (same as original) ---
    T_OVERFLOW = 2^32 / 1e6; % 約4294.967296秒
    T_TOL = 5.0;             % 許容誤差（秒）
    threshold_sec = T_OVERFLOW - T_TOL;
    jump_sec = T_OVERFLOW;
    CLUSTER_VAR_THRESHOLD = 0.35;

    % Read all files and collect agent sets
    file_tables = cell(size(file_list));
    agent_sets = cell(size(file_list));

    for i = 1:numel(file_list)
        try
            T = readtable(file_list{i});
        catch ME
            warning('Failed to read %s: %s', file_list{i}, ME.message);
            continue;
        end
        % Keep expected columns if present
        if ~all(ismember({'time_pc_sec_abs','a0','agent_id'}, T.Properties.VariableNames))
            warning('Skipping %s: missing required columns', file_list{i});
            continue;
        end
        % Ensure chunk_id exists (if not, create a dummy single chunk)
        if ~ismember('chunk_id', T.Properties.VariableNames)
            T.chunk_id = ones(height(T),1);
        end
        T = T(T.agent_id ~= 99, :);
        if isempty(T)
            warning('Skipping %s: only agent_id == 99 found after filtering.', file_list{i});
            continue;
        end
        % Filter out spurious agents (with very few samples compared to the main agent(s))
        all_agents = unique(T.agent_id);
        agent_counts = zeros(size(all_agents));
        for k = 1:length(all_agents)
            agent_counts(k) = sum(T.agent_id == all_agents(k));
        end
        if ~isempty(agent_counts)
            max_count = max(agent_counts);
            if max_count >= 100
                min_count_thresh = max(50, 0.05 * max_count);
            else
                min_count_thresh = max(5, 0.05 * max_count);
            end
            valid_agents = all_agents(agent_counts >= min_count_thresh);
            T = T(ismember(T.agent_id, valid_agents), :);
        end
        if isempty(T)
            warning('Skipping %s: no valid agents after filtering spurious ones.', file_list{i});
            continue;
        end
        file_tables{i} = T(:,{'time_pc_sec_abs','a0','agent_id','chunk_id'});
        % Apply overflow / chunk-start corrections per-file (same logic as original)
        file_tables{i} = correct_large_jump_matlab(file_tables{i}, threshold_sec, jump_sec);
        file_tables{i} = correct_chunk_start_times_matlab(file_tables{i}, 4000.0, T_OVERFLOW);
        FT = file_tables{i};
        agent_sets{i} = unique(FT.agent_id);
    end

    valid_idx = ~cellfun(@isempty, file_tables);
    file_tables = file_tables(valid_idx);
    file_list = file_list(valid_idx);
    agent_sets = agent_sets(valid_idx);
    if isempty(file_tables)
        error('No valid data files to plot.');
    end
    
    % Skip CSV files with fewer than 2 agents
    agent_counts = cellfun(@numel, agent_sets);
    valid_agent_idx = agent_counts >= 2;
    
    % Display information about skipped files
    skipped_file_idx = find(~valid_agent_idx);
    for k = 1:numel(skipped_file_idx)
        i = skipped_file_idx(k);
        [~, name] = fileparts(file_list{i});
        fprintf('[INFO] Skipping %s: only %d agent(s) found (need at least 2).\n', ...
            name, agent_counts(i));
    end
    
    file_tables = file_tables(valid_agent_idx);
    file_list = file_list(valid_agent_idx);
    agent_sets = agent_sets(valid_agent_idx);
    if isempty(file_tables)
        warning('No CSV files with 2 or more agents found. Exiting.');
        return;
    end

    % Determine agents common to all files; if none, fall back to per-file reference
    common_agents = agent_sets{1};
    for i = 2:numel(agent_sets)
        common_agents = intersect(common_agents, agent_sets{i});
    end

    use_dynamic_reference = isempty(common_agents);
    if use_dynamic_reference
        common_agents = unique(cat(1, agent_sets{:}));
        fprintf('[INFO] No common agents across files; using per-file reference agents.\n');
    end

    base_agent_per_file = zeros(numel(agent_sets),1);
    if use_dynamic_reference
        for i = 1:numel(agent_sets)
            base_agent_per_file(i) = min(agent_sets{i});
        end
        ref_agent_for_label = mode(base_agent_per_file);
        agents_to_plot = common_agents;
    else
        base_agent = min(common_agents);
        base_agent_per_file(:) = base_agent;
        ref_agent_for_label = base_agent;
        agents_to_plot = common_agents;
    end

    if isempty(agents_to_plot)
        error('Not enough agents to compute relative phases (need at least 2).');
    end
    agents_for_cluster = setdiff(agents_to_plot, ref_agent_for_label, 'stable');
    if isempty(agents_for_cluster)
        agents_for_cluster = agents_to_plot;
    end

    % Prepare per-file phase series using original pipeline
    allow_missing_agents = 1;
    phase_series_by_file = cell(numel(file_tables), 1);
    for f = 1:numel(file_tables)
        phase_series_by_file{f} = compute_phase_series_for_file( ...
            file_tables{f}, base_agent_per_file(f), n_seconds_to_cut, plot_duration, ...
            allow_missing_agents, apply_filter, filter_window_size, n_sync, m_sync);
    end

    % Common plotting parameters
    line_style = '-';
    max_plot_time = min(plot_duration - n_seconds_to_cut, 120);

    y_label_str = sprintf('$$\\phi_j - \\phi_k\\quad(k=%d)$$', displayed_agent_id(ref_agent_for_label, agent_display_offset));

    if overlay_mode
        % ===== OVERLAY MODE: selected CSV in one figure =====
        figure('Visible','on');
        ax = axes('Parent', gcf);
        hold(ax,'on');
        colors = lines(numel(file_list));
        line_handles = gobjects(numel(file_list),1);
        legend_labels = cell(numel(file_list),1);
        for f = 1:numel(file_list)
            [~, legend_labels{f}] = fileparts(file_list{f});
        end

        ylabel(ax, y_label_str,'Interpreter','latex');
        ylim(ax, [-pi, pi]);
        yticks(ax, [-pi,0,pi]);
        yticklabels(ax, {'-\pi','0','\pi'});
        set(ax,'TickLabelInterpreter','latex');
        yl = yline(ax, 0, 'Color', [0.3 0.3 0.3], 'LineStyle', '-', 'LineWidth', 0.8);
        yl.Annotation.LegendInformation.IconDisplayStyle = 'off';

        for p = 1:numel(agents_to_plot)
            ag = agents_to_plot(p);
            for f = 1:numel(file_list)
                series_struct = phase_series_by_file{f};
                series_entry = get_agent_series_entry(series_struct, ag);
                if isempty(series_entry)
                    continue;
                end
                if isempty(series_entry.time) || isempty(series_entry.phase)
                    continue;
                end
                h = plot(ax, series_entry.time, series_entry.phase, ...
                    'Color', colors(f,:), 'LineWidth', 0.8, 'LineStyle', line_style);
                if ~isgraphics(line_handles(f))
                    line_handles(f) = h;
                end
            end
        end

        xlim(ax, [0, max_plot_time]);
        xlabel(ax, 'Time (s)','Interpreter','latex');
        grid(ax, 'on');
        hold(ax,'off');

        valid_handles = isgraphics(line_handles);
        if any(valid_handles)
            %legend(ax, line_handles(valid_handles), legend_labels(valid_handles), ...
            %    'Location','eastoutside','Interpreter','latex');
        end

        if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
            tuneFigure();
        end

    else
        % ===== DEFAULT MODE: selected CSV in one figure =====
        for f = 1:numel(file_list)
            figure('Visible','on');
            ax = axes('Parent', gcf);
            hold(ax,'on');
            [~, file_label] = fileparts(file_list{f});
            set(gcf, 'Name', sprintf('Chunk: %s', file_label));
            
            colors = lines(numel(agents_to_plot));

            ylabel(ax, y_label_str,'Interpreter','latex');
            ylim(ax, [-pi, pi]);
            yticks(ax, [-pi,0,pi]);
            yticklabels(ax, {'-\pi','0','\pi'});
            set(ax,'TickLabelInterpreter','latex');
            yl = yline(ax, 0, 'Color', [0.3 0.3 0.3], 'LineStyle', '-', 'LineWidth', 0.8);
            yl.Annotation.LegendInformation.IconDisplayStyle = 'off';

            series_struct = phase_series_by_file{f};
            agent_legend = {};
            line_handles_sep = [];
            
            for p = 1:numel(agents_to_plot)
                ag = agents_to_plot(p);
                series_entry = get_agent_series_entry(series_struct, ag);
                if isempty(series_entry)
                    continue;
                end
                if isempty(series_entry.time) || isempty(series_entry.phase)
                    continue;
                end
                h = plot(ax, series_entry.time, series_entry.phase, ...
                    'Color', colors(p,:), 'LineWidth', 0.8, 'LineStyle', line_style);
                line_handles_sep = [line_handles_sep; h];
                agent_legend = [agent_legend; {sprintf('Agent %d', displayed_agent_id(ag, agent_display_offset))}];
            end

            xlim(ax, [0, max_plot_time]);
            xlabel(ax, 'Time (s)','Interpreter','latex');
            grid(ax, 'on');
            hold(ax,'off');
            
            if ~isempty(line_handles_sep)
                legend(ax, line_handles_sep, agent_legend, ...
                    'Location','eastoutside','Interpreter','latex');
            end

            if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
                tuneFigure();
            end
        end
    end

    cluster_info = cluster_phase_window_means(phase_series_by_file, agents_for_cluster, sample_window, CLUSTER_VAR_THRESHOLD);

    if do_save_figure && (exist('saveFigure', 'file') == 2 || exist('saveFigure', 'builtin'))
        saveFigure();
    end

    if plot_order_parameter
        plot_weighted_order_parameter_time_series(phase_series_by_file, file_list, agents_to_plot, ...
            max_plot_time, overlay_mode, line_style, svd_weights_file, agent_id_map, agent_display_offset);
        plot_agent_weighted_input_amplitude_time_series(phase_series_by_file, file_list, agents_to_plot, ...
            max_plot_time, overlay_mode, line_style, svd_weights_file, agent_id_map, agent_display_offset);
    end

    if nargout >= 1
        varargout{1} = cluster_info;
    end
    if nargout >= 2
        varargout{2} = phase_series_by_file;
    end
end

function cluster_info = cluster_phase_window_means(phase_series_by_file, agents_to_plot, sample_window, cluster_var_threshold)
    window_start = sample_window(1);
    window_end = sample_window(2);
    all_samples = [];

    for f = 1:numel(phase_series_by_file)
        series_struct = phase_series_by_file{f};
        if isempty(series_struct)
            continue;
        end
        for ag = agents_to_plot(:).'
            series_entry = get_agent_series_entry(series_struct, ag);
            if isempty(series_entry)
                continue;
            end
            times = series_entry.time;
            phases = series_entry.phase;
            if isempty(times) || isempty(phases)
                continue;
            end
            mask = (times >= window_start) & (times <= window_end);
            if ~any(mask)
                continue;
            end
            window_phases = phases(mask);
            window_phases = window_phases(~isnan(window_phases));
            if isempty(window_phases)
                continue;
            end
            agent_mean = atan2(mean(sin(window_phases)), mean(cos(window_phases)));
            all_samples(end+1) = agent_mean; %#ok<AGROW>
        end
    end

    cluster_info = struct('cluster_id', {}, 'n_clusters', {}, 'mean', {}, 'stderr', {}, 'count', {}, 'circ_var', {}, 'nsamples_total', {});

    if isempty(all_samples)
        fprintf('[INFO] No valid phase samples found between %.2f and %.2f seconds.\n', window_start, window_end);
        return;
    end

    Nsamples = numel(all_samples);
    R = abs(mean(exp(1i * all_samples(:))));
    circ_var = 1 - R;

    if circ_var > cluster_var_threshold && Nsamples >= 6
        k = 2;
        pts = [cos(all_samples(:)), sin(all_samples(:))];
        try
            opts = statset('MaxIter',500);
            [idx, ~] = kmeans(pts, k, 'Replicates', 5, 'Options', opts);
        catch
            idx = ones(size(all_samples(:)));
            k = 1;
        end
    else
        k = 1;
        idx = ones(size(all_samples(:)));
    end

    fprintf('[INFO] Phase clusters over %.2f-%.2f s (Nsamples=%d, circVar=%.3f):\n', ...
        window_start, window_end, Nsamples, circ_var);

    for c = 1:k
        mask = idx == c;
        ths = all_samples(mask);
        if isempty(ths)
            continue;
        end
        mmean = atan2(mean(sin(ths)), mean(cos(ths)));
        Rcl = abs(mean(exp(1i * ths)));
        circ_std = sqrt(max(0, -2 * log(max(Rcl, eps))));
        stderr = circ_std / sqrt(max(1, numel(ths)));
        fprintf('  Cluster %d/%d: mean = %.3f rad (%.2f deg), stderr = %.3f (n=%d)\n', ...
            c, k, mmean, mmean * 180/pi, stderr, numel(ths));
        cluster_info(end+1) = struct( ...
            'cluster_id', c, ...
            'n_clusters', k, ...
            'mean', mmean, ...
            'stderr', stderr, ...
            'count', numel(ths), ...
            'circ_var', circ_var, ...
            'nsamples_total', Nsamples); %#ok<AGROW>
    end
end

function series_struct = compute_phase_series_for_file(df_all, base_agent_id, n_seconds_to_cut, plot_duration, allow_missing_agents, apply_filter, filter_window_size, n_sync, m_sync)
    if nargin < 5 || isempty(allow_missing_agents)
        allow_missing_agents = 1;
    end
    if nargin < 6 || isempty(apply_filter)
        apply_filter = true;
    end
    if nargin < 7 || isempty(filter_window_size)
        filter_window_size = 1;
    end
    if nargin < 9 || isempty(n_sync)
        n_sync = 2;
    end
    if nargin < 10 || isempty(m_sync)
        m_sync = 1;
    end

    if isempty(df_all)
        series_struct = empty_phase_series_struct();
        return;
    end

    df_main = df_all(df_all.agent_id ~= 99, :);
    if isempty(df_main)
        series_struct = empty_phase_series_struct();
        return;
    end

    min_time = min(df_main.time_pc_sec_abs);
    max_time = max(df_main.time_pc_sec_abs);
    agents = unique(df_main.agent_id, 'sorted').';

    agent_ranges = zeros(length(agents), 2);
    for i = 1:length(agents)
        agent_id = agents(i);
        sub = df_main(df_main.agent_id == agent_id, :);
        agent_ranges(i,1) = min(sub.time_pc_sec_abs);
        agent_ranges(i,2) = max(sub.time_pc_sec_abs);
    end

    if isempty(base_agent_id) || ~ismember(base_agent_id, agents)
        base_agent_id = min(agents);
    end

    start_time_abs = min_time + n_seconds_to_cut;
    max_allowed_time = plot_duration - n_seconds_to_cut;
    candidate_end_abs = min(max_time, start_time_abs + max_allowed_time);
    if candidate_end_abs < start_time_abs
        series_struct = empty_phase_series_struct();
        return;
    end

    new_time_series = (start_time_abs:0.01:candidate_end_abs) - start_time_abs;
    if isempty(new_time_series)
        series_struct = empty_phase_series_struct();
        return;
    end

    valid_counts = zeros(size(new_time_series));
    for t_idx = 1:length(new_time_series)
        t_abs = new_time_series(t_idx) + start_time_abs;
        valid_counts(t_idx) = sum(agent_ranges(:,1) <= t_abs & agent_ranges(:,2) >= t_abs);
    end

    min_valid = length(agents) - allow_missing_agents;
    valid_idx = find(valid_counts >= min_valid);
    if isempty(valid_idx)
        series_struct = empty_phase_series_struct();
        return;
    end

    new_time_series = new_time_series(valid_idx);
    if isempty(new_time_series)
        series_struct = empty_phase_series_struct();
        return;
    end

    interpolated_data = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for i = 1:length(agents)
        agent_id = agents(i);
        sub = df_main(df_main.agent_id == agent_id, :);
        sub = sortrows(sub, 'time_pc_sec_abs');
        sub.a0 = correct_phase_discontinuity(sub.a0);
        [~, ia] = unique(sub.time_pc_sec_abs);
        sub = sub(ia, :);
        t_min_agent = min(sub.time_pc_sec_abs) - start_time_abs;
        t_max_agent = max(sub.time_pc_sec_abs) - start_time_abs;
        valid_mask = (new_time_series >= t_min_agent) & (new_time_series <= t_max_agent);
        interp_a0 = nan(size(new_time_series));
        interp_a0(valid_mask) = interp1(sub.time_pc_sec_abs - start_time_abs, sub.a0, new_time_series(valid_mask), 'linear', 'extrap');
        if apply_filter
            interp_a0 = movmean(interp_a0, filter_window_size, 'omitnan');
        end
        interpolated_data(agent_id) = struct('a0', interp_a0);
    end

    base_agent_a0 = interpolated_data(base_agent_id).a0;

    series_struct = repmat(empty_phase_series_entry(), 1, numel(agents));

    for i = 1:length(agents)
        agent_id = agents(i);
        absolute_phase_unwrapped = interpolated_data(agent_id).a0 * (2*pi/256);
        absolute_phase = mod(interpolated_data(agent_id).a0, 256) * (2*pi/256);
        absolute_phase = break_wrapped_phase_jumps(absolute_phase);
        phase_raw = n_sync * interpolated_data(agent_id).a0 - m_sync * base_agent_a0;
        phase_diff = mod(phase_raw + 128, 256) - 128;
        phase_diff = phase_diff * (2*pi/256);
        phase_diff_with_nan = phase_diff;
        for j = 2:length(phase_diff_with_nan)
            if isnan(phase_diff_with_nan(j)) || isnan(phase_diff_with_nan(j-1))
                continue;
            end
            if abs(phase_diff_with_nan(j) - phase_diff_with_nan(j-1)) > pi
                phase_diff_with_nan(j) = NaN;
            end
        end
        series_struct(i).agent_id = agent_id;
        series_struct(i).time = new_time_series;
        series_struct(i).absolute_phase = absolute_phase;
        series_struct(i).absolute_phase_unwrapped = absolute_phase_unwrapped;
        if agent_id == base_agent_id
            series_struct(i).phase = zeros(size(new_time_series));
        else
            series_struct(i).phase = phase_diff_with_nan;
        end
    end
end

function plot_weighted_order_parameter_time_series(phase_series_by_file, file_list, agents_to_plot, ...
    max_plot_time, overlay_mode, line_style, svd_weights_file, agent_id_map, agent_display_offset)

    if ~exist(svd_weights_file, 'file')
        if isfolder(svd_weights_file)
            candidate = fullfile(svd_weights_file, 'agent_svd_contributions.csv');
            if exist(candidate, 'file')
                svd_weights_file = candidate;
            else
                warning('SVD weights CSV not found in folder: %s', svd_weights_file);
                return;
            end
        else
            warning('SVD weights file not found: %s', svd_weights_file);
            return;
        end
    end

    try
        weights_table = readtable(svd_weights_file);
    catch ME
        warning('Failed to read SVD weights file %s: %s', svd_weights_file, ME.message);
        return;
    end

    var_names = weights_table.Properties.VariableNames;
    mode_cols = var_names(startsWith(var_names, 'sender_v_mode'));
    if isempty(mode_cols)
        warning('No sender_v_mode columns found in %s', svd_weights_file);
        return;
    end

    n_modes = numel(mode_cols);
    fprintf('[INFO] Loaded SVD weights from %s (%d sender modes).\n', svd_weights_file, n_modes);

    N_agents = numel(agents_to_plot);
    v_weights = zeros(N_agents, n_modes);

    for a_idx = 1:N_agents
        real_ag = agents_to_plot(a_idx);
        mode_ag = get_mapped_agent_id(real_ag, agent_id_map);

        row_mask = (weights_table.agent_id == mode_ag);
        if ~any(row_mask)
            warning('Agent ID %d (mapped from real ID %d) not found in SVD weights table.', mode_ag, real_ag);
            continue;
        end

        for m = 1:n_modes
            v_weights(a_idx, m) = weights_table.(mode_cols{m})(row_mask);
        end
    end

    y_label_str = '$$|Z_l(t)| = \left|\sum_j v_{jl} e^{i \phi_j(t)}\right|$$';

    if overlay_mode
        figure('Visible','on');
        ax = axes('Parent', gcf);
        hold(ax,'on');
        set(gcf, 'Name', 'Weighted Order Parameter |Z_l(t)| (Overlay)');

        colors = lines(n_modes);
        mode_legend = cell(n_modes, 1);
        line_handles = gobjects(n_modes, 1);

        for f = 1:numel(file_list)
            series_struct = phase_series_by_file{f};
            if isempty(series_struct)
                continue;
            end

            [t_common, Z_abs] = compute_order_parameters_for_file(series_struct, agents_to_plot, v_weights, n_modes);
            if isempty(t_common)
                continue;
            end

            for m = 1:n_modes
                h = plot(ax, t_common, Z_abs(:, m), ...
                    'Color', colors(m,:), 'LineWidth', 1.2, 'LineStyle', line_style);
                if ~isgraphics(line_handles(m))
                    line_handles(m) = h;
                    mode_legend{m} = sprintf('Sender Mode %d', m);
                end
            end
        end

        format_order_parameter_axis(ax, y_label_str, max_plot_time);
        hold(ax,'off');

        valid_h = isgraphics(line_handles);
        if any(valid_h)
            legend(ax, line_handles(valid_h), mode_legend(valid_h), ...
                'Location','eastoutside','Interpreter','latex');
        end

        if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
            tuneFigure();
        end

    else
        for f = 1:numel(file_list)
            series_struct = phase_series_by_file{f};
            if isempty(series_struct)
                continue;
            end

            [t_common, Z_abs] = compute_order_parameters_for_file(series_struct, agents_to_plot, v_weights, n_modes);
            if isempty(t_common)
                continue;
            end

            figure('Visible','on');
            ax = axes('Parent', gcf);
            hold(ax,'on');
            [~, file_label] = fileparts(file_list{f});
            set(gcf, 'Name', sprintf('Weighted Order Parameter: %s', file_label));

            colors = lines(n_modes);
            line_handles = gobjects(n_modes, 1);
            mode_legend = cell(n_modes, 1);

            for m = 1:n_modes
                line_handles(m) = plot(ax, t_common, Z_abs(:, m), ...
                    'Color', colors(m,:), 'LineWidth', 1.2, 'LineStyle', line_style);
                mode_legend{m} = sprintf('Sender Mode %d', m);
            end

            format_order_parameter_axis(ax, y_label_str, max_plot_time);
            hold(ax,'off');

            legend(ax, line_handles, mode_legend, ...
                'Location','eastoutside','Interpreter','latex');

            if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
                tuneFigure();
            end
        end
    end
end

function [t_common, Z_abs] = compute_order_parameters_for_file(series_struct, agents_to_plot, v_weights, n_modes)
    t_common = [];
    Z_abs = [];

    N_agents = numel(agents_to_plot);
    phase_matrix = [];

    for a_idx = 1:N_agents
        ag = agents_to_plot(a_idx);
        series_entry = get_agent_series_entry(series_struct, ag);
        if isempty(series_entry) || isempty(series_entry.time)
            return;
        end

        if isfield(series_entry, 'absolute_phase_unwrapped') && ~isempty(series_entry.absolute_phase_unwrapped)
            ag_phase = series_entry.absolute_phase_unwrapped(:);
        elseif isfield(series_entry, 'absolute_phase') && ~isempty(series_entry.absolute_phase)
            ag_phase = series_entry.absolute_phase(:);
        else
            return;
        end

        if isempty(t_common)
            t_common = series_entry.time(:);
            phase_matrix = zeros(numel(t_common), N_agents);
        end

        phase_matrix(:, a_idx) = ag_phase;
    end

    if isempty(t_common)
        return;
    end

    Z_complex = exp(1i * phase_matrix) * v_weights;
    Z_abs = abs(Z_complex);
end

function format_order_parameter_axis(ax, y_label_str, max_plot_time)
    ylabel(ax, y_label_str,'Interpreter','latex');
    set(ax,'TickLabelInterpreter','latex');
    xlim(ax, [0, max_plot_time]);
    xlabel(ax, 'Time (s)','Interpreter','latex');
    grid(ax, 'on');
end

function mode_ag = get_mapped_agent_id(real_ag, agent_id_map)
    mode_ag = real_ag;
    if isempty(agent_id_map)
        return;
    end

    if isnumeric(agent_id_map) && size(agent_id_map, 2) == 2
        idx = find(agent_id_map(:, 1) == real_ag, 1);
        if ~isempty(idx)
            mode_ag = agent_id_map(idx, 2);
        end
    elseif isa(agent_id_map, 'containers.Map')
        if isKey(agent_id_map, real_ag)
            mode_ag = agent_id_map(real_ag);
        end
    elseif isa(agent_id_map, 'function_handle')
        mode_ag = agent_id_map(real_ag);
    elseif isstruct(agent_id_map) && isfield(agent_id_map, 'real_id') && isfield(agent_id_map, 'mode_id')
        idx = find([agent_id_map.real_id] == real_ag, 1);
        if ~isempty(idx)
            mode_ag = agent_id_map(idx).mode_id;
        end
    end
end

function display_id = displayed_agent_id(agent_id, offset)
    display_id = agent_id + offset;
end

function phase_for_plot = break_wrapped_phase_jumps(phase_for_plot)
    for j = 2:length(phase_for_plot)
        if isnan(phase_for_plot(j)) || isnan(phase_for_plot(j-1))
            continue;
        end
        if abs(phase_for_plot(j) - phase_for_plot(j-1)) > pi
            phase_for_plot(j) = NaN;
        end
    end
end

function series_struct = empty_phase_series_struct()
    series_struct = struct('agent_id', {}, 'time', {}, 'phase', {}, ...
        'absolute_phase', {}, 'absolute_phase_unwrapped', {});
end

function series_entry = empty_phase_series_entry()
    series_entry = struct('agent_id', [], 'time', [], 'phase', [], ...
        'absolute_phase', [], 'absolute_phase_unwrapped', []);
end

function corrected_phase = correct_phase_discontinuity(phase_data)
    % same logic as in plot_relative_phase_matlab_2modules.m
    corrected_phase = phase_data;
    for i = 2:length(corrected_phase)
        diffv = corrected_phase(i) - corrected_phase(i - 1);
        if diffv < -128
            corrected_phase(i:end) = corrected_phase(i:end) + 256;
        elseif diffv > 128
            corrected_phase(i:end) = corrected_phase(i:end) - 256;
        end
    end
end

function df_all = correct_large_jump_matlab(df_all, threshold_sec, jump_sec)
    % グループ化（agent_id, chunk_id 単位）
    [G, ~] = findgroups(df_all.agent_id, df_all.chunk_id);
    fprintf('[INFO] Found %d unique chunks.\n', max(G));

    % 該当ブロックを修正
    for i = 1:max(G)
        idx = find(G == i);
        if isempty(idx)
            continue;
        end

        % 時系列を並び替え
        [~, sorted_idx_rel] = sort(df_all.time_pc_sec_abs(idx));
        idx = idx(sorted_idx_rel);
        
        time_series = df_all.time_pc_sec_abs(idx);
        time_diff = [0; diff(time_series)];

        jump_idx = find(time_diff > threshold_sec);
        for j = 1:length(jump_idx)
            fix_range = jump_idx(j):length(time_series);
            df_all.time_pc_sec_abs(idx(fix_range)) = df_all.time_pc_sec_abs(idx(fix_range)) - jump_sec;
            fprintf('[FIX] Corrected overflow at index %d, subtracted %.6f sec.\n', idx(jump_idx(j)), jump_sec);
        end
    end

end

function df_all = correct_chunk_start_times_matlab(df_all, threshold_sec, jump_sec)

    [G, agent_keys, chunk_keys] = findgroups(df_all.agent_id, df_all.chunk_id);
    chunk_start = splitapply(@min, df_all.time_pc_sec_abs, G);
    median_start = median(chunk_start);

    for i = 1:max(G)
        idx = find(G == i);
        if isempty(idx)
            continue;  % 空グループスキップ
        end
        start_time = min(df_all.time_pc_sec_abs(idx));
        if start_time - median_start > threshold_sec
            df_all.time_pc_sec_abs(idx) = df_all.time_pc_sec_abs(idx) - jump_sec;
            aid = agent_keys(i);
            cid = chunk_keys(i);
            fprintf('[FIX] Corrected chunk time for agent %d, chunk %d: %.3f → %.3f\n', ...
                aid, cid, start_time, start_time - jump_sec);
        end
    end
end

function series_entry = get_agent_series_entry(series_struct, agent_id)
    series_entry = [];
    if isempty(series_struct)
        return;
    end
    agent_ids = [series_struct.agent_id];
    match_idx = find(agent_ids == agent_id, 1, 'first');
    if isempty(match_idx)
        return;
    end
    series_entry = series_struct(match_idx);
end

function plot_agent_weighted_input_amplitude_time_series(phase_series_by_file, file_list, agents_to_plot, ...
    max_plot_time, overlay_mode, line_style, svd_weights_file, agent_id_map, agent_display_offset)

    if ~exist(svd_weights_file, 'file')
        if isfolder(svd_weights_file)
            candidate = fullfile(svd_weights_file, 'agent_svd_contributions.csv');
            if exist(candidate, 'file')
                svd_weights_file = candidate;
            else
                warning('SVD weights CSV not found in folder: %s', svd_weights_file);
                return;
            end
        else
            warning('SVD weights file not found: %s', svd_weights_file);
            return;
        end
    end

    [svd_dir, ~, ~] = fileparts(svd_weights_file);
    summary_file = fullfile(svd_dir, 'network_svd_modes_summary.csv');
    if ~exist(summary_file, 'file')
        warning('SVD summary CSV (network_svd_modes_summary.csv) not found in directory: %s', svd_dir);
        return;
    end

    try
        weights_table = readtable(svd_weights_file);
        summary_table = readtable(summary_file);
    catch ME
        warning('Failed to read SVD files: %s', ME.message);
        return;
    end

    % Extract mode columns for receiver (u) and sender (v)
    var_names = weights_table.Properties.VariableNames;
    u_col_info = struct('col_name', {}, 'mode_num', {});
    v_col_info = struct('col_name', {}, 'mode_num', {});

    for k = 1:numel(var_names)
        tok_u = regexp(var_names{k}, '^receiver_u_mode(\d+)$', 'tokens');
        if ~isempty(tok_u)
            u_col_info(end+1) = struct('col_name', var_names{k}, 'mode_num', str2double(tok_u{1}{1})); %#ok<AGROW>
        end
        tok_v = regexp(var_names{k}, '^sender_v_mode(\d+)$', 'tokens');
        if ~isempty(tok_v)
            v_col_info(end+1) = struct('col_name', var_names{k}, 'mode_num', str2double(tok_v{1}{1})); %#ok<AGROW>
        end
    end

    if isempty(u_col_info) || isempty(v_col_info)
        warning('Missing receiver_u_mode or sender_v_mode columns in %s', svd_weights_file);
        return;
    end

    % Sort modes numerically
    [~, idx_u] = sort([u_col_info.mode_num]);
    u_col_info = u_col_info(idx_u);

    [~, idx_v] = sort([v_col_info.mode_num]);
    v_col_info = v_col_info(idx_v);

    u_modes = [u_col_info.mode_num];
    v_modes = [v_col_info.mode_num];

    if ~isequal(u_modes, v_modes)
        warning('Receiver and Sender mode numbers do not match in %s', svd_weights_file);
        return;
    end

    if ~all(ismember({'mode_l', 'singular_value_sigma'}, summary_table.Properties.VariableNames))
        warning('Required columns (mode_l, singular_value_sigma) missing in %s', summary_file);
        return;
    end

    summary_modes = summary_table.mode_l(:).';
    if ~all(ismember(u_modes, summary_modes))
        warning('Mode numbers in SVD contributions table do not match network_svd_modes_summary.csv.');
        return;
    end

    L = numel(u_modes);
    sigma = zeros(L, 1);
    for l = 1:L
        m_num = u_modes(l);
        row_s = (summary_table.mode_l == m_num);
        if ~any(row_s)
            warning('Mode %d not found in %s', m_num, summary_file);
            return;
        end
        sigma(l) = summary_table.singular_value_sigma(row_s);
    end

    N = numel(agents_to_plot);
    U = zeros(N, L);
    V = zeros(N, L);

    for i = 1:N
        real_ag = agents_to_plot(i);
        mode_ag = get_mapped_agent_id(real_ag, agent_id_map);

        row_w = (weights_table.agent_id == mode_ag);
        if ~any(row_w)
            warning('Agent ID %d (mapped from real ID %d) not found in SVD weights table %s.', ...
                mode_ag, real_ag, svd_weights_file);
            return;
        end

        for l = 1:L
            U(i, l) = weights_table.(u_col_info(l).col_name)(row_w);
            V(i, l) = weights_table.(v_col_info(l).col_name)(row_w);
        end
    end

    % Numerical verification of reconstructed matrix W = U * diag(sigma) * V'
    W_reconstructed = U * diag(sigma) * V.';
    test_phasors = exp(1i * 2 * pi * rand(10, N));
    G_test_modes = (test_phasors * V .* sigma.') * U.';
    G_test_direct = test_phasors * W_reconstructed.';
    diff_norm = norm(G_test_modes - G_test_direct, 'fro');
    ref_norm = norm(G_test_direct, 'fro');
    rel_err = diff_norm / max(ref_norm, eps);
    if rel_err > 1e-9
        warning('Numerical verification failed! Relative error between mode sum and direct W product = %.2e', rel_err);
    else
        fprintf('[INFO] Numerical verification passed: mode-sum vs direct W product rel. error = %.2e\n', rel_err);
    end

    y_label_str = '$|G_i(t)|=\left|\sum_{\ell}\sigma_{\ell}u_{i\ell}Z_{\ell}(t)\right|$';

    if overlay_mode
        figure('Visible','on');
        ax = axes('Parent', gcf);
        hold(ax, 'on');
        set(gcf, 'Name', 'Agent-wise Weighted Input Amplitude (Overlay)');

        colors = lines(N);
        line_handles = gobjects(N, 1);
        agent_legend = cell(N, 1);

        for f = 1:numel(file_list)
            series_struct = phase_series_by_file{f};
            if isempty(series_struct)
                continue;
            end

            [t_common, G_abs] = compute_agent_weighted_input_amplitude_for_file( ...
                series_struct, agents_to_plot, U, V, sigma);
            if isempty(t_common)
                continue;
            end

            for i = 1:N
                ag = agents_to_plot(i);
                h = plot(ax, t_common, G_abs(:, i), ...
                    'Color', colors(i,:), 'LineWidth', 1.2, 'LineStyle', line_style);
                if f == 1
                    line_handles(i) = h;
                    disp_id = displayed_agent_id(ag, agent_display_offset);
                    agent_legend{i} = sprintf('Agent %d', disp_id);
                end
            end
        end

        format_weighted_input_amplitude_axis(ax, y_label_str, max_plot_time);
        hold(ax, 'off');

        valid_h = isgraphics(line_handles);
        if any(valid_h)
            legend(ax, line_handles(valid_h), agent_legend(valid_h), ...
                'Location','eastoutside','Interpreter','latex');
        end

        if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
            tuneFigure();
        end

    else
        for f = 1:numel(file_list)
            series_struct = phase_series_by_file{f};
            if isempty(series_struct)
                continue;
            end

            [t_common, G_abs] = compute_agent_weighted_input_amplitude_for_file( ...
                series_struct, agents_to_plot, U, V, sigma);
            if isempty(t_common)
                continue;
            end

            figure('Visible','on');
            ax = axes('Parent', gcf);
            hold(ax, 'on');
            [~, file_label] = fileparts(file_list{f});
            set(gcf, 'Name', sprintf('Agent-wise Weighted Input Amplitude: %s', file_label));

            colors = lines(N);
            line_handles = gobjects(N, 1);
            agent_legend = cell(N, 1);

            for i = 1:N
                ag = agents_to_plot(i);
                line_handles(i) = plot(ax, t_common, G_abs(:, i), ...
                    'Color', colors(i,:), 'LineWidth', 1.2, 'LineStyle', line_style);
                disp_id = displayed_agent_id(ag, agent_display_offset);
                agent_legend{i} = sprintf('Agent %d', disp_id);
            end

            format_weighted_input_amplitude_axis(ax, y_label_str, max_plot_time);
            hold(ax, 'off');

            legend(ax, line_handles, agent_legend, ...
                'Location','eastoutside','Interpreter','latex');

            if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
                tuneFigure();
            end
        end
    end
end

function [t_common, G_abs] = compute_agent_weighted_input_amplitude_for_file(series_struct, agents_to_plot, U, V, sigma)
    t_common = [];
    G_abs = [];

    N = numel(agents_to_plot);
    phase_matrix = [];

    for a_idx = 1:N
        ag = agents_to_plot(a_idx);
        series_entry = get_agent_series_entry(series_struct, ag);
        if isempty(series_entry) || isempty(series_entry.time)
            return;
        end

        if isfield(series_entry, 'absolute_phase_unwrapped') && ~isempty(series_entry.absolute_phase_unwrapped)
            ag_phase = series_entry.absolute_phase_unwrapped(:);
        elseif isfield(series_entry, 'absolute_phase') && ~isempty(series_entry.absolute_phase)
            ag_phase = series_entry.absolute_phase(:);
        else
            return;
        end

        if isempty(t_common)
            t_common = series_entry.time(:);
            phase_matrix = zeros(numel(t_common), N);
        end

        phase_matrix(:, a_idx) = ag_phase;
    end

    if isempty(t_common)
        return;
    end

    phase_phasors = exp(1i * phase_matrix);      % T x N
    Z_complex = phase_phasors * V;              % T x L
    G_complex = (Z_complex .* sigma.') * U.';   % T x N
    G_abs = abs(G_complex);                     % T x N
end

function format_weighted_input_amplitude_axis(ax, y_label_str, max_plot_time)
    ylabel(ax, y_label_str, 'Interpreter', 'latex');
    set(ax, 'TickLabelInterpreter', 'latex');
    xlim(ax, [0, max_plot_time]);
    xlabel(ax, 'Time (s)', 'Interpreter', 'latex');
    grid(ax, 'on');
end
