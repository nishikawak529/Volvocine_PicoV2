function summary = plot_round_influence_graph(round_dir, M, N, varargin)
% Run pairwise phase-dynamics analysis for all pair folders and plot a digraph.
%
% Each pair folder must be named like "7-8" and contain CSV files. For a
% pair [i j], Gamma_1 is interpreted as the influence j -> i, and Gamma_2
% as the influence i -> j. Influence strength is the first-harmonic sine-fit
% amplitude after removing the constant bias:
%   Gamma(psi) ~= bias + a*sin(psi) + b*cos(psi)
%   strength = sqrt(a^2 + b^2)

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
    pair_infos = list_pair_folders(round_dir);
    if isempty(pair_infos)
        error('No pair folders like "7-8" were found under %s.', round_dir);
    end

    edge_rows = struct([]);
    pair_results = struct([]);
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
                'save_output', false);
        catch ME
            warning('Skipping pair folder %s: %s', info.folder, ME.message);
            continue;
        end

        if ~opts.keep_pair_figures
            close_pair_figures(pair_out);
        end

        row_gamma1 = build_edge_row(pair_out, info, 2, 1, pair_out.gamma1, 'Gamma_1');
        row_gamma2 = build_edge_row(pair_out, info, 1, 2, pair_out.gamma2_minus_psi, 'Gamma_2_minus_psi');
        if isempty(edge_rows)
            edge_rows = [row_gamma1; row_gamma2]; %#ok<AGROW>
        else
            edge_rows(end + 1, 1) = row_gamma1; %#ok<AGROW>
            edge_rows(end + 1, 1) = row_gamma2; %#ok<AGROW>
        end

        pair_results = append_pair_result(pair_results, info, pair_out); %#ok<AGROW>
    end

    if isempty(edge_rows)
        error('No valid pair analyses were completed.');
    end

    edge_table = struct2table(edge_rows);
    graph_info = build_influence_digraph(edge_table, opts);
    figures = struct();
    figures.digraph = plot_influence_digraph(graph_info.G, edge_table, opts);
    figures.matrix = plot_influence_matrix(edge_table, graph_info.node_ids);

    summary = struct();
    summary.round_dir = round_dir;
    summary.M = M;
    summary.N = N;
    summary.options = opts;
    summary.edge_table = edge_table;
    summary.graph = graph_info.G;
    summary.node_ids = graph_info.node_ids;
    summary.adjacency_strength = graph_info.adjacency_strength;
    summary.pair_results = pair_results;
    summary.figures = figures;

    if opts.save_output
        summary.export = save_summary_outputs(summary, opts);
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
    addParameter(p, 'keep_pair_figures', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'save_output', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'edge_weight', 'strength', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});

    opts = p.Results;
    opts.n_psi = round(opts.n_psi);
    opts.n_theta = round(opts.n_theta);
    opts.signal_column = char(opts.signal_column);
    opts.normalize_signal = logical(opts.normalize_signal);
    opts.clip_normalized_signal = logical(opts.clip_normalized_signal);
    opts.keep_pair_figures = logical(opts.keep_pair_figures);
    opts.save_output = logical(opts.save_output);
    opts.output_dir = char(opts.output_dir);
    opts.edge_weight = lower(char(opts.edge_weight));
    if ~ismember(opts.edge_weight, {'strength', 'sigma_strength'})
        error('edge_weight must be ''strength'' or ''sigma_strength''.');
    end
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

function row = build_edge_row(pair_out, info, source_pair_index, target_pair_index, gamma_values, gamma_name)
    source_agent_id = info.agent_ids(source_pair_index);
    target_agent_id = info.agent_ids(target_pair_index);
    fit = fit_first_harmonic(pair_out.psi, gamma_values);

    row = struct();
    row.pair_name = string(info.name);
    row.pair_folder = string(info.folder);
    row.source_agent_id = source_agent_id;
    row.target_agent_id = target_agent_id;
    row.gamma_name = string(gamma_name);
    row.bias = fit.bias;
    row.sin_coefficient = fit.sin_coefficient;
    row.cos_coefficient = fit.cos_coefficient;
    row.phase_rad = fit.phase_rad;
    row.strength = fit.amplitude;
    row.sigma_strength = pair_out.sigma * fit.amplitude;
    row.r2 = fit.r2;
    row.rmse = fit.rmse;
    row.delta_omega = pair_out.delta_omega;
    row.n_points = numel(pair_out.point_cloud.phi1);
    row.fit_s1_rmse = pair_out.fit_s1.rmse;
    row.fit_s2_rmse = pair_out.fit_s2.rmse;
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

function pair_results = append_pair_result(pair_results, info, pair_out)
    result = struct();
    result.pair_name = info.name;
    result.folder = info.folder;
    result.agent_ids = info.agent_ids;
    result.psi = pair_out.psi;
    result.gamma1 = pair_out.gamma1;
    result.gamma2_minus_psi = pair_out.gamma2_minus_psi;
    result.psi_dot = pair_out.psi_dot;
    result.delta_omega = pair_out.delta_omega;
    result.omega_rad_s = pair_out.omega_rad_s;
    result.fit_s1_rmse = pair_out.fit_s1.rmse;
    result.fit_s2_rmse = pair_out.fit_s2.rmse;

    if isempty(pair_results)
        pair_results = result;
    else
        pair_results(end + 1, 1) = result;
    end
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

function graph_info = build_influence_digraph(edge_table, opts)
    node_ids = unique([edge_table.source_agent_id; edge_table.target_agent_id]);
    node_ids = sort(node_ids(:));
    node_names = arrayfun(@(id) sprintf('%d', id), node_ids, 'UniformOutput', false);
    source_names = arrayfun(@(id) sprintf('%d', id), edge_table.source_agent_id, 'UniformOutput', false);
    target_names = arrayfun(@(id) sprintf('%d', id), edge_table.target_agent_id, 'UniformOutput', false);

    switch opts.edge_weight
        case 'strength'
            weights = edge_table.strength;
        case 'sigma_strength'
            weights = edge_table.sigma_strength;
    end

    G = digraph(source_names, target_names, weights, node_names);
    adjacency_strength = nan(numel(node_ids));
    for k = 1:height(edge_table)
        source_idx = find(node_ids == edge_table.source_agent_id(k), 1, 'first');
        target_idx = find(node_ids == edge_table.target_agent_id(k), 1, 'first');
        adjacency_strength(target_idx, source_idx) = edge_table.strength(k);
    end

    graph_info = struct();
    graph_info.G = G;
    graph_info.node_ids = node_ids;
    graph_info.adjacency_strength = adjacency_strength;
end

function fig = plot_influence_digraph(G, edge_table, opts)
    fig = figure('Color', 'w', 'Name', 'Round influence digraph');
    ax = axes('Parent', fig);
    edge_labels = arrayfun(@(x) sprintf('%.3g', x), G.Edges.Weight, 'UniformOutput', false);
    [x_data, y_data] = get_preferred_node_positions(G);
    p = plot(ax, G, 'XData', x_data, 'YData', y_data, 'EdgeLabel', edge_labels, ...
        'ArrowSize', 13, 'NodeFontSize', 12, 'MarkerSize', 8, ...
        'NodeColor', [0.15, 0.15, 0.15], 'EdgeColor', [0.0, 0.4470, 0.7410]);
    axis(ax, 'equal');
    xlim(ax, [0.5, 2.5]);
    ylim(ax, [0.5, 2.5]);
    title(ax, sprintf('Directed influence graph (%s)', opts.edge_weight), 'Interpreter', 'none');

    if numedges(G) > 0
        weights = G.Edges.Weight;
        p.LineWidth = scale_edge_width(weights);
        p.EdgeCData = weights;
        colormap(ax, parula);
        cb = colorbar(ax);
        cb.Label.String = opts.edge_weight;
    end

    subtitle_text = sprintf('edge i -> j: first-harmonic amplitude of Gamma effect on agent j from agent i');
    text(ax, 0.5, -0.08, subtitle_text, 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'Interpreter', 'none');

end

function [x_data, y_data] = get_preferred_node_positions(G)
    node_names = G.Nodes.Name;
    if isstring(node_names)
        node_names = cellstr(node_names);
    end
    node_ids = cellfun(@str2double, node_names);

    x_data = nan(1, numnodes(G));
    y_data = nan(1, numnodes(G));

    preferred_ids = [8, 10, 7, 9];
    preferred_x = [1, 2, 1, 2];
    preferred_y = [2, 2, 1, 1];
    for k = 1:numel(preferred_ids)
        idx = find(node_ids == preferred_ids(k), 1, 'first');
        if ~isempty(idx)
            x_data(idx) = preferred_x(k);
            y_data(idx) = preferred_y(k);
        end
    end

    missing = ~isfinite(x_data) | ~isfinite(y_data);
    if any(missing)
        n_missing = nnz(missing);
        angles = linspace(0, 2*pi, n_missing + 1);
        x_data(missing) = 1.5 + 0.75 * cos(angles(1:end-1));
        y_data(missing) = 1.5 + 0.75 * sin(angles(1:end-1));
    end
end

function widths = scale_edge_width(weights)
    weights = double(weights(:));
    if isempty(weights) || all(~isfinite(weights))
        widths = 1.5;
        return;
    end
    w_min = min(weights, [], 'omitnan');
    w_max = max(weights, [], 'omitnan');
    if ~isfinite(w_min) || ~isfinite(w_max) || abs(w_max - w_min) < eps
        widths = 2.5 * ones(size(weights));
        return;
    end
    widths = 1.0 + 5.0 * (weights - w_min) / (w_max - w_min);
end

function fig = plot_influence_matrix(edge_table, node_ids)
    labels = arrayfun(@(id) sprintf('%d', id), node_ids, 'UniformOutput', false);
    A = nan(numel(node_ids));
    for k = 1:height(edge_table)
        source_idx = find(node_ids == edge_table.source_agent_id(k), 1, 'first');
        target_idx = find(node_ids == edge_table.target_agent_id(k), 1, 'first');
        A(target_idx, source_idx) = edge_table.strength(k);
    end

    fig = figure('Color', 'w', 'Name', 'Round influence matrix');
    ax = axes('Parent', fig);
    imagesc(ax, A);
    axis(ax, 'equal', 'tight');
    colorbar(ax);
    colormap(ax, parula);
    xticks(ax, 1:numel(labels));
    yticks(ax, 1:numel(labels));
    xticklabels(ax, labels);
    yticklabels(ax, labels);
    xlabel(ax, 'source agent');
    ylabel(ax, 'target agent');
    title(ax, 'Influence strength: target row, source column');
end

function export = save_summary_outputs(summary, opts)
    output_dir = opts.output_dir;
    if isempty(output_dir)
        output_dir = fullfile(summary.round_dir, 'influence_graph_exports');
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    csv_path = fullfile(output_dir, 'round_influence_edges.csv');
    png_graph_path = fullfile(output_dir, 'round_influence_digraph.png');
    png_matrix_path = fullfile(output_dir, 'round_influence_matrix.png');

    writetable(summary.edge_table, csv_path);
    if isfield(summary.figures, 'digraph') && ishandle(summary.figures.digraph)
        saveas(summary.figures.digraph, png_graph_path);
    end
    if isfield(summary.figures, 'matrix') && ishandle(summary.figures.matrix)
        saveas(summary.figures.matrix, png_matrix_path);
    end

    export = struct();
    export.output_dir = output_dir;
    export.csv_path = csv_path;
    export.png_graph_path = png_graph_path;
    export.png_matrix_path = png_matrix_path;
end
