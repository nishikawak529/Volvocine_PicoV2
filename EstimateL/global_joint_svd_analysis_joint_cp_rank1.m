function all_global_results = global_joint_svd_analysis_joint_cp_rank1(round_dir, M, varargin)
%GLOBAL_JOINT_SVD_ANALYSIS_JOINT_CP_RANK1 Joint rank-1 CP analysis.
%
% This function preserves the data loading, Fourier coefficient estimation,
% marginal-term removal, directed-edge metadata, and network visualization of
% global_joint_svd_analysis.  It replaces the two-stage SVD factorization by
% the direct constrained rank-1 problem
%
%   min sum_p ||C(:,:,p) - w(p) * a * b.'||_F^2,
%
% where ||a||_2=||b||_2=1, w is real, and a and b are conjugate symmetric in
% Fourier order -M:M.  The nonconjugate transpose b.' is intentional.
%
% Each call analyzes only the supplied round_dir.  Separate experimental
% data sets (for example separate colonies, if supplied as separate folders)
% must therefore be passed in separate calls and are factorized independently.
%
% Usage:
%   results = global_joint_svd_analysis_joint_cp_rank1();
%   results = global_joint_svd_analysis_joint_cp_rank1( ...
%       fullfile('EstimateL', 'Round'), 10);
%
% Useful CP options (name-value pairs):
%   'NumStarts'         20
%   'MaxIter'           1000
%   'Tol'               1e-10
%   'RandomSeed'        0
%   'DisplayFullArrays' true
%   'SyntheticTestOnly' false

    % Keep the existing directed-graph color and line-width scales.
    clim_limit = 0.06;
    linewidth_limit = 0.06;

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'SStick');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end

    opts = parse_options(varargin{:});
    validateattributes(M, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');

    % This public path exercises the same local CP solver without data I/O.
    if opts.SyntheticTestOnly
        all_global_results = run_synthetic_validation(M, opts);
        return;
    end

    pair_infos = list_pair_folders(round_dir);
    if isempty(pair_infos)
        error('No pair folders like "7-8" were found under %s.', round_dir);
    end

    % Use a CP-specific output subtree so the original SVD outputs are not
    % overwritten.
    analysis_out_dir = fullfile(round_dir, 'low_rank_analysis', ...
        sprintf('M%d', M), 'global_joint_cp_rank1');
    if ~exist(analysis_out_dir, 'dir')
        mkdir(analysis_out_dir);
    end

    log_file_path = fullfile(analysis_out_dir, ...
        'global_joint_cp_rank1_summary.txt');
    if exist(log_file_path, 'file')
        delete(log_file_path);
    end
    diary(log_file_path);
    diary_cleanup = onCleanup(@() diary('off'));

    fprintf('[INFO] Starting direct joint rank-1 CP analysis\n');
    fprintf('  Round directory: %s\n', round_dir);
    fprintf('  Output directory: %s\n', analysis_out_dir);
    fprintf('  Fourier order: M=%d; coefficient order is -M:M\n', M);
    fprintf('  Remove self-only terms: %s\n', mat2str(opts.RemoveSelfOnly));
    fprintf('  Remove constant term: %s\n', mat2str(opts.RemoveConstant));
    fprintf('  Remove other-only terms: %s\n', mat2str(opts.RemoveOtherOnly));
    fprintf('  CP multi-starts: %d\n', opts.NumStarts);
    fprintf('  CP maximum iterations/start: %d\n', opts.MaxIter);
    fprintf('  CP relative objective tolerance: %.3e\n', opts.Tol);
    fprintf('  CP random seed: %d\n\n', opts.RandomSeed);
    if opts.ProfileRank ~= 1
        fprintf(['[INFO] ProfileRank=%g is accepted for compatibility, but ', ...
            'this function always fits CP rank 1.\n\n'], opts.ProfileRank);
    end

    % ---------------------------------------------------------------------
    % Step 1: Load each pair and estimate the same coefficient matrices as
    % the original analysis.  Rows are receiver modes m and columns are
    % sender modes n for both C1 and C2.
    % ---------------------------------------------------------------------
    pair_data_list = struct([]);
    for k = 1:numel(pair_infos)
        info = pair_infos(k);
        csv_pattern = fullfile(info.folder, '*.csv');
        fprintf('[INFO] Loading Pair %d/%d: %d-%d\n', k, ...
            numel(pair_infos), info.agent_ids(1), info.agent_ids(2));

        try
            pair_out = plot_phase_dynamics_from_csv(csv_pattern, ...
                info.agent_ids, M, ...
                'analysis_start_sec', opts.analysis_start_sec, ...
                'analysis_duration_sec', opts.analysis_duration_sec, ...
                'sample_dt', opts.sample_dt, ...
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

        phi1_all = pair_out.point_cloud.phi1;
        phi2_all = pair_out.point_cloud.phi2;

        [C1_full, m_values, n_values] = estimate_fourier_coeff_matrix( ...
            phi1_all, phi2_all, pair_out.point_cloud.s1, M);
        [C2_full, m_values_2, n_values_2] = estimate_fourier_coeff_matrix( ...
            phi2_all, phi1_all, pair_out.point_cloud.s2, M);

        expected_modes = (-M:M).';
        if ~isequal(m_values, expected_modes) || ...
                ~isequal(n_values, expected_modes) || ...
                ~isequal(m_values_2, expected_modes) || ...
                ~isequal(n_values_2, expected_modes)
            error('Unexpected Fourier coefficient ordering for pair %s.', ...
                info.name);
        end

        [C1_analysis, ~, ~] = remove_phase_marginal_terms( ...
            C1_full, phi1_all, phi2_all, pair_out.point_cloud.s1, ...
            m_values, n_values, info.agent_ids(1), info.agent_ids, ...
            opts.RemoveSelfOnly, opts.RemoveConstant, opts.RemoveOtherOnly);
        [C2_analysis, ~, ~] = remove_phase_marginal_terms( ...
            C2_full, phi2_all, phi1_all, pair_out.point_cloud.s2, ...
            m_values, n_values, info.agent_ids(2), ...
            [info.agent_ids(2), info.agent_ids(1)], ...
            opts.RemoveSelfOnly, opts.RemoveConstant, opts.RemoveOtherOnly);

        entry = struct();
        entry.pair_name = info.name;
        entry.agent_ids = info.agent_ids;
        entry.C1 = C1_analysis; % target first ID <- source second ID
        entry.C2 = C2_analysis; % target second ID <- source first ID
        entry.m_values = m_values;
        entry.n_values = n_values;

        if isempty(pair_data_list)
            pair_data_list = entry;
        else
            pair_data_list(end+1) = entry; %#ok<AGROW>
        end
    end

    if isempty(pair_data_list)
        error('No valid pair data was loaded.');
    end

    % ---------------------------------------------------------------------
    % Step 2: Build C_tensor(:,:,p) and preserve the original p-to-edge map.
    % C2 is already receiver-by-sender and must not be transposed.
    % ---------------------------------------------------------------------
    C_blocks = {};
    interaction_meta = struct('pair_name', {}, 'target_id', {}, ...
        'source_id', {}, 'direction', {});

    for k = 1:numel(pair_data_list)
        p_data = pair_data_list(k);
        C_blocks{end+1} = p_data.C1; %#ok<AGROW>
        meta = struct('pair_name', p_data.pair_name, ...
            'target_id', p_data.agent_ids(1), ...
            'source_id', p_data.agent_ids(2), 'direction', 's1');
        if isempty(interaction_meta)
            interaction_meta = meta;
        else
            interaction_meta(end+1) = meta; %#ok<AGROW>
        end

        C_blocks{end+1} = p_data.C2; %#ok<AGROW>
        meta = struct('pair_name', p_data.pair_name, ...
            'target_id', p_data.agent_ids(2), ...
            'source_id', p_data.agent_ids(1), 'direction', 's2');
        interaction_meta(end+1) = meta; %#ok<AGROW>
    end

    m_vals = pair_data_list(1).m_values(:);
    n_vals = pair_data_list(1).n_values(:);
    n_receiver_modes = numel(m_vals);
    n_sender_modes = numel(n_vals);
    n_blocks = numel(C_blocks);

    C_tensor = complex(zeros(n_receiver_modes, n_sender_modes, n_blocks));
    for p = 1:n_blocks
        C_p = C_blocks{p};
        if ~isequal(size(C_p), [n_receiver_modes, n_sender_modes])
            error('Coefficient matrix p=%d has inconsistent size.', p);
        end
        % Required explicit tensor assembly.
        C_tensor(:, :, p) = C_p;
    end
    P = size(C_tensor, 3);

    if P ~= numel(interaction_meta)
        error('Tensor direction count and interaction metadata do not match.');
    end
    unique_agents = unique([[interaction_meta.target_id], ...
        [interaction_meta.source_id]]);
    complete_direction_count = numel(unique_agents) * ...
        (numel(unique_agents) - 1);
    fprintf('[INFO] Constructed C_tensor with size %d x %d x %d.\n', ...
        size(C_tensor, 1), size(C_tensor, 2), P);
    if P == 12
        fprintf('[INFO] Verified the current data contain all 12 directions.\n');
    else
        warning(['Expected 12 directions for the current four-agent data, ', ...
            'but constructed P=%d after loading/skipping pairs.'], P);
    end
    if P == complete_direction_count
        fprintf('[INFO] Direction metadata form a complete %d-agent digraph.\n', ...
            numel(unique_agents));
    else
        warning('Direction set is not a complete digraph for the loaded agents.');
    end
    for p = 1:P
        fprintf('  p=%2d: %d <- %d (%s, %s)\n', p, ...
            interaction_meta(p).target_id, interaction_meta(p).source_id, ...
            interaction_meta(p).pair_name, interaction_meta(p).direction);
    end
    fprintf('\n');

    tensor_symmetry_partner = conj(flip(flip(C_tensor, 1), 2));
    tensor_norm = norm(C_tensor(:));
    tensor_conjugate_symmetry_relative_error = ...
        norm(C_tensor(:) - tensor_symmetry_partner(:)) / max(tensor_norm, eps);
    fprintf('[INFO] Tensor conjugate-symmetry relative mismatch: %.3e\n\n', ...
        tensor_conjugate_symmetry_relative_error);

    % ---------------------------------------------------------------------
    % Step 3: Direct rank-1 CP-ALS with conjugate-symmetry projection after
    % every a and b update.  No SVD is used in the estimation.
    % ---------------------------------------------------------------------
    previous_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(previous_rng_state));
    rng(opts.RandomSeed, 'twister');
    [a, b, w, cp_info] = fit_joint_cp_rank1_conjsym( ...
        C_tensor, opts.NumStarts, opts.MaxIter, opts.Tol);
    clear rng_cleanup;

    phi_grid = linspace(0, 2*pi, 512).';
    receiver_basis = exp(1i * phi_grid * m_vals.');
    sender_basis = exp(1i * phi_grid * n_vals.');

    % Resolve the remaining two real sign ambiguities deterministically.
    a_complex = receiver_basis * a;
    [~, idx_a_peak] = max(abs(a_complex));
    if real(a_complex(idx_a_peak)) < 0
        a = -a;
        w = -w;
    end
    b_complex = sender_basis * b;
    [~, idx_b_peak] = max(abs(b_complex));
    if real(b_complex(idx_b_peak)) < 0
        b = -b;
        w = -w;
    end

    a_complex = receiver_basis * a;
    b_complex = sender_basis * b;
    max_imag_a = max(abs(imag(a_complex)));
    max_imag_b = max(abs(imag(b_complex)));
    imag_tol_a = 1e-10 * max(1, max(abs(real(a_complex))));
    imag_tol_b = 1e-10 * max(1, max(abs(real(b_complex))));
    fprintf('[INFO] Maximum |imag(a(phi))|: %.3e (tolerance %.3e)\n', ...
        max_imag_a, imag_tol_a);
    fprintf('[INFO] Maximum |imag(b(phi))|: %.3e (tolerance %.3e)\n', ...
        max_imag_b, imag_tol_b);
    if max_imag_a > imag_tol_a
        warning('Receiver waveform has a non-negligible imaginary component.');
    end
    if max_imag_b > imag_tol_b
        warning('Sender waveform has a non-negligible imaginary component.');
    end
    % Only after the explicit checks above, form real arrays for plotting.
    a_values = real(a_complex);
    b_values = real(b_complex);

    % Required nonconjugate reconstruction.
    C_fit = complex(zeros(size(C_tensor)));
    for p = 1:P
        C_fit(:, :, p) = w(p) * a * b.';
    end

    [objective, relative_residual, explained_fraction, ...
        per_direction_relative_residual, per_direction_explained_fraction, ...
        zero_direction] = compute_cp_metrics(C_tensor, C_fit);

    a_symmetry_error = norm(a - conj(flipud(a)));
    b_symmetry_error = norm(b - conj(flipud(b)));

    fprintf('\n[INFO] Selected minimum-objective solution among starts (not a claim of global optimality).\n');
    fprintf('  Selected start: %d of %d\n', cp_info.best_start, opts.NumStarts);
    fprintf('  Accepted ALS iterations: %d\n', cp_info.iterations);
    fprintf('  Converged by relative objective change: %s\n', ...
        mat2str(cp_info.converged));
    fprintf('  Final objective J: %.16e\n', objective);
    fprintf('  Overall relative residual: %.16e\n', relative_residual);
    fprintf('  Explained fraction: %.16e (%.8f%%)\n', ...
        explained_fraction, 100 * explained_fraction);
    fprintf('  ||a||_2: %.16g; ||b||_2: %.16g\n', norm(a), norm(b));
    fprintf('  Conjugate-symmetry error ||a-conj(flipud(a))||_2: %.3e\n', ...
        a_symmetry_error);
    fprintf('  Conjugate-symmetry error ||b-conj(flipud(b))||_2: %.3e\n', ...
        b_symmetry_error);
    fprintf('\n[INFO] Per-direction CP metrics:\n');
    fprintf('%-4s %-10s %-8s %-8s %-15s %-15s %-13s\n', ...
        'p', 'Pair', 'Target', 'Source', 'RelResidual', 'Explained', 'w');
    for p = 1:P
        fprintf('%-4d %-10s %-8d %-8d %-15.8e %-15.8e %-13.6g\n', ...
            p, interaction_meta(p).pair_name, ...
            interaction_meta(p).target_id, interaction_meta(p).source_id, ...
            per_direction_relative_residual(p), ...
            per_direction_explained_fraction(p), w(p));
    end

    fprintf('\n[INFO] a coefficients (-M:M):\n');
    disp(a);
    fprintf('[INFO] b coefficients (-M:M):\n');
    disp(b);
    fprintf('[INFO] signed direction weights w:\n');
    disp(w);
    if opts.DisplayFullArrays
        fprintf('[INFO] Reconstructed coefficient tensor C_fit:\n');
        disp(C_fit);
    else
        fprintf(['[INFO] C_fit display suppressed by DisplayFullArrays=false; ', ...
            'the complete array is stored in the returned result and MAT file.\n']);
    end

    % Per-edge result structures preserve target/source metadata for the
    % graph and simulation export.
    individual_metrics = repmat(struct( ...
        'pair_name', '', 'target_id', 0, 'source_id', 0, ...
        'direction', '', 'total_energy', 0, 'residual_energy', 0, ...
        'relative_residual', 0, 'explained_fraction', 0, ...
        'zero_direction', false, 'weight_r1', 0), P, 1);
    b_values_by_interaction = repmat(struct( ...
        'pair_name', '', 'target_id', 0, 'source_id', 0, ...
        'direction', '', 'b_values', [], 'beta_coeff', []), P, 1);
    for p = 1:P
        C_p = C_tensor(:, :, p);
        R_p = C_p - C_fit(:, :, p);
        individual_metrics(p).pair_name = interaction_meta(p).pair_name;
        individual_metrics(p).target_id = interaction_meta(p).target_id;
        individual_metrics(p).source_id = interaction_meta(p).source_id;
        individual_metrics(p).direction = interaction_meta(p).direction;
        individual_metrics(p).total_energy = sum(abs(C_p(:)).^2);
        individual_metrics(p).residual_energy = sum(abs(R_p(:)).^2);
        individual_metrics(p).relative_residual = ...
            per_direction_relative_residual(p);
        individual_metrics(p).explained_fraction = ...
            per_direction_explained_fraction(p);
        individual_metrics(p).zero_direction = zero_direction(p);
        individual_metrics(p).weight_r1 = w(p);

        b_values_by_interaction(p).pair_name = interaction_meta(p).pair_name;
        b_values_by_interaction(p).target_id = interaction_meta(p).target_id;
        b_values_by_interaction(p).source_id = interaction_meta(p).source_id;
        b_values_by_interaction(p).direction = interaction_meta(p).direction;
        b_values_by_interaction(p).b_values = w(p) * b_values;
        b_values_by_interaction(p).beta_coeff = w(p) * b;
    end

    component = struct();
    component.r = 1;
    component.a_values = a_values(:);
    component.alpha_coeff = a(:);
    component.b_values_by_interaction = b_values_by_interaction;
    component.explained_fraction = explained_fraction;

    all_global_results = struct();
    all_global_results.algorithm = ...
        'multi-start conjugate-symmetric direct rank-1 CP-ALS';
    all_global_results.phi_grid = phi_grid;
    all_global_results.m_values = m_vals;
    all_global_results.n_values = n_vals;
    all_global_results.C_tensor = C_tensor;
    all_global_results.C_fit = C_fit;
    all_global_results.a = a(:);
    all_global_results.b = b(:);
    all_global_results.w = w(:);
    all_global_results.a_values_complex = a_complex(:);
    all_global_results.b_values_complex = b_complex(:);
    all_global_results.a_values = a_values(:);
    all_global_results.b_values = b_values(:);
    all_global_results.max_imag_a = max_imag_a;
    all_global_results.max_imag_b = max_imag_b;
    all_global_results.a_conjugate_symmetry_error = a_symmetry_error;
    all_global_results.b_conjugate_symmetry_error = b_symmetry_error;
    all_global_results.tensor_conjugate_symmetry_relative_error = ...
        tensor_conjugate_symmetry_relative_error;
    all_global_results.objective = objective;
    all_global_results.relative_residual = relative_residual;
    all_global_results.explained_fraction = explained_fraction;
    all_global_results.per_direction_relative_residual = ...
        per_direction_relative_residual;
    all_global_results.per_direction_explained_fraction = ...
        per_direction_explained_fraction;
    all_global_results.iterations = cp_info.iterations;
    all_global_results.best_start = cp_info.best_start;
    all_global_results.cp_info = cp_info;
    all_global_results.components = component;
    all_global_results.interaction_meta = interaction_meta;
    all_global_results.individual_metrics = individual_metrics;
    all_global_results.b_shared_r1 = b_values(:);
    all_global_results.beta_shared_r1 = b(:);
    all_global_results.weights_r1 = w(:);
    all_global_results.unique_agents = unique_agents;
    all_global_results.M = M;
    all_global_results.P = P;
    all_global_results.analysis_out_dir = analysis_out_dir;

    diary('off');
    clear diary_cleanup;
    fprintf('[INFO] Saved CP summary log to %s\n', log_file_path);

    % ---------------------------------------------------------------------
    % Step 4: Plot and save the corresponding rank-1 CP profiles.
    % ---------------------------------------------------------------------
    self_profile_dir = fullfile(analysis_out_dir, 'agent_self_profiles');
    if ~exist(self_profile_dir, 'dir')
        mkdir(self_profile_dir);
    end

    n_interactions = numel(interaction_meta);
    colors = lines(n_interactions);
    legend_labels = cell(1, n_interactions);
    for p = 1:n_interactions
        legend_labels{p} = sprintf('%s (%d->%d)', ...
            interaction_meta(p).pair_name, interaction_meta(p).source_id, ...
            interaction_meta(p).target_id);
    end

    fig_global_overlay = figure('Color', 'w', ...
        'Position', [100, 100, 1000, 460], ...
        'Name', 'Global Joint Rank-1 CP Shared Profiles');
    t_lay = tiledlayout(fig_global_overlay, 1, 2, ...
        'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_lay, sprintf(['Global Joint Rank-1 CP Approximation ', ...
        '(explained = %.1f%%)'], 100 * explained_fraction), ...
        'FontWeight', 'bold', 'FontSize', 12);

    ax_in = nexttile(t_lay, 1);
    hold(ax_in, 'on');
    grid(ax_in, 'on');
    box(ax_in, 'on');
    plot(ax_in, phi_grid, a_values, 'LineWidth', 2.8, ...
        'Color', [0, 0.447, 0.741]);
    ylabel(ax_in, 'a (Shared Receiver)', 'FontWeight', 'bold');
    xlabel(ax_in, '\phi_{target} (Receiver Phase)');
    set_phase_axis(ax_in);
    legend(ax_in, {'Shared Receiver Profile'}, ...
        'Location', 'best', 'FontSize', 8);
    title(ax_in, 'Shared Receiver Profile a(\phi)');

    ax_out = nexttile(t_lay, 2);
    hold(ax_out, 'on');
    grid(ax_out, 'on');
    box(ax_out, 'on');
    for p = 1:n_interactions
        plot(ax_out, phi_grid, w(p) * b_values, ...
            'LineWidth', 1.5, 'Color', colors(p, :));
    end
    mean_weight = mean(abs(w));
    plot(ax_out, phi_grid, mean_weight * b_values, ...
        'k--', 'LineWidth', 3.0);
    ylabel(ax_out, 'w_p b (Directed Senders)', 'FontWeight', 'bold');
    xlabel(ax_out, '\phi_{source} (Sender Phase)');
    set_phase_axis(ax_out);
    legend(ax_out, [legend_labels, {'Shared Sender (mean |w| scale)'}], ...
        'Location', 'eastoutside', 'FontSize', 7, 'Interpreter', 'none');
    title(ax_out, 'Direction-scaled Shared Sender Profiles');

    global_save_path = fullfile(self_profile_dir, ...
        'global_joint_cp_rank1_profiles_overlay.png');
    saveas(fig_global_overlay, global_save_path);
    fprintf('[INFO] Saved rank-1 CP profiles plot to: %s\n', global_save_path);

    % Export phase-domain curves.
    t_data = table(phi_grid, a_values(:), b_values(:), ...
        'VariableNames', {'phi', 'a_shared', 'b_shared'});
    for p = 1:n_interactions
        col_name = sprintf('wb_%s_%dto%d', ...
            interaction_meta(p).pair_name, interaction_meta(p).source_id, ...
            interaction_meta(p).target_id);
        col_name = strrep(col_name, '-', '_');
        t_data.(col_name) = w(p) * b_values(:);
    end
    csv_global_path = fullfile(self_profile_dir, ...
        'global_joint_cp_rank1_profiles_data.csv');
    writetable(t_data, csv_global_path);
    fprintf('[INFO] Saved rank-1 CP profile CSV to: %s\n', csv_global_path);

    t_metrics = struct2table(individual_metrics);
    csv_metrics_path = fullfile(analysis_out_dir, ...
        'global_joint_cp_rank1_individual_metrics.csv');
    writetable(t_metrics, csv_metrics_path);
    fprintf('[INFO] Saved per-direction CP metrics CSV to: %s\n', ...
        csv_metrics_path);

    % Save the full coefficient tensors and solver diagnostics.
    fit_mat_path = fullfile(analysis_out_dir, ...
        'global_joint_cp_rank1_fit.mat');
    save(fit_mat_path, 'C_tensor', 'C_fit', 'a', 'b', 'w', ...
        'm_vals', 'n_vals', 'phi_grid', 'a_complex', 'b_complex', ...
        'cp_info', 'interaction_meta', 'individual_metrics', ...
        'relative_residual', 'explained_fraction');
    fprintf('[INFO] Saved full rank-1 CP fit MAT file to: %s\n', fit_mat_path);

    % Preserve the original simulation export convention, including
    % coupling_matrix(target, source) = w(target <- source).
    sim_params = struct();
    sim_params.M = M;
    sim_params.phi_grid = phi_grid(:);
    sim_params.a_shared_vals = a_values(:);
    sim_params.a_shared_coeff = a(:);
    sim_params.b_shared_vals = b_values(:);
    sim_params.b_shared_coeff = b(:);
    coupling_weights = struct('source_id', {}, 'target_id', {}, 'weight', {});
    for p = 1:P
        coupling_weights(p).source_id = interaction_meta(p).source_id;
        coupling_weights(p).target_id = interaction_meta(p).target_id;
        coupling_weights(p).weight = w(p);
    end
    sim_params.coupling_weights = coupling_weights;
    agent_ids = unique_agents(:).';
    n_agents = numel(agent_ids);
    coupling_matrix = zeros(n_agents, n_agents);
    for p = 1:P
        target_idx = find(agent_ids == interaction_meta(p).target_id, 1);
        source_idx = find(agent_ids == interaction_meta(p).source_id, 1);
        if ~isempty(target_idx) && ~isempty(source_idx)
            coupling_matrix(target_idx, source_idx) = w(p);
        end
    end
    sim_params.agent_ids = agent_ids;
    sim_params.coupling_matrix = coupling_matrix;
    sim_mat_path = fullfile(analysis_out_dir, ...
        'global_joint_cp_rank1_simulation_parameters.mat');
    save(sim_mat_path, '-struct', 'sim_params');
    fprintf('[INFO] Saved CP simulation parameters MAT to: %s\n', sim_mat_path);

    % Preserve the original directed graph orientation and visual scaling.
    fig_digraph = figure('Color', 'w', 'Position', [100, 100, 520, 450], ...
        'Name', 'Global Joint Rank-1 CP Influence Digraph');
    ax_dig = axes('Parent', fig_digraph);
    source_agent_id = [individual_metrics.source_id].';
    target_agent_id = [individual_metrics.target_id].';
    strength = [individual_metrics.weight_r1].';
    node_names = arrayfun(@(id) sprintf('%d', id), unique_agents(:), ...
        'UniformOutput', false);
    source_names = arrayfun(@(id) sprintf('%d', id), source_agent_id(:), ...
        'UniformOutput', false);
    target_names = arrayfun(@(id) sprintf('%d', id), target_agent_id(:), ...
        'UniformOutput', false);
    G_dig = digraph(source_names, target_names, strength(:), node_names);
    [x_data, y_data] = get_preferred_node_positions(G_dig, round_dir);

    p_dig = plot(ax_dig, G_dig, 'XData', x_data, 'YData', y_data, ...
        'NodeLabel', {}, 'ArrowSize', 16, 'ArrowPosition', 0.75, ...
        'MarkerSize', 8, 'NodeColor', [0.15, 0.15, 0.15], ...
        'EdgeColor', [0.0, 0.4470, 0.7410]);
    axis(ax_dig, 'equal');
    margin = 0.45;
    rx = max(x_data, [], 'omitnan') - min(x_data, [], 'omitnan') + 2*margin;
    ry = max(y_data, [], 'omitnan') - min(y_data, [], 'omitnan') + 2*margin;
    max_r = max(rx, ry);
    cx = (min(x_data, [], 'omitnan') + max(x_data, [], 'omitnan')) / 2;
    cy = (min(y_data, [], 'omitnan') + max(y_data, [], 'omitnan')) / 2;
    xlim(ax_dig, [cx - max_r/2, cx + max_r/2]);
    ylim(ax_dig, [cy - max_r/2, cy + max_r/2]);
    title(ax_dig, 'Global Joint Rank-1 CP Directed Influence Graph', ...
        'Interpreter', 'none');

    if numedges(G_dig) > 0
        w_vals = G_dig.Edges.Weight;
        p_dig.LineWidth = scale_edge_width(abs(w_vals), linewidth_limit);
        p_dig.EdgeCData = w_vals;
        p_dig.EdgeColor = 'flat';
        caxis(ax_dig, [-clim_limit, clim_limit]);
        n_colors = 256;
        half_colors = n_colors / 2;
        c_blue = [0.0, 0.0, 0.85];
        c_center = [0.95, 0.95, 0.95];
        c_red = [0.85, 0.0, 0.0];
        custom_cmap = [ ...
            linspace(c_blue(1), c_center(1), half_colors)', ...
            linspace(c_blue(2), c_center(2), half_colors)', ...
            linspace(c_blue(3), c_center(3), half_colors)'; ...
            linspace(c_center(1), c_red(1), half_colors)', ...
            linspace(c_center(2), c_red(2), half_colors)', ...
            linspace(c_center(3), c_red(3), half_colors)'];
        colormap(ax_dig, custom_cmap);
        cb = colorbar(ax_dig);
        cb.Label.String = 'w_p (Coupling Strength)';
    end

    draw_edge_labels(ax_dig, G_dig, x_data, y_data);
    draw_node_labels(ax_dig, G_dig, x_data, y_data);

    graph_save_path = fullfile(self_profile_dir, ...
        'global_joint_cp_rank1_influence_graph.png');
    saveas(fig_digraph, graph_save_path);
    fprintf('[INFO] Saved rank-1 CP influence graph to: %s\n', ...
        graph_save_path);

    if ~opts.keep_figures
        close(fig_digraph);
        close(fig_global_overlay);
    end

    fprintf('[INFO] Global joint rank-1 CP analysis completed: %s\n', ...
        analysis_out_dir);
end

function opts = parse_options(varargin)
    p = inputParser;

    % Data extraction parameters retained from the original function.
    addParameter(p, 'analysis_start_sec', 10, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
    addParameter(p, 'analysis_duration_sec', 80, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'sample_dt', 0.01, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'signal_column', 'a2', ...
        @(x) ischar(x) || isstring(x));
    addParameter(p, 'normalize_signal', true, ...
        @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'tail_percent', 10, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x < 50);
    addParameter(p, 'clip_normalized_signal', true, ...
        @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'clip_limit', 0.5, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'use_cache', true, ...
        @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_indices', [], ...
        @(x) isempty(x) || isnumeric(x));

    % Legacy option accepted for call compatibility; this solver is rank 1.
    addParameter(p, 'ProfileRank', 1, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1);
    addParameter(p, 'RemoveSelfOnly', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'RemoveConstant', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'RemoveOtherOnly', false, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'keep_figures', false, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));

    % Direct CP-ALS parameters.
    addParameter(p, 'NumStarts', 20, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
        x >= 1 && x == floor(x));
    addParameter(p, 'MaxIter', 1000, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
        x >= 1 && x == floor(x));
    addParameter(p, 'Tol', 1e-10, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'RandomSeed', 0, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
        x >= 0 && x == floor(x));
    addParameter(p, 'DisplayFullArrays', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'SyntheticTestOnly', false, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));

    parse(p, varargin{:});
    opts = p.Results;
    opts.signal_column = char(opts.signal_column);
    opts.normalize_signal = logical(opts.normalize_signal);
    opts.clip_normalized_signal = logical(opts.clip_normalized_signal);
    opts.use_cache = logical(opts.use_cache);
    opts.cache_dir = char(opts.cache_dir);
    opts.file_indices = double(opts.file_indices);
    opts.RemoveSelfOnly = logical(opts.RemoveSelfOnly);
    opts.RemoveConstant = logical(opts.RemoveConstant);
    opts.RemoveOtherOnly = logical(opts.RemoveOtherOnly);
    opts.keep_figures = logical(opts.keep_figures);
    opts.NumStarts = double(opts.NumStarts);
    opts.MaxIter = double(opts.MaxIter);
    opts.Tol = double(opts.Tol);
    opts.RandomSeed = double(opts.RandomSeed);
    opts.DisplayFullArrays = logical(opts.DisplayFullArrays);
    opts.SyntheticTestOnly = logical(opts.SyntheticTestOnly);
end

function pair_infos = list_pair_folders(round_dir)
    if isstring(round_dir)
        round_dir = char(round_dir);
    end
    pair_infos = struct([]);
    files = dir(round_dir);
    for i = 1:numel(files)
        if files(i).isdir && ~strcmp(files(i).name, '.') && ...
                ~strcmp(files(i).name, '..') && ...
                ~isempty(regexp(files(i).name, '^\d+-\d+$', 'once'))
            parts = strsplit(files(i).name, '-');
            ids = [str2double(parts{1}), str2double(parts{2})];
            entry = struct('name', files(i).name, ...
                'folder', fullfile(round_dir, files(i).name), ...
                'agent_ids', ids);
            if isempty(pair_infos)
                pair_infos = entry;
            else
                pair_infos(end+1) = entry; %#ok<AGROW>
            end
        end
    end
    if ~isempty(pair_infos)
        [~, order] = sort({pair_infos.name});
        pair_infos = pair_infos(order);
    end
end

function [C, m_values, n_values] = estimate_fourier_coeff_matrix(phi1, phi2, y, M)
    m_values = (-M:M).';
    n_values = (-M:M).';
    term_m = repelem(m_values, numel(n_values));
    term_n = repmat(n_values, numel(m_values), 1);
    n_terms = numel(term_m);
    n_samples = numel(y);
    G = complex(zeros(n_terms, n_terms));
    h = complex(zeros(n_terms, 1));
    chunk_size = 20000;

    for start_idx = 1:chunk_size:n_samples
        end_idx = min(start_idx + chunk_size - 1, n_samples);
        sub_phi1 = phi1(start_idx:end_idx);
        sub_phi2 = phi2(start_idx:end_idx);
        sub_y = y(start_idx:end_idx);
        A_chunk = exp(1i * (sub_phi1 * term_m.' + ...
            sub_phi2 * term_n.'));
        G = G + A_chunk' * A_chunk;
        h = h + A_chunk' * sub_y;
    end

    lambda = 1e-4 * n_samples;
    coeff = (G + lambda * eye(n_terms)) \ h;
    C = reshape(coeff, numel(n_values), numel(m_values)).';
end

function [C_out, y_rec, info] = remove_phase_marginal_terms( ...
        C, phi1, phi2, y, m_values, n_values, target_agent_id, ...
        phase_agent_ids, remove_self, remove_constant, remove_other) %#ok<INUSD>
    term_m = repelem(m_values(:), numel(n_values));
    term_n = repmat(n_values(:), numel(m_values), 1);
    coeff_vector = reshape(C.', [], 1);
    total_energy = sum(abs(coeff_vector).^2);

    is_self_term = (term_m ~= 0 & term_n == 0);
    is_constant_term = (term_m == 0 & term_n == 0);
    is_other_term = (term_m == 0 & term_n ~= 0);
    mask_to_remove = false(size(coeff_vector));
    if remove_self
        mask_to_remove = mask_to_remove | is_self_term;
    end
    if remove_constant
        mask_to_remove = mask_to_remove | is_constant_term;
    end
    if remove_other
        mask_to_remove = mask_to_remove | is_other_term;
    end

    coeff_analysis = coeff_vector;
    coeff_analysis(mask_to_remove) = 0;
    C_out = reshape(coeff_analysis, numel(n_values), numel(m_values)).';
    A = exp(1i * (phi1(:) * term_m.' + phi2(:) * term_n.'));
    y_rec = real(A * coeff_analysis);

    info = struct();
    info.total_coeff_energy = total_energy;
    info.removed_self_energy_ratio = ...
        sum(abs(coeff_vector(is_self_term)).^2) / max(total_energy, eps);
    info.removed_constant_energy_ratio = ...
        sum(abs(coeff_vector(is_constant_term)).^2) / max(total_energy, eps);
    info.removed_other_energy_ratio = ...
        sum(abs(coeff_vector(is_other_term)).^2) / max(total_energy, eps);
    removed_total_energy = sum(abs(coeff_vector(mask_to_remove)).^2);
    info.removed_total_coeff_energy = removed_total_energy;
    info.removed_total_energy_ratio = removed_total_energy / max(total_energy, eps);
    info.remaining_energy_ratio = 1 - info.removed_total_energy_ratio;
    info.reconstruction_mismatch_rmse = ...
        sqrt(mean((y(:) - real(A * coeff_vector)).^2));
end

function [a_best, b_best, w_best, info] = ...
        fit_joint_cp_rank1_conjsym(C_tensor, num_starts, max_iter, tol)
% Direct multi-start block-coordinate CP rank-1 solver.
    if ~isnumeric(C_tensor) || isempty(C_tensor)
        error('C_tensor must be a nonempty numeric array.');
    end
    if any(~isfinite(C_tensor(:)))
        error('C_tensor contains NaN or Inf; the CP objective is undefined.');
    end
    q1 = size(C_tensor, 1);
    q2 = size(C_tensor, 2);
    P = size(C_tensor, 3);
    if q1 ~= q2 || mod(q1, 2) ~= 1
        error('C_tensor must have odd square Fourier-mode slices.');
    end
    if P < 1
        error('C_tensor must contain at least one direction.');
    end

    data_scale = max(abs(C_tensor(:)));
    if data_scale == 0
        a_best = zeros(q1, 1);
        b_best = zeros(q2, 1);
        a_best((q1 + 1) / 2) = 1;
        b_best((q2 + 1) / 2) = 1;
        w_best = zeros(P, 1);
        info = struct('objective', 0, 'normalized_objective', 0, ...
            'iterations', 0, 'best_start', 0, 'converged', true, ...
            'relative_objective_change', 0, 'objective_history', 0, ...
            'start_objectives', zeros(num_starts, 1), ...
            'start_iterations', zeros(num_starts, 1), ...
            'start_converged', true(num_starts, 1), ...
            'start_status', {repmat({'zero tensor'}, num_starts, 1)}, ...
            'data_scale', data_scale);
        return;
    end
    C_work = C_tensor / data_scale;
    norm_floor = 1e-14;

    start_objectives = inf(num_starts, 1);
    start_iterations = zeros(num_starts, 1);
    start_converged = false(num_starts, 1);
    start_status = repmat({'not run'}, num_starts, 1);
    best_normalized_objective = inf;
    a_best = [];
    b_best = [];
    w_best_work = [];
    best_iterations = 0;
    best_start = 0;
    best_converged = false;
    best_relative_change = NaN;
    best_history = [];

    for start_idx = 1:num_starts
        a = random_conjugate_symmetric_unit_vector(q1);
        b = random_conjugate_symmetric_unit_vector(q2);
        w = update_real_weights(C_work, a, b);
        if ~all(isfinite(w)) || norm(w) <= norm_floor
            w = randn(P, 1);
            w_norm = norm(w);
            if ~isfinite(w_norm) || w_norm <= norm_floor
                start_status{start_idx} = 'degenerate initial weights';
                continue;
            end
            w = w / w_norm;
        end

        J_previous = cp_objective(C_work, a, b, w);
        if ~isfinite(J_previous)
            start_status{start_idx} = 'nonfinite initial objective';
            continue;
        end
        history = nan(max_iter + 1, 1);
        history(1) = J_previous;
        accepted_iterations = 0;
        converged = false;
        relative_change = Inf;
        status = 'maximum iterations reached';

        for iter = 1:max_iter
            a_raw = zeros(size(a));
            for p = 1:P
                a_raw = a_raw + w(p) * C_work(:, :, p) * conj(b);
            end
            a_raw = projectConjugateSymmetry(a_raw);
            a_norm = norm(a_raw);
            if ~isfinite(a_norm) || a_norm <= norm_floor
                status = 'degenerate receiver update';
                break;
            end
            a_new = a_raw / a_norm;

            b_raw = zeros(size(b));
            for p = 1:P
                % Nonconjugate transpose is required by a*b.'.
                b_raw = b_raw + w(p) * C_work(:, :, p).' * conj(a_new);
            end
            b_raw = projectConjugateSymmetry(b_raw);
            b_norm = norm(b_raw);
            if ~isfinite(b_norm) || b_norm <= norm_floor
                status = 'degenerate sender update';
                break;
            end
            b_new = b_raw / b_norm;

            w_new = update_real_weights(C_work, a_new, b_new);
            if any(~isfinite(w_new))
                status = 'nonfinite weight update';
                break;
            end

            % The explicit objective is evaluated after every full update.
            J_new = cp_objective(C_work, a_new, b_new, w_new);
            if ~isfinite(J_new)
                status = 'nonfinite objective';
                break;
            end

            a = a_new;
            b = b_new;
            w = w_new;
            accepted_iterations = iter;
            history(iter + 1) = J_new;
            relative_change = abs(J_previous - J_new) / ...
                max([abs(J_previous), abs(J_new), eps]);
            J_previous = J_new;

            if relative_change <= tol
                converged = true;
                status = 'converged';
                break;
            end
        end

        start_objectives(start_idx) = J_previous;
        start_iterations(start_idx) = accepted_iterations;
        start_converged(start_idx) = converged;
        start_status{start_idx} = status;

        % Strict < makes equal-objective ties deterministic (earliest start).
        if J_previous < best_normalized_objective
            best_normalized_objective = J_previous;
            a_best = a;
            b_best = b;
            w_best_work = w;
            best_iterations = accepted_iterations;
            best_start = start_idx;
            best_converged = converged;
            best_relative_change = relative_change;
            best_history = history(1:accepted_iterations + 1);
        end
    end

    if isempty(a_best) || isempty(b_best) || isempty(w_best_work)
        error('All CP-ALS starts failed to produce a finite iterate.');
    end

    % Projection is repeated before returning to make the constraints
    % explicit even after all finite-precision operations.
    a_best = projectConjugateSymmetry(a_best);
    b_best = projectConjugateSymmetry(b_best);
    a_best = a_best / norm(a_best);
    b_best = b_best / norm(b_best);
    w_best = data_scale * update_real_weights(C_work, a_best, b_best);
    objective = cp_objective(C_tensor, a_best, b_best, w_best);

    info = struct();
    info.objective = objective;
    info.normalized_objective = best_normalized_objective;
    info.iterations = best_iterations;
    info.best_start = best_start;
    info.converged = best_converged;
    info.relative_objective_change = best_relative_change;
    info.objective_history = best_history;
    info.start_objectives = start_objectives;
    info.start_iterations = start_iterations;
    info.start_converged = start_converged;
    info.start_status = start_status;
    info.data_scale = data_scale;
end

function x = projectConjugateSymmetry(x)
% Projection for coefficient order -M:M.
    x = 0.5 * (x + conj(flipud(x)));
    center_idx = (numel(x) + 1) / 2;
    x(center_idx) = real(x(center_idx));
end

function x = random_conjugate_symmetric_unit_vector(n)
    x = randn(n, 1) + 1i * randn(n, 1);
    x = projectConjugateSymmetry(x);
    x_norm = norm(x);
    if ~isfinite(x_norm) || x_norm <= 1e-14
        x = zeros(n, 1);
        x((n + 1) / 2) = 1;
    else
        x = x / x_norm;
    end
end

function w = update_real_weights(C_tensor, a, b)
    P = size(C_tensor, 3);
    w = zeros(P, 1);
    for p = 1:P
        % a' is conjugate transpose; conj(b) matches the b.' model.
        w(p) = real(a' * C_tensor(:, :, p) * conj(b));
    end
end

function J = cp_objective(C_tensor, a, b, w)
    P = size(C_tensor, 3);
    J = 0;
    for p = 1:P
        residual = C_tensor(:, :, p) - w(p) * a * b.';
        J = J + sum(abs(residual(:)).^2);
    end
end

function [objective, relative_residual, explained_fraction, ...
        per_relative, per_explained, zero_direction] = ...
        compute_cp_metrics(C_tensor, C_fit)
    P = size(C_tensor, 3);
    residual_tensor = C_tensor - C_fit;
    data_norm = norm(C_tensor(:));
    residual_norm = norm(residual_tensor(:));
    objective = residual_norm^2;
    if data_norm == 0
        if residual_norm == 0
            relative_residual = 0;
            explained_fraction = 1;
        else
            relative_residual = Inf;
            explained_fraction = -Inf;
        end
    else
        relative_residual = residual_norm / data_norm;
        explained_fraction = 1 - (residual_norm / data_norm)^2;
    end

    per_relative = zeros(P, 1);
    per_explained = zeros(P, 1);
    zero_direction = false(P, 1);
    for p = 1:P
        C_p = C_tensor(:, :, p);
        R_p = C_p - C_fit(:, :, p);
        C_norm = norm(C_p(:));
        R_norm = norm(R_p(:));
        if C_norm == 0
            zero_direction(p) = true;
            if R_norm == 0
                per_relative(p) = 0;
                per_explained(p) = 1;
            else
                per_relative(p) = Inf;
                per_explained(p) = -Inf;
            end
        else
            per_relative(p) = R_norm / C_norm;
            per_explained(p) = 1 - (R_norm / C_norm)^2;
        end
    end
end

function result = run_synthetic_validation(M, opts)
% Exact noiseless test using the same local direct CP solver.
    P = 12;
    q = 2 * M + 1;
    old_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(old_rng_state));
    rng(opts.RandomSeed, 'twister');

    a_true = random_conjugate_symmetric_unit_vector(q);
    b_true = random_conjugate_symmetric_unit_vector(q);
    w_true = linspace(-0.9, 0.8, P).' + 0.05 * randn(P, 1);
    C_tensor = complex(zeros(q, q, P));
    for p = 1:P
        C_tensor(:, :, p) = w_true(p) * a_true * b_true.';
    end

    [a, b, w, info] = fit_joint_cp_rank1_conjsym( ...
        C_tensor, opts.NumStarts, opts.MaxIter, opts.Tol);
    C_fit = complex(zeros(size(C_tensor)));
    for p = 1:P
        C_fit(:, :, p) = w(p) * a * b.';
    end
    relative_error = norm(C_tensor(:) - C_fit(:)) / norm(C_tensor(:));
    a_symmetry_error = norm(a - conj(flipud(a)));
    b_symmetry_error = norm(b - conj(flipud(b)));

    fprintf('[SYNTHETIC TEST] q=%d, P=%d\n', q, P);
    fprintf('  Relative reconstruction error: %.16e\n', relative_error);
    fprintf('  Objective: %.16e\n', info.objective);
    fprintf('  Selected start: %d; iterations: %d\n', ...
        info.best_start, info.iterations);
    fprintf('  ||a||_2=%.16g, ||b||_2=%.16g\n', norm(a), norm(b));
    fprintf('  Symmetry errors: a=%.3e, b=%.3e\n', ...
        a_symmetry_error, b_symmetry_error);
    if opts.DisplayFullArrays
        fprintf('  Estimated a:\n');
        disp(a);
        fprintf('  Estimated b:\n');
        disp(b);
        fprintf('  Estimated w:\n');
        disp(w);
        fprintf('  Reconstructed C_fit:\n');
        disp(C_fit);
    end

    if relative_error > 1e-10 || ...
            abs(norm(a) - 1) > 1e-12 || abs(norm(b) - 1) > 1e-12 || ...
            a_symmetry_error > 1e-12 || b_symmetry_error > 1e-12 || ...
            any(~isfinite(C_fit(:)))
        error('Synthetic rank-1 CP validation failed.');
    end

    result = struct();
    result.passed = true;
    result.relative_error = relative_error;
    result.C_tensor = C_tensor;
    result.C_fit = C_fit;
    result.a_true = a_true;
    result.b_true = b_true;
    result.w_true = w_true;
    result.a = a;
    result.b = b;
    result.w = w;
    result.a_symmetry_error = a_symmetry_error;
    result.b_symmetry_error = b_symmetry_error;
    result.cp_info = info;
    clear rng_cleanup;
end

function set_phase_axis(ax)
    set(ax, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
    set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
    xlim(ax, [0, 2*pi]);
end

function [x_data, y_data] = get_preferred_node_positions(G, round_dir)
    node_names = G.Nodes.Name;
    if isstring(node_names)
        node_names = cellstr(node_names);
    end
    node_ids = cellfun(@str2double, node_names);
    x_data = nan(1, numnodes(G));
    y_data = nan(1, numnodes(G));

    is_round6 = (nargin >= 2 && contains(lower(round_dir), 'round6')) || ...
        all(ismember([7, 8, 9, 10, 11, 12], node_ids));

    if is_round6
        % Hexagonal arrangement for Round 6 (7..12):
        %    9   12
        %  8        11
        %    7   10
        preferred_ids = [7, 8, 9, 10, 11, 12];
        preferred_x   = [1.5, 1.0, 1.5, 2.5, 3.0, 2.5];
        preferred_y   = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    else
        % Standard 4-agent layout (7, 8, 9, 10):
        % 8  10
        % 7   9
        preferred_ids = [8, 10, 7, 9];
        preferred_x   = [1, 2, 1, 2];
        preferred_y   = [2, 2, 1, 1];
    end

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
        cx = mean(x_data(~missing), 'omitnan');
        cy = mean(y_data(~missing), 'omitnan');
        if ~isfinite(cx), cx = 2.0; end
        if ~isfinite(cy), cy = 2.0; end
        x_data(missing) = cx + 1.0 * cos(angles(1:end-1));
        y_data(missing) = cy + 1.0 * sin(angles(1:end-1));
    end
end

function widths = scale_edge_width(weights, max_limit)
    weights = double(weights(:));
    if isempty(weights) || all(~isfinite(weights))
        widths = 1.5;
        return;
    end
    if nargin >= 2 && ~isempty(max_limit)
        clamped_weights = min(max(weights, 0), max_limit);
        widths = 1.0 + 5.0 * (clamped_weights / max_limit);
    else
        w_min = min(weights, [], 'omitnan');
        w_max = max(weights, [], 'omitnan');
        if ~isfinite(w_min) || ~isfinite(w_max) || ...
                abs(w_max - w_min) < eps
            widths = 2.5 * ones(size(weights));
            return;
        end
        widths = 1.0 + 5.0 * (weights - w_min) / (w_max - w_min);
    end
end

function draw_edge_labels(ax, G, x_data, y_data)
    offset_dist = 0.08;
    for e = 1:numedges(G)
        source_node = G.Edges.EndNodes{e, 1};
        target_node = G.Edges.EndNodes{e, 2};
        source_idx = find(strcmp(G.Nodes.Name, source_node), 1, 'first');
        target_idx = find(strcmp(G.Nodes.Name, target_node), 1, 'first');
        xs = x_data(source_idx);
        ys = y_data(source_idx);
        xt = x_data(target_idx);
        yt = y_data(target_idx);
        dx = xt - xs;
        dy = yt - ys;
        edge_length = hypot(dx, dy);
        if edge_length > 0
            x_label = xs + 0.35 * dx + offset_dist * dy / edge_length;
            y_label = ys + 0.35 * dy - offset_dist * dx / edge_length;
        else
            x_label = xs;
            y_label = ys + offset_dist;
        end
        text(ax, x_label, y_label, sprintf('%.3g', G.Edges.Weight(e)), ...
            'FontSize', 9, 'Color', [0.1, 0.1, 0.1], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'BackgroundColor', 'w', 'Margin', 1);
    end
end

function draw_node_labels(ax, G, x_data, y_data)
    node_offset = 0.12;
    cx = mean(x_data, 'omitnan');
    cy = mean(y_data, 'omitnan');
    if ~isfinite(cx), cx = 1.5; end
    if ~isfinite(cy), cy = 1.5; end
    for n = 1:numnodes(G)
        x_node = x_data(n);
        y_node = y_data(n);
        dx = x_node - cx;
        dy = y_node - cy;
        radial_length = hypot(dx, dy);
        if radial_length > 0
            x_label = x_node + node_offset * dx / radial_length;
            y_label = y_node + node_offset * dy / radial_length;
        else
            x_label = x_node;
            y_label = y_node + node_offset;
        end
        text(ax, x_label, y_label, G.Nodes.Name{n}, ...
            'FontSize', 12, 'FontWeight', 'bold', ...
            'Color', [0.15, 0.15, 0.15], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
end
