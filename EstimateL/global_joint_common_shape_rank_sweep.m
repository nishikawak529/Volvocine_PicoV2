function rank_sweep_results = global_joint_common_shape_rank_sweep(round_dir, M, varargin)
%GLOBAL_JOINT_COMMON_SHAPE_RANK_SWEEP Sweep the matrix rank of one common shape.
%
% This is NOT a rank-R CP model with direction-specific component weights.
% Every directed interaction p uses one real signed scalar w(p):
%
%   C_fit(:,:,p) = w(p) * G_complex,
%   rank(G_real) <= R,  ||G_real||_F = 1,
%   G_complex = T * G_real * T.'.
%
% The matrix rank R is the number of separable terms in the single common
% bivariate interaction function.  All terms share the same P-by-1 vector w.
%
% Usage:
%   results = global_joint_common_shape_rank_sweep();
%   results = global_joint_common_shape_rank_sweep( ...
%       fullfile('EstimateL','SStick'), 10, 'Ranks', 1:10);
%
% Main solver defaults:
%   Ranks      = 1:10
%   NumStarts  = 20
%   MaxIter    = 1000
%   Tol        = 1e-10
%   RandomSeed = 0

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'Round6');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end

    opts = parse_options(varargin{:});
    validateattributes(M, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');
    ranks_requested = opts.Ranks(:).';
    if any(diff(ranks_requested) <= 0)
        error('Ranks must contain unique values in strictly increasing order.');
    end
    L_expected = 2 * M + 1;
    if max(ranks_requested) > L_expected
        error('Every requested rank must be at most 2*M+1=%d.', L_expected);
    end

    if opts.SyntheticTestOnly
        rank_sweep_results = run_synthetic_validation(M, opts);
        return;
    end

    output_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), ...
        sprintf('global_joint_common_shape_rank_sweep_R%d_R%d', ...
        ranks_requested(1), ranks_requested(end)));
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    log_path = fullfile(output_dir, ...
        'global_joint_common_shape_rank_sweep_summary.txt');
    if exist(log_path, 'file')
        delete(log_path);
    end
    diary(log_path);
    diary_cleanup = onCleanup(@() diary('off'));

    fprintf('[INFO] Common-shape matrix-rank sweep (not rank-R CP)\n');
    fprintf('  Data directory: %s\n', round_dir);
    fprintf('  Output directory: %s\n', output_dir);
    fprintf('  Fourier order M=%d; requested ranks=%s\n', ...
        M, mat2str(ranks_requested));
    fprintf('  One shared real weight vector w has size P x 1 at every rank.\n');
    fprintf('  NumStarts=%d, MaxIter=%d, Tol=%.3e, RandomSeed=%d\n\n', ...
        opts.NumStarts, opts.MaxIter, opts.Tol, opts.RandomSeed);

    total_timer = tic;

    % Build the coefficient tensor exactly once.  No rank-dependent data
    % loading or Fourier estimation occurs below this call.
    [C_tensor, interaction_meta, m_values, n_values, unique_agents] = ...
        build_coefficient_tensor(round_dir, M, opts);
    P = size(C_tensor, 3);
    L = size(C_tensor, 1);
    if ~isequal(size(C_tensor, 1), size(C_tensor, 2), L_expected)
        error('C_tensor must have size (2*M+1) x (2*M+1) x P.');
    end
    if ~isequal(m_values(:), (-M:M).') || ...
            ~isequal(n_values(:), (-M:M).')
        error('Fourier coefficient order must be -M:M for both axes.');
    end
    if any(~isfinite(C_tensor(:)))
        error('C_tensor contains NaN or Inf.');
    end
    tensor_norm = norm(C_tensor(:));
    if tensor_norm == 0
        error('C_tensor has zero norm; explained fraction is undefined.');
    end
    fprintf('[INFO] C_tensor constructed once with size %d x %d x %d.\n', ...
        L, L, P);
    if P == 12
        fprintf('[INFO] Verified 12 directed interactions in the current data.\n');
    else
        warning('Current tensor has P=%d directions rather than 12.', P);
    end
    for p = 1:P
        fprintf('  p=%2d: %d -> %d (%s, %s)\n', p, ...
            interaction_meta(p).source_id, interaction_meta(p).target_id, ...
            interaction_meta(p).pair_name, interaction_meta(p).direction);
    end
    fprintf('\n');

    % Complex exponential coefficients <-> real orthonormal trigonometric
    % coefficients.
    [T, transform_info] = build_real_fourier_transform(m_values, M);
    [D_tensor, transform_info] = transform_coefficient_tensor( ...
        C_tensor, T, transform_info);
    fprintf('[INFO] Real-basis transform diagnostics:\n');
    fprintf('  ||E*T-Psi||_F                  = %.16e\n', ...
        transform_info.basis_reconstruction_error);
    fprintf('  ||T''*T-I||_F                  = %.16e\n', ...
        transform_info.unitarity_error);
    fprintf('  C -> D -> C relative error     = %.16e\n', ...
        transform_info.tensor_roundtrip_relative_error);
    fprintf('  ||imag(D_tensor)||/||D_tensor||= %.16e\n\n', ...
        transform_info.D_imaginary_relative_norm);
    if transform_info.D_imaginary_relative_norm > 1e-10
        warning(['D_tensor has a non-negligible imaginary part. Updates use ', ...
            'real(D_tensor), but all reported objectives use complex C_tensor.']);
    end

    phi_grid = linspace(0, 2*pi, 512).';

    % Read an existing rank-1 MAT file without running or overwriting the
    % existing rank-1 analysis.  It is accepted only if C_tensor matches.
    rank1_reference = load_existing_rank1_reference( ...
        round_dir, M, opts, C_tensor, D_tensor, T);
    if rank1_reference.available
        fprintf('[INFO] Existing rank-1 result accepted as a read-only baseline.\n');
        fprintf('  Reference MAT: %s\n', rank1_reference.path);
        fprintf('  Tensor mismatch: %.3e\n', rank1_reference.tensor_relative_mismatch);
        fprintf('  Existing objective: %.16e\n', rank1_reference.objective);
        fprintf('  Existing explained: %.10f%%\n\n', ...
            100 * rank1_reference.explained_fraction);
        rank1_baseline = rank1_reference.baseline;
    else
        fprintf('[INFO] Existing rank-1 baseline unavailable: %s\n\n', ...
            rank1_reference.reason);
        rank1_baseline = struct([]);
    end

    old_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(old_rng_state));
    rng(opts.RandomSeed, 'twister');

    % Missing intermediate ranks are evaluated internally so that every
    % R>1 uses the best rank-(R-1) w as start 1 and retains the complete
    % rank-(R-1) fit as a feasible baseline.
    max_rank = max(ranks_requested);
    [all_rank_fits, all_solver_diagnostics] = fit_rank_sequence( ...
        C_tensor, D_tensor, T, m_values, n_values, phi_grid, ...
        max_rank, opts, rank1_baseline);
    clear rng_cleanup;

    requested_fits = all_rank_fits(ranks_requested).';
    K = numel(ranks_requested);
    explained_fraction = reshape([requested_fits.explained_fraction], K, 1);
    explained_percent = reshape([requested_fits.explained_percent], K, 1);
    incremental_explained_fraction = reshape( ...
        [requested_fits.incremental_explained_fraction], K, 1);
    relative_residual = reshape([requested_fits.relative_residual], K, 1);
    objective = reshape([requested_fits.objective], K, 1);
    runtime_seconds = reshape([requested_fits.runtime_seconds], K, 1);
    weights_by_rank = zeros(P, K);
    per_direction_relative_residual = zeros(P, K);
    per_direction_explained_fraction = zeros(P, K);
    for k = 1:K
        if ~isequal(size(requested_fits(k).w), [P, 1]) || ...
                ~isreal(requested_fits(k).w)
            error('Rank %d did not return one real P-by-1 weight vector.', ...
                ranks_requested(k));
        end
        weights_by_rank(:, k) = requested_fits(k).w;
        per_direction_relative_residual(:, k) = ...
            requested_fits(k).per_direction_relative_residual;
        per_direction_explained_fraction(:, k) = ...
            requested_fits(k).per_direction_explained_fraction;
    end

    % Same-tensor rank-1 comparison; signs are deliberately not compared.
    rank1_comparison = compare_rank1_results( ...
        rank1_reference, all_rank_fits(1), C_tensor);
    if rank1_comparison.available
        fprintf('[INFO] Rank-1 equivalence comparison on the same C_tensor:\n');
        fprintf('  Sweep objective:      %.16e\n', rank1_comparison.sweep_objective);
        fprintf('  Existing objective:   %.16e\n', rank1_comparison.reference_objective);
        fprintf('  Absolute difference:  %.3e\n', rank1_comparison.objective_absolute_difference);
        fprintf('  Sweep explained:      %.10f%%\n', ...
            100 * rank1_comparison.sweep_explained_fraction);
        fprintf('  Existing explained:   %.10f%%\n', ...
            100 * rank1_comparison.reference_explained_fraction);
        fprintf('  C_fit relative difference: %.3e\n\n', ...
            rank1_comparison.C_fit_relative_difference);
    end

    if opts.RunSyntheticValidation
        synthetic_validation = run_synthetic_validation(M, opts);
    else
        synthetic_validation = struct('ran', false, 'passed', NaN);
    end

    output_files = make_output_paths(output_dir, ranks_requested(end));
    rank_sweep_results = struct();
    rank_sweep_results.model = ...
        'rank-R common bivariate shape with one shared P-by-1 weight vector';
    rank_sweep_results.ranks = ranks_requested(:);
    rank_sweep_results.explained_fraction = explained_fraction;
    rank_sweep_results.explained_percent = explained_percent;
    rank_sweep_results.incremental_explained_fraction = ...
        incremental_explained_fraction;
    rank_sweep_results.relative_residual = relative_residual;
    rank_sweep_results.objective = objective;
    rank_sweep_results.runtime_seconds = runtime_seconds;
    rank_sweep_results.weights_by_rank = weights_by_rank;
    rank_sweep_results.per_direction_relative_residual = ...
        per_direction_relative_residual;
    rank_sweep_results.per_direction_explained_fraction = ...
        per_direction_explained_fraction;
    rank_sweep_results.fits = requested_fits;
    rank_sweep_results.C_tensor = C_tensor;
    rank_sweep_results.D_tensor = D_tensor;
    rank_sweep_results.interaction_meta = interaction_meta;
    rank_sweep_results.m_values = m_values(:);
    rank_sweep_results.n_values = n_values(:);
    rank_sweep_results.T = T;
    rank_sweep_results.transform_info = transform_info;
    rank_sweep_results.options = opts;
    rank_sweep_results.internal_ranks = (1:max_rank).';
    rank_sweep_results.internal_fits = all_rank_fits(:);
    rank_sweep_results.solver_diagnostics = ...
        all_solver_diagnostics(ranks_requested).';
    rank_sweep_results.rank1_comparison = rank1_comparison;
    rank_sweep_results.synthetic_validation = synthetic_validation;
    rank_sweep_results.unique_agents = unique_agents;
    rank_sweep_results.round_dir = round_dir;
    rank_sweep_results.output_dir = output_dir;
    rank_sweep_results.output_files = output_files;
    rank_sweep_results.total_runtime_seconds = toc(total_timer);

    fprintf('[INFO] Requested-rank summary:\n');
    fprintf('%-6s %-14s %-14s %-14s %-13s %-8s %-10s %s\n', ...
        'Rank', 'Objective', 'RelResidual', 'Explained(%)', ...
        'Increment(pp)', 'Start', 'Iterations', 'Status');
    for k = 1:K
        fit_k = requested_fits(k);
        fprintf('%-6d %-14.6e %-14.6e %-14.8f %-13.8f %-8d %-10d %s\n', ...
            fit_k.rank, fit_k.objective, fit_k.relative_residual, ...
            fit_k.explained_percent, ...
            100 * fit_k.incremental_explained_fraction, ...
            fit_k.best_start, fit_k.iterations, fit_k.solver_status);
    end
    fprintf('\n');

    save_rank_sweep_outputs(rank_sweep_results, opts);
    fprintf('[INFO] Total runtime: %.3f seconds\n', ...
        rank_sweep_results.total_runtime_seconds);
    fprintf('[INFO] Rank-sweep analysis completed: %s\n', output_dir);

    diary('off');
    clear diary_cleanup;
end

% Local implementation helpers follow.

function opts = parse_options(varargin)
    p = inputParser;
    addParameter(p, 'Ranks', 1:10, ...
        @(x) isnumeric(x) && isvector(x) && ~isempty(x) && ...
        all(isfinite(x)) && all(x >= 1) && all(x == floor(x)));
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

    % Data options copied from the current rank-1 analysis.
    addParameter(p, 'analysis_start_sec', 10, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
    addParameter(p, 'analysis_duration_sec', 80, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'sample_dt', 0.01, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'signal_column', 'a2', ...
        @(x) ischar(x) || isstring(x));
    addParameter(p, 'normalize_signal', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'tail_percent', 10, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x < 50);
    addParameter(p, 'clip_normalized_signal', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'clip_limit', 0.5, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'use_cache', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_indices', [], ...
        @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'RemoveSelfOnly', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'RemoveConstant', true, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'RemoveOtherOnly', false, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'keep_figures', false, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));

    addParameter(p, 'ExistingRank1Mat', '', ...
        @(x) ischar(x) || isstring(x));
    addParameter(p, 'RunSyntheticValidation', false, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'SyntheticTestOnly', false, ...
        @(x) islogical(x) || (isnumeric(x) && isscalar(x)));

    parse(p, varargin{:});
    opts = p.Results;
    opts.Ranks = double(opts.Ranks(:).');
    opts.NumStarts = double(opts.NumStarts);
    opts.MaxIter = double(opts.MaxIter);
    opts.Tol = double(opts.Tol);
    opts.RandomSeed = double(opts.RandomSeed);
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
    opts.ExistingRank1Mat = char(opts.ExistingRank1Mat);
    opts.RunSyntheticValidation = logical(opts.RunSyntheticValidation);
    opts.SyntheticTestOnly = logical(opts.SyntheticTestOnly);
end

function [C_tensor, interaction_meta, m_values, n_values, unique_agents] = ...
        build_coefficient_tensor(round_dir, M, opts)
% Load and estimate every directed coefficient matrix once.
    pair_infos = list_pair_folders(round_dir);
    if isempty(pair_infos)
        error('No pair folders like 7-8 were found under %s.', round_dir);
    end
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
                'plot_surfaces', false, 'plot_gamma', false, ...
                'save_output', false, 'use_cache', opts.use_cache, ...
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
            error('Unexpected Fourier ordering for pair %s.', info.name);
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
        entry = struct('pair_name', info.name, 'agent_ids', info.agent_ids, ...
            'C1', C1_analysis, 'C2', C2_analysis, ...
            'm_values', m_values, 'n_values', n_values);
        if isempty(pair_data_list)
            pair_data_list = entry;
        else
            pair_data_list(end+1) = entry; %#ok<AGROW>
        end
    end
    if isempty(pair_data_list), error('No valid pair data was loaded.'); end
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
        % C2 already has rows=receiver and columns=sender. Do not transpose.
        C_blocks{end+1} = p_data.C2; %#ok<AGROW>
        meta = struct('pair_name', p_data.pair_name, ...
            'target_id', p_data.agent_ids(2), ...
            'source_id', p_data.agent_ids(1), 'direction', 's2');
        interaction_meta(end+1) = meta; %#ok<AGROW>
    end
    m_values = pair_data_list(1).m_values(:);
    n_values = pair_data_list(1).n_values(:);
    Lm = numel(m_values);
    Ln = numel(n_values);
    P = numel(C_blocks);
    C_tensor = complex(zeros(Lm, Ln, P));
    for p = 1:P
        C_p = C_blocks{p};
        if ~isequal(size(C_p), [Lm, Ln])
            error('Coefficient matrix p=%d has inconsistent size.', p);
        end
        C_tensor(:, :, p) = C_p;
    end
    unique_agents = unique([[interaction_meta.target_id], ...
        [interaction_meta.source_id]]);
    directed_pairs = [[interaction_meta.source_id].', ...
        [interaction_meta.target_id].'];
    expected_count = numel(unique_agents) * (numel(unique_agents) - 1);
    unique_count = size(unique(directed_pairs, 'rows'), 1);
    if P ~= expected_count || unique_count ~= expected_count || ...
            any(directed_pairs(:, 1) == directed_pairs(:, 2))
        warning('Loaded directions do not form a unique complete digraph.');
    end
end

function pair_infos = list_pair_folders(round_dir)
    if isstring(round_dir), round_dir = char(round_dir); end
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
    if remove_self, mask_to_remove = mask_to_remove | is_self_term; end
    if remove_constant, mask_to_remove = mask_to_remove | is_constant_term; end
    if remove_other, mask_to_remove = mask_to_remove | is_other_term; end
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

function [T, info] = build_real_fourier_transform(m_values, M)
    L = numel(m_values);
    if L ~= 2*M + 1 || ~isequal(m_values(:), (-M:M).')
        error('Real-basis transform requires coefficient order -M:M.');
    end
    phi_test = 2*pi*(0:L-1)'/L;
    E = exp(1i * phi_test * m_values(:).');
    Psi = zeros(L, L);
    Psi(:, 1) = 1;
    col = 2;
    for harmonic = 1:M
        Psi(:, col) = sqrt(2) * cos(harmonic * phi_test);
        Psi(:, col + 1) = sqrt(2) * sin(harmonic * phi_test);
        col = col + 2;
    end
    T = E \ Psi;
    info = struct();
    info.phi_test = phi_test;
    info.E = E;
    info.Psi = Psi;
    info.basis_reconstruction_error = norm(E*T - Psi, 'fro');
    info.unitarity_error = norm(T'*T - eye(L), 'fro');
end

function [D_tensor, info] = transform_coefficient_tensor(C_tensor, T, info)
    P = size(C_tensor, 3);
    D_tensor = complex(zeros(size(C_tensor)));
    C_back = complex(zeros(size(C_tensor)));
    for p = 1:P
        D_tensor(:, :, p) = T' * C_tensor(:, :, p) * conj(T);
        C_back(:, :, p) = T * D_tensor(:, :, p) * T.';
    end
    info.tensor_roundtrip_relative_error = ...
        norm(C_tensor(:) - C_back(:)) / max(norm(C_tensor(:)), eps);
    info.D_imaginary_relative_norm = ...
        norm(imag(D_tensor(:))) / max(norm(D_tensor(:)), eps);
end

function reference = load_existing_rank1_reference( ...
        round_dir, M, opts, C_tensor, D_tensor, T)
    reference = struct('available', false, 'reason', '', 'path', '', ...
        'tensor_relative_mismatch', NaN, 'objective', NaN, ...
        'relative_residual', NaN, 'explained_fraction', NaN, ...
        'C_fit', [], 'baseline', struct([]));
    if isempty(opts.ExistingRank1Mat)
        reference.path = fullfile(round_dir, 'low_rank_analysis', ...
            sprintf('M%d', M), 'global_joint_cp_rank1', ...
            'global_joint_cp_rank1_fit.mat');
    else
        reference.path = opts.ExistingRank1Mat;
    end
    if ~exist(reference.path, 'file')
        reference.reason = sprintf('file not found: %s', reference.path);
        return;
    end
    saved = load(reference.path, 'C_tensor', 'C_fit', 'a', 'b', 'w');
    needed = {'C_tensor', 'C_fit', 'a', 'b', 'w'};
    for k = 1:numel(needed)
        if ~isfield(saved, needed{k})
            reference.reason = sprintf('missing variable %s in %s', ...
                needed{k}, reference.path);
            return;
        end
    end
    if ~isequal(size(saved.C_tensor), size(C_tensor))
        reference.reason = 'saved and current tensors have different sizes';
        return;
    end
    reference.tensor_relative_mismatch = ...
        norm(saved.C_tensor(:) - C_tensor(:)) / max(norm(C_tensor(:)), eps);
    if reference.tensor_relative_mismatch > 1e-10
        reference.reason = sprintf('tensor mismatch %.3e exceeds tolerance', ...
            reference.tensor_relative_mismatch);
        return;
    end
    if ~isequal(size(saved.C_fit), size(C_tensor)) || ...
            numel(saved.w) ~= size(C_tensor, 3)
        reference.reason = 'saved rank-1 fit dimensions are incompatible';
        return;
    end
    residual = C_tensor - saved.C_fit;
    reference.objective = sum(abs(residual(:)).^2);
    reference.relative_residual = norm(residual(:)) / norm(C_tensor(:));
    reference.explained_fraction = 1 - reference.relative_residual^2;
    reference.C_fit = saved.C_fit;

    G_reference_complex = saved.a(:) * saved.b(:).';
    G_reference_real_raw = real(T' * G_reference_complex * conj(T));
    [U_ref, S_ref, V_ref] = svd(G_reference_real_raw, 'econ');
    G_reference_real = U_ref(:, 1) * S_ref(1, 1) * V_ref(:, 1)';
    shape_scale = norm(G_reference_real, 'fro');
    if ~isfinite(shape_scale) || shape_scale <= 1e-14
        reference.reason = 'saved rank-1 shape is numerically degenerate';
        return;
    end
    G_reference_real = G_reference_real / shape_scale;
    w_reference = update_common_weights(D_tensor, G_reference_real);
    reference.baseline = struct('G_real', G_reference_real, ...
        'w', w_reference, 'source', 'existing_rank1_mat');
    reference.available = true;
    reference.reason = '';
end

function [all_fits, solver_diagnostics] = fit_rank_sequence( ...
        C_tensor, D_tensor, T, m_values, n_values, phi_grid, ...
        max_rank, opts, rank1_baseline)
    data_scale = norm(C_tensor(:));
    if ~isfinite(data_scale) || data_scale <= 0
        error('Cannot scale a zero or nonfinite C_tensor.');
    end
    C_work = C_tensor / data_scale;
    D_work = D_tensor / data_scale;
    P = size(C_tensor, 3);
    all_fits = struct([]);
    solver_diagnostics = struct([]);

    for rank_value = 1:max_rank
        rank_timer = tic;
        if rank_value == 1
            baseline_raw = rank1_baseline;
            baseline_label = 'existing_rank1';
        else
            baseline_raw = struct('G_real', all_fits(rank_value-1).G_real, ...
                'w', all_fits(rank_value-1).w, ...
                'source', sprintf('rank_%d', rank_value-1));
            baseline_label = sprintf('rank_%d', rank_value-1);
        end

        initial_weights = zeros(P, opts.NumStarts);
        first_random_start = 1;
        if ~isempty(baseline_raw)
            warm_w = baseline_raw.w(:) / data_scale;
            warm_norm = norm(warm_w);
            if isfinite(warm_norm) && warm_norm > 1e-14
                initial_weights(:, 1) = warm_w / warm_norm;
                first_random_start = 2;
            end
        end
        for start_idx = first_random_start:opts.NumStarts
            random_w = randn(P, 1);
            random_norm = norm(random_w);
            if ~isfinite(random_norm) || random_norm <= 1e-14
                random_w = ones(P, 1);
                random_norm = norm(random_w);
            end
            initial_weights(:, start_idx) = random_w / random_norm;
        end

        solver = fit_common_shape_rank_multistart( ...
            C_work, D_work, T, rank_value, initial_weights, ...
            opts.MaxIter, opts.Tol);
        candidate_meta = struct();
        candidate_meta.iterations = solver.iterations;
        candidate_meta.best_start = solver.best_start;
        candidate_meta.converged = solver.converged;
        candidate_meta.solver_status = sprintf( ...
            'optimized_start_%d:%s', solver.best_start, solver.status);
        candidate_meta.objective_history = ...
            solver.objective_history * data_scale^2;
        candidate_meta.start_objectives = ...
            solver.start_objectives * data_scale^2;
        candidate_meta.start_iterations = solver.start_iterations;
        candidate_meta.start_converged = solver.start_converged;
        candidate_meta.start_status = solver.start_status;
        candidate_fit = finalize_common_shape_fit( ...
            C_tensor, D_tensor, T, m_values, n_values, phi_grid, ...
            solver.G_real, data_scale * solver.w, rank_value, candidate_meta);

        baseline_available = ~isempty(baseline_raw);
        if baseline_available
            baseline_meta = struct();
            baseline_meta.iterations = 0;
            baseline_meta.best_start = 0;
            baseline_meta.converged = true;
            baseline_meta.solver_status = sprintf( ...
                'feasible_%s_baseline', baseline_label);
            baseline_meta.objective_history = [];
            baseline_meta.start_objectives = [];
            baseline_meta.start_iterations = [];
            baseline_meta.start_converged = [];
            baseline_meta.start_status = {};
            baseline_fit = finalize_common_shape_fit( ...
                C_tensor, D_tensor, T, m_values, n_values, phi_grid, ...
                baseline_raw.G_real, baseline_raw.w, rank_value, baseline_meta);
        else
            baseline_fit = struct([]);
        end

        if baseline_available && ...
                ~(candidate_fit.objective < baseline_fit.objective)
            selected_fit = baseline_fit;
            selected_fit.solver_status = sprintf( ...
                'fallback_to_%s_baseline; candidate_start_%d_J=%.16e', ...
                baseline_label, solver.best_start, candidate_fit.objective);
            selected_fit.best_start = 0;
            selected_fit.iterations = 0;
            selected_fit.converged = true;
            selected_source = 'baseline';
        else
            selected_fit = candidate_fit;
            selected_source = 'optimized';
        end

        selected_fit.runtime_seconds = toc(rank_timer);
        if rank_value == 1
            selected_fit.incremental_explained_fraction = ...
                selected_fit.explained_fraction;
        else
            selected_fit.incremental_explained_fraction = ...
                selected_fit.explained_fraction - ...
                all_fits(rank_value-1).explained_fraction;
            monotonic_tolerance = 1e-12;
            if selected_fit.incremental_explained_fraction < ...
                    -monotonic_tolerance
                error(['Stored rank-%d fit is worse than the stored rank-%d ', ...
                    'fit; baseline monotonicity failed.'], ...
                    rank_value, rank_value-1);
            end
        end
        if isempty(all_fits)
            all_fits = selected_fit;
        else
            selected_fit = orderfields(selected_fit, all_fits(1));
            all_fits(rank_value) = selected_fit;
        end

        diagnostic = struct();
        diagnostic.rank = rank_value;
        diagnostic.selected_source = selected_source;
        diagnostic.candidate_objective = candidate_fit.objective;
        if baseline_available
            diagnostic.baseline_objective = baseline_fit.objective;
        else
            diagnostic.baseline_objective = NaN;
        end
        diagnostic.best_optimized_start = solver.best_start;
        diagnostic.best_optimized_status = solver.status;
        diagnostic.start_objectives = solver.start_objectives * data_scale^2;
        diagnostic.start_iterations = solver.start_iterations;
        diagnostic.start_converged = solver.start_converged;
        diagnostic.start_status = solver.start_status;
        if isempty(solver_diagnostics)
            solver_diagnostics = diagnostic;
        else
            diagnostic = orderfields(diagnostic, solver_diagnostics(1));
            solver_diagnostics(rank_value) = diagnostic;
        end

        fprintf(['[RANK %2d] explained=%11.8f%%, increment=%11.8f pp, ', ...
            'relres=%.6e, J=%.6e, start=%d, iter=%d\n'], ...
            rank_value, selected_fit.explained_percent, ...
            100 * selected_fit.incremental_explained_fraction, ...
            selected_fit.relative_residual, selected_fit.objective, ...
            selected_fit.best_start, selected_fit.iterations);
        fprintf(['          ||G||F=%.16g, symmetry=%.3e, maxImag(g)=%.3e, ', ...
            'componentCheck=%.3e, status=%s\n'], ...
            norm(selected_fit.G_real, 'fro'), ...
            selected_fit.G_complex_conjugate_symmetry_relative_error, ...
            selected_fit.g_values_max_imaginary, ...
            selected_fit.phase_component_reconstruction_relative_error, ...
            selected_fit.solver_status);
    end
    fprintf('\n');
end

function solver = fit_common_shape_rank_multistart( ...
        C_work, D_work, T, rank_value, initial_weights, max_iter, tol)
    num_starts = size(initial_weights, 2);
    start_objectives = inf(num_starts, 1);
    start_iterations = zeros(num_starts, 1);
    start_converged = false(num_starts, 1);
    start_status = repmat({'not run'}, num_starts, 1);
    best_objective = inf;
    best = struct([]);

    for start_idx = 1:num_starts
        one = run_one_common_shape_start( ...
            C_work, D_work, T, rank_value, ...
            initial_weights(:, start_idx), max_iter, tol);
        start_objectives(start_idx) = one.objective;
        start_iterations(start_idx) = one.iterations;
        start_converged(start_idx) = one.converged;
        start_status{start_idx} = one.status;
        if one.valid && one.objective < best_objective
            best_objective = one.objective;
            best = one;
            best.best_start = start_idx;
        end
    end
    if isempty(best)
        error('All starts failed for common-shape rank %d.', rank_value);
    end
    solver = best;
    solver.start_objectives = start_objectives;
    solver.start_iterations = start_iterations;
    solver.start_converged = start_converged;
    solver.start_status = start_status;
end

function result = run_one_common_shape_start( ...
        C_work, D_work, T, rank_value, w_initial, max_iter, tol)
    P = size(D_work, 3);
    L = size(D_work, 1);
    result = struct('valid', false, 'G_real', [], 'w', [], ...
        'objective', Inf, 'iterations', 0, 'converged', false, ...
        'status', 'not started', 'objective_history', []);
    w = real(w_initial(:));
    w_norm = norm(w);
    if numel(w) ~= P || ~isfinite(w_norm) || w_norm <= 1e-14
        result.status = 'invalid_initial_weight';
        return;
    end
    w = w / w_norm;
    accepted_G = [];
    accepted_w = [];
    accepted_objective = Inf;
    history = nan(max_iter, 1);
    accepted_iterations = 0;
    converged = false;
    status = 'maximum_iterations_reached';
    X_work = real(D_work);

    for iter = 1:max_iter
        denom = sum(w.^2);
        if ~isfinite(denom) || denom <= 1e-28
            status = 'zero_or_nonfinite_weight_denominator';
            break;
        end
        H_raw = zeros(L, L);
        for p = 1:P
            H_raw = H_raw + w(p) * X_work(:, :, p);
        end
        H_raw = H_raw / denom;
        if any(~isfinite(H_raw(:)))
            status = 'nonfinite_shape_update';
            break;
        end
        [U_raw, S_raw, V_raw] = svd(H_raw, 'econ');
        r_eff = min(rank_value, min(size(H_raw)));
        H_rankR = U_raw(:, 1:r_eff) * ...
            S_raw(1:r_eff, 1:r_eff) * V_raw(:, 1:r_eff)';
        shape_scale = norm(H_rankR, 'fro');
        if ~isfinite(shape_scale) || shape_scale <= 1e-14
            status = 'zero_or_nonfinite_rank_truncated_shape';
            break;
        end
        G_candidate = H_rankR / shape_scale;
        w_candidate = update_common_weights(D_work, G_candidate);
        if numel(w_candidate) ~= P || any(~isfinite(w_candidate)) || ...
                norm(w_candidate) <= 1e-14
            status = 'zero_or_nonfinite_weight_update';
            break;
        end
        G_complex_candidate = T * G_candidate * T.';
        C_fit_candidate = reconstruct_common_fit( ...
            G_complex_candidate, w_candidate, size(C_work));
        residual_candidate = C_work - C_fit_candidate;
        objective_candidate = sum(abs(residual_candidate(:)).^2);
        if ~isfinite(objective_candidate)
            status = 'nonfinite_objective';
            break;
        end

        if accepted_iterations > 0 && ...
                objective_candidate > accepted_objective
            objective_increase = objective_candidate - accepted_objective;
            numerical_floor = 128 * eps(max(1, accepted_objective));
            if objective_increase <= numerical_floor
                converged = true;
                status = 'numerical_objective_floor';
            else
                converged = false;
                status = sprintf('objective_increase_rejected:%.3e', ...
                    objective_increase);
            end
            break;
        end

        if accepted_iterations > 0
            relative_change = abs(accepted_objective - objective_candidate) / ...
                max([abs(accepted_objective), abs(objective_candidate), eps]);
        else
            relative_change = Inf;
        end
        accepted_G = G_candidate;
        accepted_w = w_candidate;
        accepted_objective = objective_candidate;
        accepted_iterations = iter;
        history(iter) = objective_candidate;
        w = w_candidate;
        if iter > 1 && relative_change <= tol
            converged = true;
            status = 'converged';
            break;
        end
    end

    if accepted_iterations == 0
        result.status = status;
        return;
    end
    result.valid = true;
    result.G_real = accepted_G;
    result.w = accepted_w;
    result.objective = accepted_objective;
    result.iterations = accepted_iterations;
    result.converged = converged;
    result.status = status;
    result.objective_history = history(1:accepted_iterations);
end

function w = update_common_weights(D_tensor, G_real)
    denominator = sum(abs(G_real(:)).^2);
    if ~isfinite(denominator) || denominator <= 1e-28
        error('Cannot update weights from a zero or nonfinite common shape.');
    end
    P = size(D_tensor, 3);
    w = zeros(P, 1);
    for p = 1:P
        Dp = D_tensor(:, :, p);
        w(p) = real(sum(conj(G_real(:)) .* Dp(:))) / denominator;
    end
end

function C_fit = reconstruct_common_fit(G_complex, w, tensor_size)
    P = tensor_size(3);
    C_fit = complex(zeros(tensor_size));
    for p = 1:P
        C_fit(:, :, p) = w(p) * G_complex;
    end
end

function fit = finalize_common_shape_fit( ...
        C_tensor, ~, T, m_values, n_values, phi_grid, ...
        G_real, w, rank_value, solver_meta)
    P = size(C_tensor, 3);
    L = size(C_tensor, 1);
    if rank_value > L
        error('Rank %d exceeds common-shape matrix dimension %d.', rank_value, L);
    end
    G_real = real(G_real);
    shape_scale = norm(G_real, 'fro');
    if ~isfinite(shape_scale) || shape_scale <= 1e-14
        error('Cannot finalize a zero or nonfinite common shape.');
    end
    G_real = G_real / shape_scale;
    w = real(w(:)) * shape_scale;
    if ~isequal(size(w), [P, 1]) || any(~isfinite(w))
        error('Final weight vector must be finite, real, and P-by-1.');
    end
    G_complex = T * G_real * T.';
    receiver_basis = exp(1i * phi_grid * m_values(:).');
    sender_basis = exp(1i * phi_grid * n_values(:).');
    g_values = receiver_basis * G_complex * sender_basis.';
    [~, peak_index] = max(abs(g_values(:)));
    if real(g_values(peak_index)) < 0
        G_real = -G_real;
        G_complex = -G_complex;
        w = -w;
        g_values = -g_values;
    end

    C_fit = reconstruct_common_fit(G_complex, w, size(C_tensor));
    [objective, relative_residual, explained_fraction, ...
        per_direction_relative_residual, ...
        per_direction_explained_fraction] = ...
        compute_common_shape_metrics(C_tensor, C_fit);

    [U, S, V] = svd(G_real, 'econ');
    for component = 1:rank_value
        [~, sign_index] = max(abs(U(:, component)));
        if U(sign_index, component) < 0
            U(:, component) = -U(:, component);
            V(:, component) = -V(:, component);
        end
    end
    sigma = diag(S(1:rank_value, 1:rank_value));
    A_real = U(:, 1:rank_value);
    B_real = V(:, 1:rank_value);
    A_complex = T * A_real;
    B_complex = T * B_real;
    G_complex_check = A_complex * diag(sigma) * B_complex.';
    G_complex_check_relative_error = ...
        norm(G_complex(:) - G_complex_check(:)) / ...
        max(norm(G_complex(:)), eps);

    a_values = receiver_basis * A_complex;
    b_values = sender_basis * B_complex;
    g_values_component_check = ...
        a_values * diag(sigma) * b_values.';
    phase_component_reconstruction_relative_error = ...
        norm(g_values(:) - g_values_component_check(:)) / ...
        max(norm(g_values(:)), eps);
    G_symmetry_partner = conj(flip(flip(G_complex, 1), 2));
    G_complex_conjugate_symmetry_relative_error = ...
        norm(G_complex(:) - G_symmetry_partner(:)) / ...
        max(norm(G_complex(:)), eps);
    g_values_max_imaginary = max(abs(imag(g_values(:))));

    if G_complex_conjugate_symmetry_relative_error > 1e-10
        warning('Rank %d G_complex conjugate symmetry error is %.3e.', ...
            rank_value, G_complex_conjugate_symmetry_relative_error);
    end
    real_wave_tolerance = 1e-10 * max(1, max(abs(real(g_values(:)))));
    if g_values_max_imaginary > real_wave_tolerance
        warning('Rank %d common phase function max imaginary part is %.3e.', ...
            rank_value, g_values_max_imaginary);
    end
    if G_complex_check_relative_error > 1e-10 || ...
            phase_component_reconstruction_relative_error > 1e-10
        warning('Rank %d separable-component reconstruction check failed.', ...
            rank_value);
    end

    residual_check = C_tensor - C_fit;
    objective_check = sum(abs(residual_check(:)).^2);
    objective_consistency_absolute_error = abs(objective - objective_check);
    common_weight_model_relative_error = 0;
    for p = 1:P
        model_slice = w(p) * G_complex;
        slice_error = norm(C_fit(:, :, p) - model_slice, 'fro') / ...
            max(norm(model_slice, 'fro'), eps);
        common_weight_model_relative_error = max( ...
            common_weight_model_relative_error, slice_error);
    end

    fit = struct();
    fit.rank = rank_value;
    fit.G_real = G_real;
    fit.G_complex = G_complex;
    fit.w = w;
    fit.U = U;
    fit.S = S;
    fit.V = V;
    fit.sigma = sigma;
    fit.A_real = A_real;
    fit.B_real = B_real;
    fit.A_complex = A_complex;
    fit.B_complex = B_complex;
    fit.C_fit = C_fit;
    fit.objective = objective;
    fit.relative_residual = relative_residual;
    fit.explained_fraction = explained_fraction;
    fit.explained_percent = 100 * explained_fraction;
    fit.incremental_explained_fraction = NaN;
    fit.per_direction_relative_residual = ...
        per_direction_relative_residual;
    fit.per_direction_explained_fraction = ...
        per_direction_explained_fraction;
    fit.iterations = solver_meta.iterations;
    fit.best_start = solver_meta.best_start;
    fit.converged = solver_meta.converged;
    fit.runtime_seconds = NaN;
    fit.solver_status = solver_meta.solver_status;
    fit.objective_history = solver_meta.objective_history;
    fit.start_objectives = solver_meta.start_objectives;
    fit.start_iterations = solver_meta.start_iterations;
    fit.start_converged = solver_meta.start_converged;
    fit.start_status = solver_meta.start_status;
    fit.G_complex_check_relative_error = ...
        G_complex_check_relative_error;
    fit.phase_component_reconstruction_relative_error = ...
        phase_component_reconstruction_relative_error;
    fit.G_complex_conjugate_symmetry_relative_error = ...
        G_complex_conjugate_symmetry_relative_error;
    fit.g_values_max_imaginary = g_values_max_imaginary;
    fit.objective_consistency_absolute_error = ...
        objective_consistency_absolute_error;
    fit.common_weight_model_relative_error = ...
        common_weight_model_relative_error;
    fit.G_real_frobenius_norm = norm(G_real, 'fro');
    fit.G_complex_frobenius_norm = norm(G_complex, 'fro');
end

function [objective, relative_residual, explained_fraction, ...
        per_relative, per_explained] = ...
        compute_common_shape_metrics(C_tensor, C_fit)
    residual = C_tensor - C_fit;
    data_norm = norm(C_tensor(:));
    residual_norm = norm(residual(:));
    objective = sum(abs(residual(:)).^2);
    if data_norm == 0
        error('Cannot define explained fraction for a zero tensor.');
    end
    relative_residual = residual_norm / data_norm;
    explained_fraction = 1 - relative_residual^2;
    P = size(C_tensor, 3);
    per_relative = zeros(P, 1);
    per_explained = zeros(P, 1);
    for p = 1:P
        C_p = C_tensor(:, :, p);
        R_p = residual(:, :, p);
        C_norm = norm(C_p(:));
        R_norm = norm(R_p(:));
        if C_norm == 0
            if R_norm == 0
                per_relative(p) = 0;
                per_explained(p) = 1;
            else
                per_relative(p) = Inf;
                per_explained(p) = -Inf;
            end
        else
            per_relative(p) = R_norm / C_norm;
            per_explained(p) = 1 - per_relative(p)^2;
        end
    end
end

function comparison = compare_rank1_results(reference, sweep_fit, C_tensor)
    comparison = struct('available', false, 'reason', '', ...
        'reference_objective', NaN, 'sweep_objective', NaN, ...
        'objective_absolute_difference', NaN, ...
        'objective_relative_difference', NaN, ...
        'reference_explained_fraction', NaN, ...
        'sweep_explained_fraction', NaN, ...
        'explained_fraction_difference', NaN, ...
        'C_fit_relative_difference', NaN);
    if ~reference.available
        comparison.reason = reference.reason;
        return;
    end
    comparison.available = true;
    comparison.reference_objective = reference.objective;
    comparison.sweep_objective = sweep_fit.objective;
    comparison.objective_absolute_difference = ...
        abs(sweep_fit.objective - reference.objective);
    comparison.objective_relative_difference = ...
        comparison.objective_absolute_difference / ...
        max(reference.objective, eps);
    comparison.reference_explained_fraction = ...
        reference.explained_fraction;
    comparison.sweep_explained_fraction = sweep_fit.explained_fraction;
    comparison.explained_fraction_difference = ...
        sweep_fit.explained_fraction - reference.explained_fraction;
    comparison.C_fit_relative_difference = ...
        norm(sweep_fit.C_fit(:) - reference.C_fit(:)) / ...
        max(norm(reference.C_fit(:)), eps);
    comparison.reason = '';
    allowed = 1e-10 * max(1, sum(abs(C_tensor(:)).^2));
    if sweep_fit.objective > reference.objective + allowed
        warning('New rank-1 fit is worse than the matching existing fit.');
    end
end

function validation = run_synthetic_validation(M, opts)
% Rank-3 real-shape test plus same-tensor legacy rank-1 comparison.
    L = 2*M + 1;
    if L < 5
        error('Synthetic ranks 1:5 require M>=2.');
    end
    P = 12;
    m_values = (-M:M).';
    n_values = m_values;
    [T, transform_info] = build_real_fourier_transform(m_values, M);
    phi_grid = linspace(0, 2*pi, 512).';
    old_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(old_rng_state));
    rng(opts.RandomSeed, 'twister');

    [Q_left, ~] = qr(randn(L, 3), 0);
    [Q_right, ~] = qr(randn(L, 3), 0);
    G_true_real = Q_left * diag([3, 2, 1]) * Q_right';
    G_true_real = G_true_real / norm(G_true_real, 'fro');
    w_true = linspace(-1.1, 0.9, P).' + 0.05 * randn(P, 1);
    D_true = zeros(L, L, P);
    C_true = complex(zeros(L, L, P));
    for p = 1:P
        D_true(:, :, p) = w_true(p) * G_true_real;
        C_true(:, :, p) = T * D_true(:, :, p) * T.';
    end
    test_opts = opts;
    test_opts.Ranks = 1:5;
    [fits, ~] = fit_rank_sequence(C_true, D_true, T, ...
        m_values, n_values, phi_grid, 5, test_opts, struct([]));
    observed_explained = reshape([fits.explained_fraction], [], 1);
    observed_relative_residual = reshape([fits.relative_residual], [], 1);
    expected_explained = [9/14; 13/14; 1; 1; 1];
    monotonic = all(diff(observed_explained) >= -1e-12);
    exact_from_rank3 = all(observed_relative_residual(3:5) < 1e-10);

    all_weight_shapes_valid = true;
    all_common_weight_models_valid = true;
    all_symmetry_valid = true;
    all_phase_real = true;
    all_objectives_consistent = true;
    for rank_value = 1:5
        fit_r = fits(rank_value);
        all_weight_shapes_valid = all_weight_shapes_valid && ...
            isequal(size(fit_r.w), [P, 1]) && isreal(fit_r.w);
        all_common_weight_models_valid = all_common_weight_models_valid && ...
            fit_r.common_weight_model_relative_error < 1e-12;
        all_symmetry_valid = all_symmetry_valid && ...
            fit_r.G_complex_conjugate_symmetry_relative_error < 1e-10;
        all_phase_real = all_phase_real && ...
            fit_r.g_values_max_imaginary < 1e-10;
        direct_residual = C_true - fit_r.C_fit;
        direct_objective = sum(abs(direct_residual(:)).^2);
        all_objectives_consistent = all_objectives_consistent && ...
            abs(direct_objective - fit_r.objective) <= ...
            1e-12 * max(1, direct_objective);
    end

    % The existing rank-1 solver's synthetic path performs no file I/O.
    legacy = global_joint_svd_analysis_joint_cp_rank1([], M, ...
        'SyntheticTestOnly', true, 'NumStarts', opts.NumStarts, ...
        'MaxIter', opts.MaxIter, 'Tol', opts.Tol, ...
        'RandomSeed', opts.RandomSeed, 'DisplayFullArrays', false);
    legacy_C = legacy.C_tensor;
    [legacy_D, ~] = transform_coefficient_tensor( ...
        legacy_C, T, transform_info);
    legacy_G_complex = legacy.a(:) * legacy.b(:).';
    legacy_G_real_raw = real(T' * legacy_G_complex * conj(T));
    [Ul, Sl, Vl] = svd(legacy_G_real_raw, 'econ');
    legacy_G_real = Ul(:, 1) * Sl(1, 1) * Vl(:, 1)';
    legacy_G_real = legacy_G_real / norm(legacy_G_real, 'fro');
    legacy_w = update_common_weights(legacy_D, legacy_G_real);
    legacy_baseline = struct('G_real', legacy_G_real, ...
        'w', legacy_w, 'source', 'legacy_synthetic_rank1');
    rng(opts.RandomSeed, 'twister');
    [new_rank1_fit, ~] = fit_rank_sequence( ...
        legacy_C, legacy_D, T, m_values, n_values, phi_grid, ...
        1, test_opts, legacy_baseline);
    legacy_residual = legacy_C - legacy.C_fit;
    legacy_objective = sum(abs(legacy_residual(:)).^2);
    legacy_relative_residual = norm(legacy_residual(:)) / norm(legacy_C(:));
    legacy_explained = 1 - legacy_relative_residual^2;
    rank1_objective_difference = abs( ...
        new_rank1_fit(1).objective - legacy_objective);
    rank1_objective_tolerance = 1e-12 * max([ ...
        sum(abs(legacy_C(:)).^2), ...
        abs(new_rank1_fit(1).objective), abs(legacy_objective), eps]);
    rank1_consistent = new_rank1_fit(1).relative_residual < 1e-10 && ...
        legacy_relative_residual < 1e-10 && ...
        abs(new_rank1_fit(1).explained_fraction - legacy_explained) < 1e-12 && ...
        rank1_objective_difference <= rank1_objective_tolerance;

    validation = struct();
    validation.ran = true;
    validation.passed = monotonic && exact_from_rank3 && ...
        all_weight_shapes_valid && all_common_weight_models_valid && ...
        all_symmetry_valid && all_phase_real && ...
        all_objectives_consistent && rank1_consistent;
    validation.expected_explained_fraction = expected_explained;
    validation.observed_explained_fraction = observed_explained;
    validation.observed_relative_residual = observed_relative_residual;
    validation.monotonic = monotonic;
    validation.exact_from_rank3 = exact_from_rank3;
    validation.all_weight_shapes_valid = all_weight_shapes_valid;
    validation.all_common_weight_models_valid = ...
        all_common_weight_models_valid;
    validation.all_symmetry_valid = all_symmetry_valid;
    validation.all_phase_functions_real = all_phase_real;
    validation.all_objectives_consistent = all_objectives_consistent;
    validation.rank1_legacy_consistent = rank1_consistent;
    validation.legacy_rank1_objective = legacy_objective;
    validation.new_rank1_objective = new_rank1_fit(1).objective;
    validation.rank1_objective_difference = rank1_objective_difference;
    validation.rank1_objective_tolerance = rank1_objective_tolerance;
    validation.transform_info = transform_info;
    validation.fits = fits(:);

    fprintf('[SYNTHETIC] Rank-3 common-shape validation:\n');
    for rank_value = 1:5
        fprintf('  R=%d: explained=%.12f%%, relres=%.3e\n', ...
            rank_value, 100*observed_explained(rank_value), ...
            observed_relative_residual(rank_value));
    end
    fprintf('  Legacy/new rank-1 objectives: %.3e / %.3e\n', ...
        legacy_objective, new_rank1_fit(1).objective);
    fprintf('  Validation passed: %s\n', mat2str(validation.passed));
    if ~validation.passed
        error('Synthetic common-shape rank-sweep validation failed.');
    end
    clear rng_cleanup;
end

function paths = make_output_paths(output_dir, maximum_rank)
    paths = struct();
    paths.summary_csv = fullfile(output_dir, ...
        'global_joint_common_shape_rank_sweep_summary.csv');
    paths.weights_csv = fullfile(output_dir, ...
        'global_joint_common_shape_weights_by_rank.csv');
    paths.per_direction_csv = fullfile(output_dir, ...
        'global_joint_common_shape_per_direction_metrics_by_rank.csv');
    paths.results_mat = fullfile(output_dir, ...
        'global_joint_common_shape_rank_sweep_results.mat');
    paths.explained_png = fullfile(output_dir, ...
        'global_joint_common_shape_explained_vs_rank.png');
    paths.explained_fig = fullfile(output_dir, ...
        'global_joint_common_shape_explained_vs_rank.fig');
    paths.weights_png = fullfile(output_dir, ...
        'global_joint_common_shape_weights_vs_rank.png');
    paths.weights_fig = fullfile(output_dir, ...
        'global_joint_common_shape_weights_vs_rank.fig');
    paths.per_direction_png = fullfile(output_dir, ...
        'global_joint_common_shape_per_direction_explained_vs_rank.png');
    paths.per_direction_fig = fullfile(output_dir, ...
        'global_joint_common_shape_per_direction_explained_vs_rank.fig');
    paths.influence_graph_png = fullfile(output_dir, sprintf( ...
        'global_joint_common_shape_rank%d_influence_graph.png', maximum_rank));
    paths.influence_graph_fig = fullfile(output_dir, sprintf( ...
        'global_joint_common_shape_rank%d_influence_graph.fig', maximum_rank));
end

function save_rank_sweep_outputs(results, opts)
    ranks = results.ranks(:);
    fits = results.fits(:);
    K = numel(ranks);
    P = numel(results.interaction_meta);
    paths = results.output_files;

    best_start = reshape([fits.best_start], K, 1);
    iterations = reshape([fits.iterations], K, 1);
    converged = reshape([fits.converged], K, 1);
    solver_status = {fits.solver_status}.';
    summary_table = table(ranks, results.objective, ...
        results.relative_residual, results.explained_fraction, ...
        results.explained_percent, ...
        results.incremental_explained_fraction, ...
        100*results.incremental_explained_fraction, ...
        results.runtime_seconds, best_start, iterations, converged, ...
        solver_status, 'VariableNames', { ...
        'Rank', 'Objective', 'RelativeResidual', 'ExplainedFraction', ...
        'ExplainedPercent', 'IncrementalExplainedFraction', ...
        'IncrementalExplainedPercentagePoints', 'RuntimeSeconds', ...
        'BestStart', 'Iterations', 'Converged', 'SolverStatus'});
    writetable(summary_table, paths.summary_csv);

    total_rows = P * K;
    Rank = zeros(total_rows, 1);
    DirectionIndex = zeros(total_rows, 1);
    PairName = cell(total_rows, 1);
    SourceID = zeros(total_rows, 1);
    TargetID = zeros(total_rows, 1);
    DirectionLabel = cell(total_rows, 1);
    Weight = zeros(total_rows, 1);
    RelativeResidual = zeros(total_rows, 1);
    ExplainedFraction = zeros(total_rows, 1);
    ExplainedPercent = zeros(total_rows, 1);
    row = 0;
    for k = 1:K
        for p = 1:P
            row = row + 1;
            meta = results.interaction_meta(p);
            Rank(row) = ranks(k);
            DirectionIndex(row) = p;
            PairName{row} = meta.pair_name;
            SourceID(row) = meta.source_id;
            TargetID(row) = meta.target_id;
            DirectionLabel{row} = sprintf('%d -> %d', ...
                meta.source_id, meta.target_id);
            Weight(row) = results.weights_by_rank(p, k);
            RelativeResidual(row) = ...
                results.per_direction_relative_residual(p, k);
            ExplainedFraction(row) = ...
                results.per_direction_explained_fraction(p, k);
            ExplainedPercent(row) = 100 * ExplainedFraction(row);
        end
    end
    weights_table = table(Rank, DirectionIndex, PairName, SourceID, ...
        TargetID, DirectionLabel, Weight);
    writetable(weights_table, paths.weights_csv);
    per_direction_table = table(Rank, DirectionIndex, PairName, ...
        SourceID, TargetID, DirectionLabel, RelativeResidual, ...
        ExplainedFraction, ExplainedPercent);
    writetable(per_direction_table, paths.per_direction_csv);

    if opts.keep_figures
        figure_visibility = 'on';
    else
        figure_visibility = 'off';
    end

    fig_explained = figure('Color', 'w', 'Visible', figure_visibility, ...
        'Position', [100, 100, 850, 720], ...
        'Name', 'Common Shape Explained Energy vs Rank');
    layout = tiledlayout(fig_explained, 2, 1, ...
        'TileSpacing', 'compact', 'Padding', 'compact');
    title(layout, ['Single-Weight Common Interaction-Function ', ...
        'Matrix-Rank Sweep'], 'FontWeight', 'bold');
    ax_top = nexttile(layout, 1);
    plot(ax_top, ranks, results.explained_percent, '-o', ...
        'LineWidth', 2.2, 'MarkerSize', 7, ...
        'Color', [0, 0.447, 0.741], 'MarkerFaceColor', [0, 0.447, 0.741]);
    grid(ax_top, 'on');
    box(ax_top, 'on');
    xlabel(ax_top, 'Common interaction-function rank R');
    ylabel(ax_top, 'Explained energy (%)');
    xticks(ax_top, ranks);
    for k = 1:K
        text(ax_top, ranks(k), results.explained_percent(k), ...
            sprintf(' %.2f%%', results.explained_percent(k)), ...
            'VerticalAlignment', 'bottom', 'FontSize', 9);
    end

    ax_bottom = nexttile(layout, 2);
    bar(ax_bottom, ranks, 100*results.incremental_explained_fraction, ...
        'FaceColor', [0.466, 0.674, 0.188]);
    grid(ax_bottom, 'on');
    box(ax_bottom, 'on');
    xlabel(ax_bottom, 'Common interaction-function rank R');
    ylabel(ax_bottom, ...
        'Incremental explained energy (percentage points)');
    xticks(ax_bottom, ranks);
    saveas(fig_explained, paths.explained_png);
    savefig(fig_explained, paths.explained_fig);

    direction_labels = cell(P, 1);
    for p = 1:P
        direction_labels{p} = sprintf('%d -> %d', ...
            results.interaction_meta(p).source_id, ...
            results.interaction_meta(p).target_id);
    end
    fig_weights = figure('Color', 'w', 'Visible', figure_visibility, ...
        'Position', [120, 120, 900, 620], ...
        'Name', 'Shared Direction Weights vs Rank');
    ax_weights = axes('Parent', fig_weights);
    imagesc(ax_weights, ranks, 1:P, results.weights_by_rank);
    set(ax_weights, 'YTick', 1:P, 'YTickLabel', direction_labels);
    xticks(ax_weights, ranks);
    xlabel(ax_weights, 'Common interaction-function rank R');
    ylabel(ax_weights, 'Directed interaction (source -> target)');
    title(ax_weights, ['Single shared w_p at each rank; ', ...
        '||G_R||_F = 1']);
    weight_limit = max(abs(results.weights_by_rank(:)));
    if ~isfinite(weight_limit) || weight_limit == 0, weight_limit = 1; end
    clim(ax_weights, [-weight_limit, weight_limit]);
    colormap(ax_weights, make_diverging_colormap(256));
    cb_weights = colorbar(ax_weights);
    cb_weights.Label.String = 'w_p';
    saveas(fig_weights, paths.weights_png);
    savefig(fig_weights, paths.weights_fig);

    fig_per_direction = figure('Color', 'w', ...
        'Visible', figure_visibility, 'Position', [140, 140, 900, 620], ...
        'Name', 'Per-Direction Explained Energy vs Rank');
    ax_per_direction = axes('Parent', fig_per_direction);
    imagesc(ax_per_direction, ranks, 1:P, ...
        100*results.per_direction_explained_fraction);
    set(ax_per_direction, 'YTick', 1:P, ...
        'YTickLabel', direction_labels);
    xticks(ax_per_direction, ranks);
    xlabel(ax_per_direction, 'Common interaction-function rank R');
    ylabel(ax_per_direction, 'Directed interaction (source -> target)');
    title(ax_per_direction, ['Per-direction explained energy (%); ', ...
        'overall result uses total tensor energy, not a simple mean']);
    colormap(ax_per_direction, parula(256));
    cb_per_direction = colorbar(ax_per_direction);
    cb_per_direction.Label.String = 'Explained energy (%)';
    saveas(fig_per_direction, paths.per_direction_png);
    savefig(fig_per_direction, paths.per_direction_fig);

    max_rank_fit = fits(end);
    fig_graph = create_influence_graph(max_rank_fit.w, ...
        results.interaction_meta, results.unique_agents, ...
        max_rank_fit.rank, figure_visibility, results.round_dir);
    saveas(fig_graph, paths.influence_graph_png);
    savefig(fig_graph, paths.influence_graph_fig);

    rank_sweep_results = results;
    save(paths.results_mat, 'rank_sweep_results', '-v7.3');

    fprintf('[INFO] Saved rank summary CSV: %s\n', paths.summary_csv);
    fprintf('[INFO] Saved weights CSV: %s\n', paths.weights_csv);
    fprintf('[INFO] Saved per-direction CSV: %s\n', ...
        paths.per_direction_csv);
    fprintf('[INFO] Saved explained PNG/FIG: %s ; %s\n', ...
        paths.explained_png, paths.explained_fig);
    fprintf('[INFO] Saved weights PNG/FIG: %s ; %s\n', ...
        paths.weights_png, paths.weights_fig);
    fprintf('[INFO] Saved per-direction PNG/FIG: %s ; %s\n', ...
        paths.per_direction_png, paths.per_direction_fig);
    fprintf('[INFO] Saved maximum-rank graph PNG/FIG: %s ; %s\n', ...
        paths.influence_graph_png, paths.influence_graph_fig);
    fprintf('[INFO] Saved complete MAT results: %s\n', paths.results_mat);

    if ~opts.keep_figures
        close(fig_explained);
        close(fig_weights);
        close(fig_per_direction);
        close(fig_graph);
    end
end

function cmap = make_diverging_colormap(n_colors)
    half_low = floor(n_colors/2);
    half_high = n_colors - half_low;
    blue = [0.0, 0.0, 0.85];
    center = [0.95, 0.95, 0.95];
    red = [0.85, 0.0, 0.0];
    low = [linspace(blue(1), center(1), half_low)', ...
        linspace(blue(2), center(2), half_low)', ...
        linspace(blue(3), center(3), half_low)'];
    high = [linspace(center(1), red(1), half_high)', ...
        linspace(center(2), red(2), half_high)', ...
        linspace(center(3), red(3), half_high)'];
    cmap = [low; high];
end

function fig = create_influence_graph( ...
        weights, interaction_meta, unique_agents, rank_value, visibility, round_dir)
    if nargin < 6
        round_dir = '';
    end
    clim_limit = 0.06;
    linewidth_limit = 0.06;
    P = numel(interaction_meta);
    source_ids = zeros(P, 1);
    target_ids = zeros(P, 1);
    for p = 1:P
        source_ids(p) = interaction_meta(p).source_id;
        target_ids(p) = interaction_meta(p).target_id;
    end
    node_names = arrayfun(@(id) sprintf('%d', id), unique_agents(:), ...
        'UniformOutput', false);
    source_names = arrayfun(@(id) sprintf('%d', id), source_ids, ...
        'UniformOutput', false);
    target_names = arrayfun(@(id) sprintf('%d', id), target_ids, ...
        'UniformOutput', false);
    graph_data = digraph(source_names, target_names, weights(:), node_names);
    [x_data, y_data] = get_preferred_node_positions(graph_data, round_dir);
    fig = figure('Color', 'w', 'Visible', visibility, ...
        'Position', [100, 100, 520, 450], ...
        'Name', sprintf('Common Shape Rank-%d Influence Graph', rank_value));
    ax = axes('Parent', fig);
    graph_plot = plot(ax, graph_data, 'XData', x_data, 'YData', y_data, ...
        'NodeLabel', {}, 'ArrowSize', 16, 'ArrowPosition', 0.75, ...
        'MarkerSize', 8, 'NodeColor', [0.15, 0.15, 0.15], ...
        'EdgeColor', [0.0, 0.4470, 0.7410]);
    axis(ax, 'equal');
    margin = 0.45;
    rx = max(x_data, [], 'omitnan') - min(x_data, [], 'omitnan') + 2*margin;
    ry = max(y_data, [], 'omitnan') - min(y_data, [], 'omitnan') + 2*margin;
    max_r = max(rx, ry);
    cx = (min(x_data, [], 'omitnan') + max(x_data, [], 'omitnan')) / 2;
    cy = (min(y_data, [], 'omitnan') + max(y_data, [], 'omitnan')) / 2;
    xlim(ax, [cx - max_r/2, cx + max_r/2]);
    ylim(ax, [cy - max_r/2, cy + max_r/2]);
    title(ax, sprintf(['Rank-%d Common-Shape Directed Influence ', ...
        'Graph (one w_p per direction)'], rank_value));
    if numedges(graph_data) > 0
        graph_plot.LineWidth = scale_edge_width( ...
            abs(graph_data.Edges.Weight), linewidth_limit);
        graph_plot.EdgeCData = graph_data.Edges.Weight;
        graph_plot.EdgeColor = 'flat';
        clim(ax, [-clim_limit, clim_limit]);
        colormap(ax, make_diverging_colormap(256));
        cb = colorbar(ax);
        cb.Label.String = 'w_p (Coupling Strength)';
    end
    draw_edge_labels(ax, graph_data, x_data, y_data);
    draw_node_labels(ax, graph_data, x_data, y_data);
end

function [x_data, y_data] = get_preferred_node_positions(G, round_dir)
    node_names = G.Nodes.Name;
    if isstring(node_names), node_names = cellstr(node_names); end
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
        angles = linspace(0, 2*pi, nnz(missing)+1);
        cx = mean(x_data(~missing), 'omitnan');
        cy = mean(y_data(~missing), 'omitnan');
        if ~isfinite(cx), cx = 2.0; end
        if ~isfinite(cy), cy = 2.0; end
        x_data(missing) = cx + 1.0*cos(angles(1:end-1));
        y_data(missing) = cy + 1.0*sin(angles(1:end-1));
    end
end

function widths = scale_edge_width(weights, max_limit)
    clamped = min(max(double(weights(:)), 0), max_limit);
    widths = 1.0 + 5.0 * clamped / max_limit;
end

function draw_edge_labels(ax, G, x_data, y_data)
    offset = 0.08;
    for edge = 1:numedges(G)
        source = G.Edges.EndNodes{edge, 1};
        target = G.Edges.EndNodes{edge, 2};
        source_idx = find(strcmp(G.Nodes.Name, source), 1);
        target_idx = find(strcmp(G.Nodes.Name, target), 1);
        dx = x_data(target_idx) - x_data(source_idx);
        dy = y_data(target_idx) - y_data(source_idx);
        edge_length = hypot(dx, dy);
        if edge_length > 0
            x_label = x_data(source_idx) + 0.35*dx + offset*dy/edge_length;
            y_label = y_data(source_idx) + 0.35*dy - offset*dx/edge_length;
        else
            x_label = x_data(source_idx);
            y_label = y_data(source_idx) + offset;
        end
        text(ax, x_label, y_label, sprintf('%.3g', G.Edges.Weight(edge)), ...
            'FontSize', 9, 'Color', [0.1, 0.1, 0.1], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'BackgroundColor', 'w', 'Margin', 1);
    end
end

function draw_node_labels(ax, G, x_data, y_data)
    node_offset = 0.12;
    for node = 1:numnodes(G)
        dx = x_data(node) - 1.5;
        dy = y_data(node) - 1.5;
        radial_length = hypot(dx, dy);
        if radial_length > 0
            x_label = x_data(node) + node_offset*dx/radial_length;
            y_label = y_data(node) + node_offset*dy/radial_length;
        else
            x_label = x_data(node);
            y_label = y_data(node) + node_offset;
        end
        text(ax, x_label, y_label, G.Nodes.Name{node}, ...
            'FontSize', 12, 'FontWeight', 'bold', ...
            'Color', [0.15, 0.15, 0.15], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
end
