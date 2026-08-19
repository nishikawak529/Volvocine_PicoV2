function rank_sweep_results = global_joint_cp_rank_sweep_component_weights( ...
        round_dir, M, varargin)
%GLOBAL_JOINT_CP_RANK_SWEEP_COMPONENT_WEIGHTS Joint CP rank sweep.
%
%   C_fit(:,:,p) = A * diag(W(p,:)) * B.'
%
% A and B have size (2*M+1)-by-R, W has size P-by-R and is real.  Each
% column of A and B is unit norm and conjugate symmetric in Fourier order
% -M:M.  The transpose B.' is intentionally nonconjugate.  Unlike
% global_joint_common_shape_rank_sweep, every separable component has its
% own direction-dependent weight vector W(:,r).
%
% The coefficient tensor is loaded/built once and reused for every rank.
% Preferred input is a MAT file containing C_tensor, interaction_meta and
% the Fourier mode vectors.  If no suitable MAT file exists, the working
% rank-1 public function is called once and its returned tensor is reused.
%
% Examples:
%   results = global_joint_cp_rank_sweep_component_weights();
%   results = global_joint_cp_rank_sweep_component_weights( ...
%       fullfile('EstimateL','Round6'), 10, 'Ranks', 1:10);
%   results = global_joint_cp_rank_sweep_component_weights( ...
%       fullfile('EstimateL','Round6'), 10, ...
%       'TensorMatFile', 'path/to/global_joint_cp_rank1_fit.mat');

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'Round6');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end
    validateattributes(M, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');
    opts = parse_options(varargin{:});
    ranks_requested = opts.Ranks(:).';
    L_expected = 2 * M + 1;
    if max(ranks_requested) > L_expected
        error('Every requested rank must be at most 2*M+1=%d.', L_expected);
    end

    if opts.SyntheticTestOnly
        rank_sweep_results = run_synthetic_validation(M, opts);
        return;
    end

    output_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), ...
        sprintf('global_joint_cp_component_weights_rank_sweep_R%d_R%d', ...
        ranks_requested(1), ranks_requested(end)));
    log_path = fullfile(output_dir, ...
        'global_joint_cp_component_weights_rank_sweep_summary.txt');
    diary_cleanup = [];
    if opts.SaveOutputs
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        if exist(log_path, 'file')
            delete(log_path);
        end
        diary(log_path);
        diary_cleanup = onCleanup(@() diary('off'));
    end

    fprintf('[INFO] Component-weight joint CP rank sweep\n');
    fprintf('  Model: C_fit(:,:,p) = A*diag(W(p,:))*B.''\n');
    fprintf('  W has size P x R and is real; components are jointly optimized.\n');
    fprintf('  Data directory: %s\n', round_dir);
    fprintf('  Fourier order M=%d; requested ranks=%s\n', ...
        M, mat2str(ranks_requested));
    fprintf('  NumStarts=%d, MaxIter=%d, Tol=%.3e, RandomSeed=%d\n\n', ...
        opts.NumStarts, opts.MaxIter, opts.Tol, opts.RandomSeed);

    total_timer = tic;
    [C_tensor, interaction_meta, m_values, n_values, input_info, ...
        rank1_reference] = obtain_coefficient_tensor(round_dir, M, opts);
    validate_tensor_and_metadata(C_tensor, interaction_meta, ...
        m_values, n_values, M);
    P = size(C_tensor, 3);
    L = size(C_tensor, 1);
    unique_agents = unique([[interaction_meta.target_id], ...
        [interaction_meta.source_id]], 'sorted');
    tensor_energy = sum(abs(C_tensor(:)).^2);
    if tensor_energy == 0
        error('C_tensor has zero energy; explained fraction is undefined.');
    end
    fprintf('[INFO] C_tensor loaded once: %d x %d x %d (%s).\n', ...
        L, L, P, input_info.source);
    if P == 12
        fprintf('[INFO] Verified 12 directed interactions.\n');
    else
        warning('Current tensor has P=%d directions rather than 12.', P);
    end

    phi_grid = linspace(0, 2*pi, 512).';
    common_reference = load_common_weight_reference( ...
        round_dir, M, max(ranks_requested), C_tensor, ...
        interaction_meta, m_values, n_values, opts);
    rank1_reference = complete_rank1_reference( ...
        rank1_reference, round_dir, M, C_tensor, opts);

    old_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(old_rng_state));
    rng(opts.RandomSeed, 'twister');
    internal_ranks = 1:max(ranks_requested);
    [internal_fits, solver_diagnostics] = fit_rank_sequence( ...
        C_tensor, m_values, n_values, phi_grid, internal_ranks, opts, ...
        rank1_reference, common_reference);

    requested_indices = zeros(size(ranks_requested));
    for k = 1:numel(ranks_requested)
        requested_indices(k) = find(internal_ranks == ranks_requested(k), 1);
    end
    fits = internal_fits(requested_indices);
    internal_explained = [internal_fits.explained_fraction].';
    for k = 1:numel(fits)
        R = fits(k).rank;
        if R == 1
            increment = fits(k).explained_fraction;
        else
            increment = fits(k).explained_fraction - internal_explained(R-1);
        end
        fits(k).incremental_explained_fraction = increment;
    end

    explained_fraction = [fits.explained_fraction].';
    explained_percent = 100 * explained_fraction;
    incremental_explained_fraction = ...
        [fits.incremental_explained_fraction].';
    relative_residual = [fits.relative_residual].';
    objective = [fits.objective].';
    runtime_seconds = [fits.runtime_seconds].';
    per_direction_relative_residual = cat(2, ...
        fits.per_direction_relative_residual);
    per_direction_explained_fraction = cat(2, ...
        fits.per_direction_explained_fraction);

    rank1_comparison = compare_rank1_reference( ...
        rank1_reference, fits, C_tensor);
    common_weight_comparison = compare_common_weight_results( ...
        common_reference, fits, C_tensor);

    rank_sweep_results = struct();
    rank_sweep_results.model = ...
        'component-weight CP: C_p=sum_r W(p,r)*a_r*b_r.''';
    rank_sweep_results.ranks = ranks_requested(:);
    rank_sweep_results.explained_fraction = explained_fraction;
    rank_sweep_results.explained_percent = explained_percent;
    rank_sweep_results.incremental_explained_fraction = ...
        incremental_explained_fraction;
    rank_sweep_results.relative_residual = relative_residual;
    rank_sweep_results.objective = objective;
    rank_sweep_results.runtime_seconds = runtime_seconds;
    rank_sweep_results.per_direction_relative_residual = ...
        per_direction_relative_residual;
    rank_sweep_results.per_direction_explained_fraction = ...
        per_direction_explained_fraction;
    rank_sweep_results.fits = fits(:);
    rank_sweep_results.C_tensor = C_tensor;
    rank_sweep_results.interaction_meta = interaction_meta;
    rank_sweep_results.m_values = m_values;
    rank_sweep_results.n_values = n_values;
    rank_sweep_results.phi_grid = phi_grid;
    rank_sweep_results.unique_agents = unique_agents(:);
    rank_sweep_results.options = opts;
    rank_sweep_results.input_info = input_info;
    rank_sweep_results.internal_ranks = internal_ranks(:);
    rank_sweep_results.internal_fits = internal_fits(:);
    rank_sweep_results.solver_diagnostics = solver_diagnostics(:);
    rank_sweep_results.rank1_comparison = rank1_comparison;
    rank_sweep_results.common_weight_comparison = ...
        common_weight_comparison;
    rank_sweep_results.synthetic_validation = struct('ran', false);
    rank_sweep_results.output_dir = output_dir;
    rank_sweep_results.output_files = make_output_paths( ...
        output_dir, max(ranks_requested));
    rank_sweep_results.total_runtime_seconds = toc(total_timer);

    if opts.RunSyntheticValidation
        fprintf('\n[INFO] Running the noiseless rank-3 validation.\n');
        rank_sweep_results.synthetic_validation = ...
            run_synthetic_validation(max(M, 3), opts);
    end

    fprintf('\n[RESULT] Component-weight CP rank sweep\n');
    for k = 1:numel(fits)
        fprintf(['  R=%2d: explained=%12.8f%%, increment=%10.8f pp, ' ...
            'relres=%.6e, J=%.6e, start=%d, iter=%d, status=%s\n'], ...
            fits(k).rank, fits(k).explained_percent, ...
            100*fits(k).incremental_explained_fraction, ...
            fits(k).relative_residual, fits(k).objective, ...
            fits(k).best_start, fits(k).iterations, ...
            fits(k).solver_status);
    end
    if common_weight_comparison.available
        fprintf('\n[COMPARISON] Component minus common explained percentage points\n');
        for k = 1:numel(common_weight_comparison.ranks)
            fprintf('  R=%2d: %+12.8f pp\n', ...
                common_weight_comparison.ranks(k), ...
                100*common_weight_comparison.explained_fraction_difference(k));
        end
    else
        fprintf('\n[COMPARISON] Common-weight result unavailable: %s\n', ...
            common_weight_comparison.reason);
    end

    if opts.DisplayFullArrays
        for k = 1:numel(fits)
            fprintf('\nW for R=%d (rows=directions, columns=components):\n', ...
                fits(k).rank);
            disp(fits(k).W);
        end
    end

    if opts.SaveOutputs
        save_rank_sweep_outputs(rank_sweep_results, opts);
        fprintf('\n[INFO] Results saved under %s\n', output_dir);
    end
    clear rng_cleanup;
    if ~isempty(diary_cleanup)
        clear diary_cleanup;
    end
end

function opts = parse_options(varargin)
    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'Ranks', 1:10, @(x) isnumeric(x) && isvector(x) && ...
        ~isempty(x) && all(isfinite(x)) && all(x >= 1) && ...
        all(x == fix(x)) && numel(unique(x)) == numel(x));
    addParameter(parser, 'NumStarts', 20, @(x) isnumeric(x) && isscalar(x) && ...
        isfinite(x) && x >= 1 && x == fix(x));
    addParameter(parser, 'MaxIter', 1000, @(x) isnumeric(x) && isscalar(x) && ...
        isfinite(x) && x >= 1 && x == fix(x));
    addParameter(parser, 'Tol', 1e-10, @(x) isnumeric(x) && isscalar(x) && ...
        isfinite(x) && x > 0);
    addParameter(parser, 'RandomSeed', 0, @(x) isnumeric(x) && isscalar(x) && ...
        isfinite(x) && x >= 0 && x == fix(x));
    addParameter(parser, 'TensorMatFile', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'ExistingRank1Mat', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'CommonWeightMatFile', '', ...
        @(x) ischar(x) || isstring(x));
    addParameter(parser, 'DisplayFullArrays', false, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RunSyntheticValidation', false, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SyntheticTestOnly', false, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SaveOutputs', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'keep_figures', false, ...
        @(x) islogical(x) && isscalar(x));

    % Options forwarded only when a coefficient MAT file is unavailable and
    % the existing rank-1 public function must build C_tensor once.
    addParameter(parser, 'analysis_start_sec', 10, @is_finite_scalar);
    addParameter(parser, 'analysis_duration_sec', 80, @is_finite_scalar);
    addParameter(parser, 'sample_dt', 0.01, @is_positive_scalar);
    addParameter(parser, 'signal_column', 'a2', ...
        @(x) ischar(x) || isstring(x));
    addParameter(parser, 'normalize_signal', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'tail_percent', 10, @is_finite_scalar);
    addParameter(parser, 'clip_normalized_signal', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'clip_limit', 0.5, @is_positive_scalar);
    addParameter(parser, 'use_cache', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'file_indices', [], ...
        @(x) isempty(x) || (isnumeric(x) && isvector(x)));
    addParameter(parser, 'RemoveSelfOnly', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RemoveConstant', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RemoveOtherOnly', false, ...
        @(x) islogical(x) && isscalar(x));
    parse(parser, varargin{:});
    opts = parser.Results;
    opts.Ranks = sort(opts.Ranks(:).');
    opts.TensorMatFile = char(opts.TensorMatFile);
    opts.ExistingRank1Mat = char(opts.ExistingRank1Mat);
    opts.CommonWeightMatFile = char(opts.CommonWeightMatFile);
    opts.signal_column = char(opts.signal_column);
    opts.cache_dir = char(opts.cache_dir);
end

function tf = is_finite_scalar(x)
    tf = isnumeric(x) && isscalar(x) && isfinite(x);
end

function tf = is_positive_scalar(x)
    tf = is_finite_scalar(x) && x > 0;
end

function [C_tensor, interaction_meta, m_values, n_values, input_info, ...
        rank1_reference] = obtain_coefficient_tensor(round_dir, M, opts)
    auto_rank1_path = fullfile(round_dir, 'low_rank_analysis', ...
        sprintf('M%d', M), 'global_joint_cp_rank1', ...
        'global_joint_cp_rank1_fit.mat');
    if ~isempty(opts.TensorMatFile)
        if ~exist(opts.TensorMatFile, 'file')
            error('TensorMatFile does not exist: %s', opts.TensorMatFile);
        end
        source_path = opts.TensorMatFile;
        loaded = load(source_path);
        [C_tensor, interaction_meta, m_values, n_values, root] = ...
            extract_tensor_payload(loaded, M, source_path);
        rank1_reference = extract_rank1_reference(root, source_path);
        input_info = struct('source', 'explicit MAT file', ...
            'path', source_path, 'rank1_function_called', false);
        return;
    end

    if exist(auto_rank1_path, 'file')
        loaded = load(auto_rank1_path);
        [C_tensor, interaction_meta, m_values, n_values, root] = ...
            extract_tensor_payload(loaded, M, auto_rank1_path);
        rank1_reference = extract_rank1_reference(root, auto_rank1_path);
        input_info = struct('source', 'existing rank-1 MAT file', ...
            'path', auto_rank1_path, 'rank1_function_called', false);
        return;
    end

    fprintf(['[INFO] No tensor MAT file was found. Calling the existing ' ...
        'rank-1 function once to construct C_tensor.\n']);
    base = global_joint_svd_analysis_joint_cp_rank1(round_dir, M, ...
        'analysis_start_sec', opts.analysis_start_sec, ...
        'analysis_duration_sec', opts.analysis_duration_sec, ...
        'sample_dt', opts.sample_dt, ...
        'signal_column', opts.signal_column, ...
        'normalize_signal', opts.normalize_signal, ...
        'tail_percent', opts.tail_percent, ...
        'clip_normalized_signal', opts.clip_normalized_signal, ...
        'clip_limit', opts.clip_limit, ...
        'use_cache', opts.use_cache, ...
        'cache_dir', opts.cache_dir, ...
        'file_indices', opts.file_indices, ...
        'RemoveSelfOnly', opts.RemoveSelfOnly, ...
        'RemoveConstant', opts.RemoveConstant, ...
        'RemoveOtherOnly', opts.RemoveOtherOnly, ...
        'NumStarts', opts.NumStarts, ...
        'MaxIter', opts.MaxIter, 'Tol', opts.Tol, ...
        'RandomSeed', opts.RandomSeed, ...
        'DisplayFullArrays', false, 'keep_figures', false);
    C_tensor = base.C_tensor;
    interaction_meta = base.interaction_meta;
    m_values = base.m_values;
    n_values = base.n_values;
    rank1_reference = extract_rank1_reference(base, ...
        'rank-1 function return value');
    input_info = struct('source', 'rank-1 function return value', ...
        'path', '', 'rank1_function_called', true);
end

function [C_tensor, interaction_meta, m_values, n_values, root] = ...
        extract_tensor_payload(loaded, M, source_path)
    root = loaded;
    if isfield(loaded, 'rank_sweep_results') && ...
            isstruct(loaded.rank_sweep_results)
        root = loaded.rank_sweep_results;
    elseif isfield(loaded, 'all_global_results') && ...
            isstruct(loaded.all_global_results)
        root = loaded.all_global_results;
    end
    if ~isfield(root, 'C_tensor')
        error('MAT file does not contain C_tensor: %s', source_path);
    end
    if ~isfield(root, 'interaction_meta')
        error('MAT file does not contain interaction_meta: %s', source_path);
    end
    C_tensor = root.C_tensor;
    interaction_meta = root.interaction_meta;
    if isfield(root, 'm_values')
        m_values = root.m_values;
    elseif isfield(root, 'm_vals')
        m_values = root.m_vals;
    else
        m_values = (-M:M).';
    end
    if isfield(root, 'n_values')
        n_values = root.n_values;
    elseif isfield(root, 'n_vals')
        n_values = root.n_vals;
    else
        n_values = (-M:M).';
    end
end

function validate_tensor_and_metadata(C_tensor, interaction_meta, ...
        m_values, n_values, M)
    L = 2*M + 1;
    if ndims(C_tensor) ~= 3 || size(C_tensor,1) ~= L || ...
            size(C_tensor,2) ~= L
        error('C_tensor must have size (2*M+1) x (2*M+1) x P.');
    end
    if any(~isfinite(C_tensor(:)))
        error('C_tensor contains NaN or Inf.');
    end
    if ~isequal(m_values(:), (-M:M).') || ...
            ~isequal(n_values(:), (-M:M).')
        error('Fourier coefficient order must be -M:M on both axes.');
    end
    if numel(interaction_meta) ~= size(C_tensor,3)
        error('interaction_meta length must equal size(C_tensor,3).');
    end
    required = {'pair_name','target_id','source_id','direction'};
    for k = 1:numel(required)
        if ~isfield(interaction_meta, required{k})
            error('interaction_meta is missing field %s.', required{k});
        end
    end
end

function reference = empty_rank1_reference()
    reference = struct('available', false, 'reason', 'not loaded', ...
        'path', '', 'tensor_relative_mismatch', NaN, ...
        'A', [], 'B', [], 'W', [], 'C_fit', [], ...
        'objective', NaN, 'explained_fraction', NaN);
end

function reference = extract_rank1_reference(root, source_path)
    reference = empty_rank1_reference();
    reference.path = source_path;
    needed = {'C_tensor','a','b','w'};
    if ~all(isfield(root, needed))
        reference.reason = 'a, b, w rank-1 factors are absent';
        return;
    end
    A = root.a(:);
    B = root.b(:);
    W = root.w(:);
    [A, B, W, ok, reason] = normalize_cp_columns(A, B, W);
    if ~ok
        reference.reason = reason;
        return;
    end
    C_fit = reconstruct_cp_tensor(A, B, W, size(root.C_tensor));
    residual = root.C_tensor - C_fit;
    objective = sum(abs(residual(:)).^2);
    energy = sum(abs(root.C_tensor(:)).^2);
    reference.available = true;
    reference.reason = 'loaded';
    reference.A = A;
    reference.B = B;
    reference.W = W;
    reference.C_fit = C_fit;
    reference.objective = objective;
    reference.explained_fraction = 1 - objective/energy;
end

function reference = complete_rank1_reference(reference, round_dir, M, ...
        C_tensor, opts)
    explicit = ~isempty(opts.ExistingRank1Mat);
    if explicit
        path = opts.ExistingRank1Mat;
    else
        path = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), ...
            'global_joint_cp_rank1', 'global_joint_cp_rank1_fit.mat');
    end
    if (~reference.available || explicit) && exist(path, 'file')
        loaded = load(path);
        [C_ref, ~, ~, ~, root] = extract_tensor_payload(loaded, M, path);
        mismatch = norm(C_ref(:)-C_tensor(:)) / max(norm(C_tensor(:)), eps);
        if mismatch <= 1e-10
            candidate = extract_rank1_reference(root, path);
            candidate.tensor_relative_mismatch = mismatch;
            if candidate.available
                reference = candidate;
            end
        elseif explicit
            error('ExistingRank1Mat tensor mismatch is %.3e.', mismatch);
        else
            reference.reason = sprintf('auto rank-1 tensor mismatch %.3e', mismatch);
        end
    elseif explicit
        error('ExistingRank1Mat does not exist: %s', path);
    end
    if reference.available
        C_recons = reconstruct_cp_tensor( ...
            reference.A, reference.B, reference.W, size(C_tensor));
        mismatch = norm(reference.C_fit(:) - C_recons(:));
        if ~isfinite(mismatch)
            reference.available = false;
            reference.reason = 'nonfinite rank-1 reconstruction';
        end
    end
end

function common = load_common_weight_reference(round_dir, M, max_rank, ...
        C_tensor, interaction_meta, m_values, n_values, opts)
    common = struct('available', false, 'reason', 'not found', 'path', '', ...
        'tensor_relative_mismatch', NaN, 'root', []);
    explicit = ~isempty(opts.CommonWeightMatFile);
    if explicit
        path = opts.CommonWeightMatFile;
    else
        path = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), ...
            sprintf('global_joint_common_shape_rank_sweep_R1_R%d', max_rank), ...
            'global_joint_common_shape_rank_sweep_results.mat');
        if ~exist(path, 'file')
            pattern = fullfile(round_dir, 'low_rank_analysis', ...
                sprintf('M%d', M), ...
                'global_joint_common_shape_rank_sweep_R1_R*', ...
                'global_joint_common_shape_rank_sweep_results.mat');
            candidates = dir(pattern);
            if ~isempty(candidates)
                [~, newest] = max([candidates.datenum]);
                path = fullfile(candidates(newest).folder, ...
                    candidates(newest).name);
            end
        end
    end
    common.path = path;
    if ~exist(path, 'file')
        if explicit
            error('CommonWeightMatFile does not exist: %s', path);
        end
        common.reason = 'common-weight MAT file not found';
        return;
    end
    loaded = load(path);
    if ~isfield(loaded, 'rank_sweep_results')
        if explicit
            error('CommonWeightMatFile lacks rank_sweep_results: %s', path);
        end
        common.reason = 'MAT lacks rank_sweep_results';
        return;
    end
    root = loaded.rank_sweep_results;
    needed = {'C_tensor','interaction_meta','m_values','n_values','fits'};
    if ~all(isfield(root, needed))
        if explicit
            error('CommonWeightMatFile is missing required fields.');
        end
        common.reason = 'common result is missing required fields';
        return;
    end
    mismatch = norm(root.C_tensor(:)-C_tensor(:)) / ...
        max(norm(C_tensor(:)), eps);
    if mismatch > 1e-10 || ~isequal(root.m_values(:), m_values(:)) || ...
            ~isequal(root.n_values(:), n_values(:)) || ...
            ~metadata_matches(root.interaction_meta, interaction_meta)
        if explicit
            error('CommonWeightMatFile does not describe the same tensor.');
        end
        common.reason = sprintf('common tensor/metadata mismatch %.3e', mismatch);
        return;
    end
    common.available = true;
    common.reason = 'loaded and tensor-matched';
    common.tensor_relative_mismatch = mismatch;
    common.root = root;
end

function tf = metadata_matches(a, b)
    tf = numel(a) == numel(b);
    if ~tf
        return;
    end
    for p = 1:numel(a)
        tf = tf && isequal(a(p).target_id, b(p).target_id) && ...
            isequal(a(p).source_id, b(p).source_id) && ...
            strcmp(char(a(p).pair_name), char(b(p).pair_name)) && ...
            strcmp(char(a(p).direction), char(b(p).direction));
        if ~tf
            return;
        end
    end
end

function [fits, diagnostics] = fit_rank_sequence(C_tensor, m_values, ...
        n_values, phi_grid, ranks, opts, rank1_reference, common_reference)
    data_scale = norm(C_tensor(:));
    C_work = C_tensor / data_scale;
    fits = repmat(empty_final_fit(), numel(ranks), 1);
    diagnostics = repmat(empty_rank_diagnostics(), numel(ranks), 1);
    previous_fit = [];

    for index = 1:numel(ranks)
        R = ranks(index);
        rank_timer = tic;
        baselines = {};
        predefined = {};

        common_candidate = common_candidate_for_rank( ...
            common_reference, R, C_work, data_scale);
        if common_candidate.available
            baselines{end+1} = common_candidate; %#ok<AGROW>
        end

        if R == 1
            legacy_candidate = rank1_candidate( ...
                rank1_reference, C_work, data_scale);
            if legacy_candidate.available
                baselines{end+1} = legacy_candidate; %#ok<AGROW>
                predefined{end+1} = legacy_candidate; %#ok<AGROW>
            end
            if common_candidate.available
                predefined{end+1} = common_candidate; %#ok<AGROW>
            end
        else
            zero_baseline = zero_padded_previous_candidate( ...
                previous_fit, C_work, data_scale);
            baselines{end+1} = zero_baseline; %#ok<AGROW>
            warm_candidate = residual_warm_start( ...
                previous_fit, C_work, data_scale, opts);
            predefined{end+1} = warm_candidate; %#ok<AGROW>
            if common_candidate.available
                predefined{end+1} = common_candidate; %#ok<AGROW>
            end
        end

        baseline = choose_best_candidate(baselines);
        solver = fit_cp_rank_multistart( ...
            C_work, R, opts, predefined);

        selection_tolerance = 128 * eps(max(1, solver.objective));
        use_baseline = baseline.available && ...
            (~solver.available || solver.objective >= ...
            baseline.objective - selection_tolerance);
        if use_baseline
            selected = baseline;
            iterations = 0;
            best_start = 0;
            converged = false;
            solver_status = sprintf('baseline_retained:%s', baseline.source);
            objective_history_work = baseline.objective;
        elseif solver.available
            selected = make_cp_candidate(C_work, solver.A, solver.B, ...
                solver.W, sprintf('optimized_start_%d', solver.best_start));
            iterations = solver.iterations;
            best_start = solver.best_start;
            converged = solver.converged;
            solver_status = sprintf('optimized_start_%d:%s', ...
                solver.best_start, solver.status);
            objective_history_work = solver.objective_history;
        else
            error('All starts failed at rank %d and no feasible baseline exists.', R);
        end

        W_physical = selected.W * data_scale;
        runtime_seconds = toc(rank_timer);
        fit = finalize_cp_fit(C_tensor, selected.A, selected.B, W_physical, ...
            R, m_values, n_values, phi_grid, iterations, best_start, ...
            converged, runtime_seconds, solver_status, ...
            objective_history_work * data_scale^2);

        if R > 1
            monotonic_tolerance = 256 * eps(max([1, ...
                previous_fit.objective, fit.objective]));
            if fit.objective > previous_fit.objective + monotonic_tolerance
                error(['Rank-%d result is worse than rank-%d despite the ' ...
                    'zero-weight baseline.'], R, R-1);
            end
        end
        fits(index) = orderfields(fit, fits(index));
        previous_fit = fit;

        diagnostics(index).rank = R;
        diagnostics(index).data_scale = data_scale;
        diagnostics(index).baseline_available = baseline.available;
        diagnostics(index).baseline_source = baseline.source;
        diagnostics(index).baseline_objective = ...
            baseline.objective * data_scale^2;
        diagnostics(index).optimized_objective = ...
            solver.objective * data_scale^2;
        diagnostics(index).start_objectives = ...
            solver.start_objectives * data_scale^2;
        diagnostics(index).start_iterations = solver.start_iterations;
        diagnostics(index).start_converged = solver.start_converged;
        diagnostics(index).start_status = solver.start_status;
        diagnostics(index).selected_baseline = use_baseline;
        diagnostics(index).common_baseline_available = ...
            common_candidate.available;
        if common_candidate.available
            diagnostics(index).common_baseline_objective = ...
                common_candidate.objective * data_scale^2;
        end

        fprintf(['[RANK %2d] explained=%12.8f%%, relres=%.6e, J=%.6e, ' ...
            'start=%d, iter=%d\n'], R, fit.explained_percent, ...
            fit.relative_residual, fit.objective, fit.best_start, ...
            fit.iterations);
        fprintf('           W size=%dx%d, max symmetry A/B=%.3e/%.3e, status=%s\n', ...
            size(fit.W,1), size(fit.W,2), ...
            max(fit.A_conjugate_symmetry_error), ...
            max(fit.B_conjugate_symmetry_error), fit.solver_status);
    end
end

function fit = empty_final_fit()
    fit = struct('rank', [], 'A', [], 'B', [], 'W', [], 'C_fit', [], ...
        'a_values', [], 'b_values', [], 'a_values_complex', [], ...
        'b_values_complex', [], 'objective', [], 'relative_residual', [], ...
        'explained_fraction', [], 'explained_percent', [], ...
        'incremental_explained_fraction', NaN, ...
        'per_direction_relative_residual', [], ...
        'per_direction_explained_fraction', [], 'iterations', [], ...
        'best_start', [], 'converged', [], 'runtime_seconds', [], ...
        'solver_status', '', 'objective_history', [], ...
        'component_weight_norm', [], 'component_display_order', [], ...
        'A_conjugate_symmetry_error', [], ...
        'B_conjugate_symmetry_error', [], 'max_imag_a_values', [], ...
        'max_imag_b_values', [], 'reconstruction_check_error', [], ...
        'objective_check_error', [], 'normal_equation_residual', []);
end

function diagnostics = empty_rank_diagnostics()
    diagnostics = struct('rank', [], 'data_scale', [], ...
        'baseline_available', false, 'baseline_source', '', ...
        'baseline_objective', Inf, 'optimized_objective', Inf, ...
        'start_objectives', [], 'start_iterations', [], ...
        'start_converged', [], 'start_status', {{}}, ...
        'selected_baseline', false, ...
        'common_baseline_available', false, ...
        'common_baseline_objective', Inf);
end

function candidate = unavailable_candidate(reason)
    candidate = struct('available', false, 'source', reason, 'A', [], ...
        'B', [], 'W', [], 'C_fit', [], 'objective', Inf);
end

function candidate = make_cp_candidate(C_tensor, A, B, W, source)
    candidate = unavailable_candidate(source);
    [A, B, W, ok, reason] = normalize_cp_columns(A, B, W);
    if ~ok
        candidate.source = sprintf('%s_invalid:%s', source, reason);
        return;
    end
    if size(A,1) ~= size(C_tensor,1) || size(B,1) ~= size(C_tensor,2) || ...
            size(W,1) ~= size(C_tensor,3) || ...
            size(A,2) ~= size(B,2) || size(A,2) ~= size(W,2)
        candidate.source = sprintf('%s_invalid_dimensions', source);
        return;
    end
    C_fit = reconstruct_cp_tensor(A, B, W, size(C_tensor));
    residual = C_tensor - C_fit;
    objective = sum(abs(residual(:)).^2);
    if ~isfinite(objective)
        candidate.source = sprintf('%s_nonfinite_objective', source);
        return;
    end
    candidate.available = true;
    candidate.source = source;
    candidate.A = A;
    candidate.B = B;
    candidate.W = real(W);
    candidate.C_fit = C_fit;
    candidate.objective = objective;
end

function best = choose_best_candidate(candidates)
    best = unavailable_candidate('none');
    for k = 1:numel(candidates)
        candidate = candidates{k};
        if candidate.available && candidate.objective < best.objective
            best = candidate;
        end
    end
end

function candidate = rank1_candidate(reference, C_work, data_scale)
    if ~reference.available
        candidate = unavailable_candidate('legacy_rank1_unavailable');
        return;
    end
    candidate = make_cp_candidate(C_work, reference.A, reference.B, ...
        reference.W/data_scale, 'legacy_rank1');
end

function candidate = common_candidate_for_rank(common, R, C_work, data_scale)
    if ~common.available
        candidate = unavailable_candidate('common_weight_unavailable');
        return;
    end
    common_fit = find_common_fit(common.root, R);
    if isempty(common_fit)
        candidate = unavailable_candidate(sprintf('common_rank_%d_missing', R));
        return;
    end
    needed = {'A_complex','B_complex','sigma','w'};
    if ~all(isfield(common_fit, needed))
        candidate = unavailable_candidate(sprintf('common_rank_%d_fields_missing', R));
        return;
    end
    A = common_fit.A_complex;
    B = common_fit.B_complex;
    sigma = common_fit.sigma(:);
    if size(A,2) < R || size(B,2) < R || numel(sigma) < R
        candidate = unavailable_candidate(sprintf('common_rank_%d_dimension_error', R));
        return;
    end
    A = A(:,1:R);
    B = B(:,1:R);
    W = common_fit.w(:) * sigma(1:R).';
    candidate = make_cp_candidate(C_work, A, B, W/data_scale, ...
        sprintf('common_weight_rank_%d', R));
end

function fit = find_common_fit(root, R)
    fit = [];
    if isfield(root, 'fits')
        values = root.fits;
        if ~isempty(values) && isfield(values, 'rank')
            index = find([values.rank] == R, 1);
            if ~isempty(index)
                fit = values(index);
                return;
            end
        end
    end
    if isfield(root, 'internal_fits')
        values = root.internal_fits;
        if ~isempty(values) && isfield(values, 'rank')
            index = find([values.rank] == R, 1);
            if ~isempty(index)
                fit = values(index);
            end
        end
    end
end

function candidate = zero_padded_previous_candidate( ...
        previous_fit, C_work, data_scale)
    L = size(C_work, 1);
    A = [previous_fit.A, random_conjugate_symmetric_unit_vector(L)];
    B = [previous_fit.B, random_conjugate_symmetric_unit_vector(L)];
    W = [previous_fit.W/data_scale, zeros(size(C_work,3),1)];
    candidate = make_cp_candidate(C_work, A, B, W, ...
        sprintf('rank_%d_zero_weight_extension', previous_fit.rank));
end

function candidate = residual_warm_start(previous_fit, C_work, ...
        data_scale, opts)
    previous_W_work = previous_fit.W / data_scale;
    previous_C_fit = reconstruct_cp_tensor(previous_fit.A, previous_fit.B, ...
        previous_W_work, size(C_work));
    residual = C_work - previous_C_fit;
    [a_new, b_new, w_new, status] = ...
        fit_residual_rank1_initialization(residual, opts);
    A = [previous_fit.A, a_new];
    B = [previous_fit.B, b_new];
    W = [previous_W_work, w_new];
    candidate = make_cp_candidate(C_work, A, B, W, ...
        sprintf('rank_%d_warm_residual_%s', previous_fit.rank, status));
end

function [a, b, w, status] = fit_residual_rank1_initialization( ...
        residual, opts)
    L = size(residual,1);
    P = size(residual,3);
    if norm(residual(:)) <= 1e-14
        a = random_conjugate_symmetric_unit_vector(L);
        b = random_conjugate_symmetric_unit_vector(L);
        w = zeros(P,1);
        status = 'zero_residual';
        return;
    end
    warm_opts = opts;
    warm_opts.NumStarts = min(5, opts.NumStarts);
    warm_opts.MaxIter = min(200, opts.MaxIter);
    warm = fit_cp_rank_multistart(residual, 1, warm_opts, {});
    if warm.available
        a = warm.A;
        b = warm.B;
        w = warm.W;
        status = sprintf('rank1_start_%d', warm.best_start);
    else
        a = random_conjugate_symmetric_unit_vector(L);
        b = random_conjugate_symmetric_unit_vector(L);
        w = zeros(P,1);
        status = 'rank1_failed';
    end
end

function solver = fit_cp_rank_multistart(C_tensor, R, opts, predefined)
    solver = empty_multistart_result(opts.NumStarts);
    best_objective = Inf;
    for start_index = 1:opts.NumStarts
        if start_index <= numel(predefined) && ...
                predefined{start_index}.available
            initialization = predefined{start_index};
        else
            initialization = random_cp_initialization(C_tensor, R);
        end
        result = run_one_cp_start(C_tensor, initialization.A, ...
            initialization.B, initialization.W, opts);
        solver.start_objectives(start_index) = result.objective;
        solver.start_iterations(start_index) = result.iterations;
        solver.start_converged(start_index) = result.converged;
        solver.start_status{start_index} = result.status;
        if result.available && result.objective < best_objective
            best_objective = result.objective;
            solver.available = true;
            solver.A = result.A;
            solver.B = result.B;
            solver.W = result.W;
            solver.objective = result.objective;
            solver.objective_history = result.objective_history;
            solver.iterations = result.iterations;
            solver.best_start = start_index;
            solver.converged = result.converged;
            solver.status = result.status;
            solver.weight_solver = result.weight_solver;
        end
    end
end

function solver = empty_multistart_result(num_starts)
    solver = struct('available', false, 'A', [], 'B', [], 'W', [], ...
        'objective', Inf, 'objective_history', [], 'iterations', 0, ...
        'best_start', 0, 'converged', false, 'status', 'all_starts_failed', ...
        'weight_solver', '', 'start_objectives', inf(num_starts,1), ...
        'start_iterations', zeros(num_starts,1), ...
        'start_converged', false(num_starts,1), ...
        'start_status', {cell(num_starts,1)});
end

function initialization = random_cp_initialization(C_tensor, R)
    L = size(C_tensor,1);
    A = complex(zeros(L,R));
    B = complex(zeros(L,R));
    for r = 1:R
        A(:,r) = random_conjugate_symmetric_unit_vector(L);
        B(:,r) = random_conjugate_symmetric_unit_vector(L);
    end
    [W, method, ok] = update_component_weights(C_tensor, A, B);
    if ~ok
        W = randn(size(C_tensor,3), R);
        method = 'random_weight_fallback';
    end
    initialization = make_cp_candidate(C_tensor, A, B, W, ...
        ['random_' method]);
end

function result = run_one_cp_start(C_tensor, A, B, W, opts)
    result = empty_start_result();
    [A, B, ~, ok, reason] = normalize_cp_columns(A, B, W);
    if ~ok
        result.status = ['invalid_initialization:' reason];
        return;
    end
    [W_initial, weight_method, weight_ok] = ...
        update_component_weights(C_tensor, A, B);
    if weight_ok
        W = W_initial;
    else
        result.status = 'initial_weight_least_squares_failed';
        return;
    end
    C_fit = reconstruct_cp_tensor(A, B, W, size(C_tensor));
    residual = C_tensor - C_fit;
    accepted_objective = sum(abs(residual(:)).^2);
    if ~isfinite(accepted_objective)
        result.status = 'nonfinite_initial_objective';
        return;
    end
    objective_history = accepted_objective;
    accepted_iterations = 0;
    converged = false;
    status = 'max_iter';
    used_weight_methods = {weight_method};
    R = size(A,2);
    P = size(C_tensor,3);

    for iteration = 1:opts.MaxIter
        A_candidate = A;
        B_candidate = B;
        W_candidate = W;
        C_fit_candidate = C_fit;
        degenerate_updates = 0;

        for r = 1:R
            component_old = A_candidate(:,r) * B_candidate(:,r).';
            W_reshaped = reshape(W_candidate(:,r), 1, 1, P);

            E_minus = (C_tensor - C_fit_candidate) + component_old .* W_reshaped;

            E_weighted_a = sum(E_minus .* W_reshaped, 3);
            a_raw = E_weighted_a * conj(B_candidate(:,r));
            a_projected = project_conjugate_symmetry(a_raw);
            a_norm = norm(a_projected);
            if isfinite(a_norm) && a_norm > 1e-14
                A_candidate(:,r) = a_projected/a_norm;
            else
                degenerate_updates = degenerate_updates + 1;
            end

            E_weighted_b = sum(permute(E_minus, [2 1 3]) .* W_reshaped, 3);
            b_raw = E_weighted_b * conj(A_candidate(:,r));
            b_projected = project_conjugate_symmetry(b_raw);
            b_norm = norm(b_projected);
            if isfinite(b_norm) && b_norm > 1e-14
                B_candidate(:,r) = b_projected/b_norm;
            else
                degenerate_updates = degenerate_updates + 1;
            end

            component_new = A_candidate(:,r) * B_candidate(:,r).';
            delta_component = component_new - component_old;
            C_fit_candidate = C_fit_candidate + delta_component .* W_reshaped;
        end

        [W_new, method, weight_ok] = update_component_weights( ...
            C_tensor, A_candidate, B_candidate);
        used_weight_methods{end+1} = method; %#ok<AGROW>
        if ~weight_ok || any(~isfinite(W_new(:)))
            status = ['weight_least_squares_failed:' method];
            break;
        end
        W_candidate = W_new;
        C_fit_candidate = reconstruct_cp_tensor( ...
            A_candidate, B_candidate, W_candidate, size(C_tensor));
        residual_candidate = C_tensor - C_fit_candidate;
        objective_candidate = sum(abs(residual_candidate(:)).^2);
        if ~isfinite(objective_candidate)
            status = 'nonfinite_objective';
            break;
        end

        increase = objective_candidate - accepted_objective;
        numerical_floor = 128 * eps(max([1, accepted_objective, ...
            sum(abs(C_tensor(:)).^2)]));
        if increase > numerical_floor
            status = sprintf('objective_increase_rejected:%.3e', increase);
            break;
        elseif increase > 0
            converged = true;
            status = 'numerical_objective_floor';
            break;
        end

        relative_change = abs(accepted_objective-objective_candidate) / ...
            max([abs(accepted_objective), abs(objective_candidate), eps]);
        A = A_candidate;
        B = B_candidate;
        W = real(W_candidate);
        C_fit = C_fit_candidate;
        accepted_objective = objective_candidate;
        accepted_iterations = iteration;
        objective_history(end+1,1) = accepted_objective; %#ok<AGROW>

        if relative_change <= opts.Tol
            converged = true;
            if degenerate_updates > 0
                status = sprintf('converged_with_%d_degenerate_updates', ...
                    degenerate_updates);
            else
                status = 'converged';
            end
            break;
        end
    end

    if any(~isfinite(A(:))) || any(~isfinite(B(:))) || ...
            any(~isfinite(W(:)))
        result.status = 'nonfinite_final_factors';
        return;
    end
    result.available = true;
    result.A = A;
    result.B = B;
    result.W = real(W);
    result.objective = accepted_objective;
    result.objective_history = objective_history;
    result.iterations = accepted_iterations;
    result.converged = converged;
    result.status = status;
    if any(strcmp(used_weight_methods, 'augmented_lsqminnorm')) || ...
            any(strcmp(used_weight_methods, 'augmented_pinv'))
        result.weight_solver = 'augmented_real_least_squares_used';
        result.status = [result.status ...
            ';weight_solver=augmented_real_least_squares'];
    else
        result.weight_solver = 'normal_equations';
        result.status = [result.status ';weight_solver=normal_equations'];
    end
end

function result = empty_start_result()
    result = struct('available', false, 'A', [], 'B', [], 'W', [], ...
        'objective', Inf, 'objective_history', [], 'iterations', 0, ...
        'converged', false, 'status', 'not_started', 'weight_solver', '');
end

function [W, method, ok, info] = update_component_weights(C_tensor, A, B)
    R = size(A,2);
    P = size(C_tensor,3);
    D = complex(zeros(numel(C_tensor(:,:,1)), R));
    for r = 1:R
        D(:,r) = reshape(A(:,r)*B(:,r).', [], 1);
    end
    X = reshape(C_tensor, [], P);
    gram = real(D' * D);
    gram = 0.5 * (gram + gram.');
    rhs = real(D' * X);
    gram_rcond = rcond(gram);
    numerical_rank = rank([real(D); imag(D)]);
    ok = true;
    if isfinite(gram_rcond) && gram_rcond >= 1e-12 && ...
            numerical_rank == R
        W = (gram \ rhs).';
        method = 'normal_equations';
    else
        D_augmented = [real(D); imag(D)];
        X_augmented = [real(X); imag(X)];
        if exist('lsqminnorm', 'file') == 2
            W = lsqminnorm(D_augmented, X_augmented).';
            method = 'augmented_lsqminnorm';
        else
            W = (pinv(D_augmented) * X_augmented).';
            method = 'augmented_pinv';
        end
    end
    W = real(W);
    if ~isequal(size(W), [P,R]) || any(~isfinite(W(:)))
        ok = false;
    end
    normal_residual = norm(real(D'*(D*W.'-X)), 'fro');
    info = struct('gram_rcond', gram_rcond, ...
        'design_rank', numerical_rank, ...
        'normal_equation_residual', normal_residual);
end

function C_fit = reconstruct_cp_tensor(A, B, W, tensor_size)
    C_fit = complex(zeros(tensor_size));
    for p = 1:tensor_size(3)
        C_fit(:,:,p) = A * diag(W(p,:)) * B.';
    end
end

function [A, B, W, ok, reason] = normalize_cp_columns(A, B, W)
    ok = true;
    reason = 'ok';
    if ~ismatrix(A) || ~ismatrix(B) || ~ismatrix(W) || ...
            size(A,2) ~= size(B,2) || size(A,2) ~= size(W,2)
        ok = false;
        reason = 'factor dimensions disagree';
        return;
    end
    W = real(W);
    for r = 1:size(A,2)
        A(:,r) = project_conjugate_symmetry(A(:,r));
        B(:,r) = project_conjugate_symmetry(B(:,r));
        norm_a = norm(A(:,r));
        norm_b = norm(B(:,r));
        if ~isfinite(norm_a) || ~isfinite(norm_b) || ...
                norm_a <= 1e-14 || norm_b <= 1e-14
            ok = false;
            reason = sprintf('zero/nonfinite factor column %d', r);
            return;
        end
        A(:,r) = A(:,r)/norm_a;
        B(:,r) = B(:,r)/norm_b;
        W(:,r) = W(:,r)*(norm_a*norm_b);
    end
    if any(~isfinite(A(:))) || any(~isfinite(B(:))) || ...
            any(~isfinite(W(:)))
        ok = false;
        reason = 'nonfinite normalized factors';
    end
end

function x = project_conjugate_symmetry(x)
    x = 0.5 * (x + conj(flipud(x)));
    middle = (numel(x)+1)/2;
    x(middle) = real(x(middle));
end

function x = random_conjugate_symmetric_unit_vector(L)
    for attempt = 1:20
        x = project_conjugate_symmetry(randn(L,1) + 1i*randn(L,1));
        x_norm = norm(x);
        if isfinite(x_norm) && x_norm > 1e-14
            x = x/x_norm;
            return;
        end
    end
    x = zeros(L,1);
    x((L+1)/2) = 1;
end

function fit = finalize_cp_fit(C_tensor, A, B, W, R, m_values, ...
        n_values, phi_grid, iterations, best_start, converged, ...
        runtime_seconds, solver_status, objective_history)
    [A, B, W, ok, reason] = normalize_cp_columns(A, B, W);
    if ~ok
        error('Final CP normalization failed: %s', reason);
    end
    receiver_basis = exp(1i * phi_grid * m_values(:).');
    sender_basis = exp(1i * phi_grid * n_values(:).');

    % Resolve the two independent sign ambiguities deterministically.
    for r = 1:R
        a_values_r = receiver_basis * A(:,r);
        [~, peak_a] = max(abs(a_values_r));
        if real(a_values_r(peak_a)) < 0
            A(:,r) = -A(:,r);
            W(:,r) = -W(:,r);
        end
        b_values_r = sender_basis * B(:,r);
        [~, peak_b] = max(abs(b_values_r));
        if real(b_values_r(peak_b)) < 0
            B(:,r) = -B(:,r);
            W(:,r) = -W(:,r);
        end
    end

    % This is only a deterministic display order, not a component-wise
    % explained-energy ranking.
    component_weight_norm = vecnorm(W, 2, 1);
    [~, display_order] = sort(component_weight_norm, 'descend');
    A = A(:,display_order);
    B = B(:,display_order);
    W = W(:,display_order);
    component_weight_norm = component_weight_norm(display_order);

    a_values_complex = receiver_basis * A;
    b_values_complex = sender_basis * B;
    max_imag_a = max(abs(imag(a_values_complex)), [], 1);
    max_imag_b = max(abs(imag(b_values_complex)), [], 1);
    waveform_scale_a = max(1, max(abs(real(a_values_complex)), [], 1));
    waveform_scale_b = max(1, max(abs(real(b_values_complex)), [], 1));
    if any(max_imag_a > 1e-10*waveform_scale_a) || ...
            any(max_imag_b > 1e-10*waveform_scale_b)
        warning('Phase-domain CP factors have larger-than-expected imaginary parts.');
    end
    a_values = real(a_values_complex);
    b_values = real(b_values_complex);

    C_fit = reconstruct_cp_tensor(A, B, W, size(C_tensor));
    [objective, relative_residual, explained_fraction, ...
        per_direction_relative_residual, ...
        per_direction_explained_fraction] = ...
        compute_cp_metrics(C_tensor, C_fit);
    C_fit_check = complex(zeros(size(C_fit)));
    for p = 1:size(C_tensor,3)
        C_fit_check(:,:,p) = A * diag(W(p,:)) * B.';
    end
    reconstruction_check_error = norm(C_fit(:)-C_fit_check(:)) / ...
        max(norm(C_fit(:)), eps);
    direct_objective = sum(abs(C_tensor(:)-C_fit(:)).^2);
    objective_check_error = abs(objective-direct_objective);

    A_symmetry_error = zeros(1,R);
    B_symmetry_error = zeros(1,R);
    for r = 1:R
        A_symmetry_error(r) = norm(A(:,r)-conj(flipud(A(:,r)))) / ...
            max(norm(A(:,r)), eps);
        B_symmetry_error(r) = norm(B(:,r)-conj(flipud(B(:,r)))) / ...
            max(norm(B(:,r)), eps);
    end
    [~, ~, ~, weight_info] = update_component_weights(C_tensor, A, B);

    if isempty(objective_history)
        objective_history = objective;
    else
        objective_history = objective_history(:);
        objective_history(end) = objective;
    end

    fit = empty_final_fit();
    fit.rank = R;
    fit.A = A;
    fit.B = B;
    fit.W = real(W);
    fit.C_fit = C_fit;
    fit.a_values = a_values;
    fit.b_values = b_values;
    fit.a_values_complex = a_values_complex;
    fit.b_values_complex = b_values_complex;
    fit.objective = objective;
    fit.relative_residual = relative_residual;
    fit.explained_fraction = explained_fraction;
    fit.explained_percent = 100*explained_fraction;
    fit.per_direction_relative_residual = per_direction_relative_residual;
    fit.per_direction_explained_fraction = per_direction_explained_fraction;
    fit.iterations = iterations;
    fit.best_start = best_start;
    fit.converged = converged;
    fit.runtime_seconds = runtime_seconds;
    fit.solver_status = solver_status;
    fit.objective_history = objective_history;
    fit.component_weight_norm = component_weight_norm(:).';
    fit.component_display_order = display_order(:).';
    fit.A_conjugate_symmetry_error = A_symmetry_error;
    fit.B_conjugate_symmetry_error = B_symmetry_error;
    fit.max_imag_a_values = max_imag_a;
    fit.max_imag_b_values = max_imag_b;
    fit.reconstruction_check_error = reconstruction_check_error;
    fit.objective_check_error = objective_check_error;
    fit.normal_equation_residual = weight_info.normal_equation_residual;
end

function [objective, relative_residual, explained_fraction, ...
        per_direction_relative_residual, ...
        per_direction_explained_fraction] = ...
        compute_cp_metrics(C_tensor, C_fit)
    residual = C_tensor - C_fit;
    objective = sum(abs(residual(:)).^2);
    total_energy = sum(abs(C_tensor(:)).^2);
    relative_residual = sqrt(objective/total_energy);
    explained_fraction = 1 - objective/total_energy;
    P = size(C_tensor,3);
    per_direction_relative_residual = zeros(P,1);
    per_direction_explained_fraction = zeros(P,1);
    for p = 1:P
        direction_energy = sum(abs(C_tensor(:,:,p)).^2, 'all');
        direction_residual = sum(abs(residual(:,:,p)).^2, 'all');
        if direction_energy > 0
            per_direction_relative_residual(p) = ...
                sqrt(direction_residual/direction_energy);
            per_direction_explained_fraction(p) = ...
                1-direction_residual/direction_energy;
        elseif direction_residual == 0
            per_direction_relative_residual(p) = 0;
            per_direction_explained_fraction(p) = 1;
        else
            per_direction_relative_residual(p) = Inf;
            per_direction_explained_fraction(p) = NaN;
        end
    end
end

function comparison = compare_rank1_reference(reference, fits, C_tensor)
    comparison = struct('available', false, 'reason', reference.reason, ...
        'path', reference.path, 'reference_objective', NaN, ...
        'component_objective', NaN, 'objective_difference', NaN, ...
        'reference_explained_fraction', NaN, ...
        'component_explained_fraction', NaN, ...
        'explained_fraction_difference', NaN, ...
        'C_fit_relative_difference', NaN);
    if ~reference.available
        return;
    end
    index = find([fits.rank] == 1, 1);
    if isempty(index)
        comparison.reason = 'rank 1 was not requested';
        return;
    end
    fit = fits(index);
    residual_reference = C_tensor-reference.C_fit;
    reference_objective = sum(abs(residual_reference(:)).^2);
    energy = sum(abs(C_tensor(:)).^2);
    reference_explained = 1-reference_objective/energy;
    comparison.available = true;
    comparison.reason = 'same-tensor comparison available';
    comparison.reference_objective = reference_objective;
    comparison.component_objective = fit.objective;
    comparison.objective_difference = fit.objective-reference_objective;
    comparison.reference_explained_fraction = reference_explained;
    comparison.component_explained_fraction = fit.explained_fraction;
    comparison.explained_fraction_difference = ...
        fit.explained_fraction-reference_explained;
    comparison.C_fit_relative_difference = ...
        norm(fit.C_fit(:)-reference.C_fit(:)) / ...
        max(norm(reference.C_fit(:)), eps);
end

function comparison = compare_common_weight_results(common, fits, C_tensor)
    comparison = struct('available', false, 'reason', common.reason, ...
        'path', common.path, 'tensor_relative_mismatch', ...
        common.tensor_relative_mismatch, 'ranks', [], ...
        'common_objective', [], 'component_objective', [], ...
        'common_explained_fraction', [], ...
        'component_explained_fraction', [], ...
        'explained_fraction_difference', []);
    if ~common.available
        return;
    end
    energy = sum(abs(C_tensor(:)).^2);
    ranks = [];
    common_objective = [];
    component_objective = [];
    common_explained = [];
    component_explained = [];
    for k = 1:numel(fits)
        common_fit = find_common_fit(common.root, fits(k).rank);
        if isempty(common_fit) || ~isfield(common_fit, 'C_fit')
            continue;
        end
        residual_common = C_tensor-common_fit.C_fit;
        J_common = sum(abs(residual_common(:)).^2);
        tolerance = 1e-10*max(1,energy);
        if fits(k).objective > J_common+tolerance
            error(['Component-weight rank-%d fit is worse than its feasible ' ...
                'common-weight baseline.'], fits(k).rank);
        end
        ranks(end+1,1) = fits(k).rank; %#ok<AGROW>
        common_objective(end+1,1) = J_common; %#ok<AGROW>
        component_objective(end+1,1) = fits(k).objective; %#ok<AGROW>
        common_explained(end+1,1) = 1-J_common/energy; %#ok<AGROW>
        component_explained(end+1,1) = fits(k).explained_fraction; %#ok<AGROW>
    end
    if isempty(ranks)
        comparison.reason = 'no common ranks overlap requested ranks';
        return;
    end
    comparison.available = true;
    comparison.reason = 'same-tensor common-weight comparison available';
    comparison.ranks = ranks;
    comparison.common_objective = common_objective;
    comparison.component_objective = component_objective;
    comparison.common_explained_fraction = common_explained;
    comparison.component_explained_fraction = component_explained;
    comparison.explained_fraction_difference = ...
        component_explained-common_explained;
end

function validation = run_synthetic_validation(M, opts)
    if M < 2
        M = 2;
    end
    old_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(old_rng_state));
    rng(opts.RandomSeed, 'twister');
    L = 2*M+1;
    P = 12;
    [T, transform_info] = build_real_fourier_transform(M);
    [Qa,~] = qr(randn(L,3), 0);
    [Qb,~] = qr(randn(L,3), 0);
    [Qw,~] = qr(randn(P,3), 0);
    A_true = T*Qa;
    B_true = T*Qb;
    W_true = Qw*diag([3,2,1]);
    C_true = reconstruct_cp_tensor(A_true, B_true, W_true, [L,L,P]);

    test_opts = opts;
    test_opts.Ranks = 1:5;
    test_opts.SaveOutputs = false;
    test_opts.RunSyntheticValidation = false;
    test_opts.SyntheticTestOnly = false;
    empty_rank1 = empty_rank1_reference();
    empty_common = struct('available', false, 'reason', 'synthetic', ...
        'path', '', 'tensor_relative_mismatch', NaN, 'root', []);
    m_values = (-M:M).';
    phi_grid = linspace(0,2*pi,512).';
    rng(opts.RandomSeed, 'twister');
    [fits, diagnostics] = fit_rank_sequence(C_true, m_values, m_values, ...
        phi_grid, 1:5, test_opts, empty_rank1, empty_common);
    observed_explained = [fits.explained_fraction].';
    observed_relative_residual = [fits.relative_residual].';
    expected_explained = [9/14;13/14;1;1;1];
    energy = sum(abs(C_true(:)).^2);
    monotonic = all(diff(observed_explained) >= -1e-12);
    exact_from_rank3 = all(observed_relative_residual(3:5) < 1e-10);
    expected_match = max(abs(observed_explained-expected_explained)) < 1e-9;
    dimensions_valid = true;
    factors_valid = true;
    objectives_valid = true;
    reconstructions_valid = true;
    histories_valid = true;
    for k = 1:numel(fits)
        R = fits(k).rank;
        dimensions_valid = dimensions_valid && ...
            isequal(size(fits(k).A),[L,R]) && ...
            isequal(size(fits(k).B),[L,R]) && ...
            isequal(size(fits(k).W),[P,R]) && isreal(fits(k).W);
        factors_valid = factors_valid && ...
            max(abs(vecnorm(fits(k).A,2,1)-1)) < 1e-12 && ...
            max(abs(vecnorm(fits(k).B,2,1)-1)) < 1e-12 && ...
            max(fits(k).A_conjugate_symmetry_error) < 1e-12 && ...
            max(fits(k).B_conjugate_symmetry_error) < 1e-12;
        J_direct = sum(abs(C_true(:)-fits(k).C_fit(:)).^2);
        objectives_valid = objectives_valid && ...
            abs(fits(k).objective-J_direct) <= 1e-12*max(1,energy);
        C_check = reconstruct_cp_tensor(fits(k).A, fits(k).B, ...
            fits(k).W, size(C_true));
        reconstructions_valid = reconstructions_valid && ...
            norm(C_check(:)-fits(k).C_fit(:)) <= 1e-12*max(1,norm(C_true(:)));
        history_tolerance = 1e-12*max(1,energy);
        histories_valid = histories_valid && ...
            all(diff(fits(k).objective_history) <= history_tolerance);
    end

    % Same-tensor rank-1 comparison with the existing public solver.
    legacy = global_joint_svd_analysis_joint_cp_rank1([], M, ...
        'SyntheticTestOnly', true, 'NumStarts', opts.NumStarts, ...
        'MaxIter', opts.MaxIter, 'Tol', opts.Tol, ...
        'RandomSeed', opts.RandomSeed, 'DisplayFullArrays', false);
    legacy_reference = extract_rank1_reference(legacy, ...
        'legacy synthetic return value');
    rng(opts.RandomSeed, 'twister');
    [legacy_tensor_fit, ~] = fit_rank_sequence(legacy.C_tensor, ...
        (-M:M).', (-M:M).', phi_grid, 1, test_opts, ...
        legacy_reference, empty_common);
    legacy_residual = legacy.C_tensor-legacy.C_fit;
    legacy_objective = sum(abs(legacy_residual(:)).^2);
    legacy_energy = sum(abs(legacy.C_tensor(:)).^2);
    legacy_explained = 1 - legacy_objective / max(1, legacy_energy);
    rank1_tolerance = 1e-12*max(1,legacy_energy);
    rank1_consistent = legacy_tensor_fit.objective <= ...
        legacy_objective+rank1_tolerance && ...
        legacy_tensor_fit.explained_fraction >= ...
        legacy_explained-1e-12;

    wrong_transpose_fit = complex(zeros(size(C_true)));
    for p = 1:P
        wrong_transpose_fit(:,:,p) = ...
            A_true*diag(W_true(p,:))*B_true';
    end
    wrong_transpose_relative_error = ...
        norm(C_true(:)-wrong_transpose_fit(:))/norm(C_true(:));

    validation = struct();
    validation.ran = true;
    validation.passed = monotonic && exact_from_rank3 && expected_match && ...
        dimensions_valid && factors_valid && objectives_valid && ...
        reconstructions_valid && histories_valid && rank1_consistent && ...
        wrong_transpose_relative_error > 1e-3;
    validation.expected_explained_fraction = expected_explained;
    validation.observed_explained_fraction = observed_explained;
    validation.observed_relative_residual = observed_relative_residual;
    validation.monotonic = monotonic;
    validation.exact_from_rank3 = exact_from_rank3;
    validation.expected_match = expected_match;
    validation.dimensions_valid = dimensions_valid;
    validation.factors_valid = factors_valid;
    validation.objectives_valid = objectives_valid;
    validation.reconstructions_valid = reconstructions_valid;
    validation.objective_histories_nonincreasing = histories_valid;
    validation.rank1_legacy_consistent = rank1_consistent;
    validation.legacy_rank1_objective = legacy_objective;
    validation.component_rank1_objective = legacy_tensor_fit.objective;
    validation.wrong_transpose_relative_error = wrong_transpose_relative_error;
    validation.transform_info = transform_info;
    validation.A_true = A_true;
    validation.B_true = B_true;
    validation.W_true = W_true;
    validation.C_tensor = C_true;
    validation.fits = fits;
    validation.solver_diagnostics = diagnostics;

    fprintf('\n[SYNTHETIC] Component-weight rank-3 validation\n');
    for R = 1:5
        fprintf('  R=%d: explained=%.12f%%, relres=%.3e\n', R, ...
            100*observed_explained(R), observed_relative_residual(R));
    end
    fprintf('  Legacy/new rank-1 objectives: %.3e / %.3e\n', ...
        legacy_objective, legacy_tensor_fit.objective);
    fprintf('  Wrong B'' reconstruction relative error: %.3e\n', ...
        wrong_transpose_relative_error);
    fprintf('  Validation passed: %s\n', mat2str(validation.passed));
    clear rng_cleanup;
end

function [T, info] = build_real_fourier_transform(M)
    L = 2*M+1;
    modes = (-M:M).';
    phi_test = 2*pi*(0:L-1).'/L;
    E = exp(1i*phi_test*modes.');
    Psi = zeros(L,L);
    Psi(:,1) = 1;
    column = 2;
    for harmonic = 1:M
        Psi(:,column) = sqrt(2)*cos(harmonic*phi_test);
        Psi(:,column+1) = sqrt(2)*sin(harmonic*phi_test);
        column = column+2;
    end
    T = E\Psi;
    info = struct('basis_reconstruction_error', norm(E*T-Psi,'fro'), ...
        'unitarity_error', norm(T'*T-eye(L),'fro'));
end

function paths = make_output_paths(output_dir, maximum_rank)
    prefix = 'global_joint_cp_component_weights_rank_sweep';
    paths = struct();
    paths.summary_log = fullfile(output_dir, [prefix '_summary.txt']);
    paths.results_mat = fullfile(output_dir, [prefix '_results.mat']);
    paths.rank_summary_csv = fullfile(output_dir, [prefix '_rank_summary.csv']);
    paths.per_direction_csv = fullfile(output_dir, ...
        [prefix '_per_direction_metrics.csv']);
    paths.max_rank_weights_csv = fullfile(output_dir, sprintf( ...
        '%s_rank%d_component_weights.csv', prefix, maximum_rank));
    paths.common_comparison_csv = fullfile(output_dir, ...
        [prefix '_common_weight_comparison.csv']);
    paths.explained_png = fullfile(output_dir, ...
        [prefix '_explained_vs_rank.png']);
    paths.explained_fig = fullfile(output_dir, ...
        [prefix '_explained_vs_rank.fig']);
    paths.per_direction_png = fullfile(output_dir, ...
        [prefix '_per_direction_explained.png']);
    paths.per_direction_fig = fullfile(output_dir, ...
        [prefix '_per_direction_explained.fig']);
    paths.max_rank_weights_png = fullfile(output_dir, sprintf( ...
        '%s_rank%d_weights_heatmap.png', prefix, maximum_rank));
    paths.max_rank_weights_fig = fullfile(output_dir, sprintf( ...
        '%s_rank%d_weights_heatmap.fig', prefix, maximum_rank));
end

function save_rank_sweep_outputs(rank_sweep_results, opts)
    paths = rank_sweep_results.output_files;
    fits = rank_sweep_results.fits;
    ranks = rank_sweep_results.ranks;
    comparison = rank_sweep_results.common_weight_comparison;

    Rank = ranks;
    Objective = rank_sweep_results.objective;
    RelativeResidual = rank_sweep_results.relative_residual;
    ExplainedFraction = rank_sweep_results.explained_fraction;
    ExplainedPercent = rank_sweep_results.explained_percent;
    IncrementalExplainedFraction = ...
        rank_sweep_results.incremental_explained_fraction;
    IncrementalExplainedPercentagePoints = ...
        100*IncrementalExplainedFraction;
    RuntimeSeconds = rank_sweep_results.runtime_seconds;
    BestStart = [fits.best_start].';
    Iterations = [fits.iterations].';
    Converged = [fits.converged].';
    SolverStatus = string({fits.solver_status}.');
    summary_table = table(Rank, Objective, RelativeResidual, ...
        ExplainedFraction, ExplainedPercent, ...
        IncrementalExplainedFraction, ...
        IncrementalExplainedPercentagePoints, RuntimeSeconds, ...
        BestStart, Iterations, Converged, SolverStatus);
    writetable(summary_table, paths.rank_summary_csv);

    P = numel(rank_sweep_results.interaction_meta);
    number_rows = numel(ranks)*P;
    Rank = zeros(number_rows,1);
    DirectionIndex = zeros(number_rows,1);
    PairName = strings(number_rows,1);
    SourceID = zeros(number_rows,1);
    TargetID = zeros(number_rows,1);
    DirectionLabel = strings(number_rows,1);
    RelativeResidual = zeros(number_rows,1);
    ExplainedFraction = zeros(number_rows,1);
    ExplainedPercent = zeros(number_rows,1);
    row = 0;
    for k = 1:numel(ranks)
        for p = 1:P
            row = row+1;
            meta = rank_sweep_results.interaction_meta(p);
            Rank(row) = ranks(k);
            DirectionIndex(row) = p;
            PairName(row) = string(meta.pair_name);
            SourceID(row) = meta.source_id;
            TargetID(row) = meta.target_id;
            DirectionLabel(row) = sprintf('%d -> %d', ...
                meta.source_id, meta.target_id);
            RelativeResidual(row) = ...
                fits(k).per_direction_relative_residual(p);
            ExplainedFraction(row) = ...
                fits(k).per_direction_explained_fraction(p);
            ExplainedPercent(row) = 100*ExplainedFraction(row);
        end
    end
    direction_table = table(Rank, DirectionIndex, PairName, SourceID, ...
        TargetID, DirectionLabel, RelativeResidual, ...
        ExplainedFraction, ExplainedPercent);
    writetable(direction_table, paths.per_direction_csv);

    maximum_fit = fits(end);
    Rmax = maximum_fit.rank;
    number_rows = P*Rmax;
    Rank = repmat(Rmax, number_rows, 1);
    DirectionIndex = zeros(number_rows,1);
    PairName = strings(number_rows,1);
    SourceID = zeros(number_rows,1);
    TargetID = zeros(number_rows,1);
    DirectionLabel = strings(number_rows,1);
    Component = zeros(number_rows,1);
    Weight = zeros(number_rows,1);
    row = 0;
    for p = 1:P
        meta = rank_sweep_results.interaction_meta(p);
        for r = 1:Rmax
            row = row+1;
            DirectionIndex(row) = p;
            PairName(row) = string(meta.pair_name);
            SourceID(row) = meta.source_id;
            TargetID(row) = meta.target_id;
            DirectionLabel(row) = sprintf('%d -> %d', ...
                meta.source_id, meta.target_id);
            Component(row) = r;
            Weight(row) = maximum_fit.W(p,r);
        end
    end
    weight_table = table(Rank, DirectionIndex, PairName, SourceID, ...
        TargetID, DirectionLabel, Component, Weight);
    writetable(weight_table, paths.max_rank_weights_csv);

    if comparison.available
        Rank = comparison.ranks;
        ComponentObjective = comparison.component_objective;
        CommonObjective = comparison.common_objective;
        ComponentExplainedFraction = ...
            comparison.component_explained_fraction;
        CommonExplainedFraction = comparison.common_explained_fraction;
        ExplainedFractionDifference = ...
            comparison.explained_fraction_difference;
        ExplainedPercentagePointDifference = ...
            100*ExplainedFractionDifference;
        comparison_table = table(Rank, ComponentObjective, CommonObjective, ...
            ComponentExplainedFraction, CommonExplainedFraction, ...
            ExplainedFractionDifference, ExplainedPercentagePointDifference);
        writetable(comparison_table, paths.common_comparison_csv);
    end

    if opts.keep_figures
        figure_visibility = 'on';
    else
        figure_visibility = 'off';
    end

    fig_summary = figure('Color','w','Position',[100,100,900,760], ...
        'Visible',figure_visibility);
    tiledlayout(fig_summary,2,1,'TileSpacing','compact','Padding','compact');
    ax1 = nexttile;
    hold(ax1,'on');
    plot(ax1,ranks,rank_sweep_results.explained_percent,'-o', ...
        'LineWidth',2.2,'MarkerSize',7,'DisplayName','Component weights');
    for k = 1:numel(ranks)
        text(ax1,ranks(k),rank_sweep_results.explained_percent(k), ...
            sprintf(' %.3f',rank_sweep_results.explained_percent(k)), ...
            'VerticalAlignment','bottom','FontSize',8);
    end
    if comparison.available
        plot(ax1,comparison.ranks,100*comparison.common_explained_fraction, ...
            '--s','LineWidth',1.8,'MarkerSize',6, ...
            'DisplayName','Common weight');
        legend(ax1,'Location','southeast');
    end
    grid(ax1,'on'); box(ax1,'on');
    xlabel(ax1,'CP rank R');
    ylabel(ax1,'Explained energy (%)');
    title(ax1,'Tensor-energy explained percentage');
    xticks(ax1,ranks);

    ax2 = nexttile;
    if comparison.available && isequal(comparison.ranks(:),ranks(:))
        common_increment = [comparison.common_explained_fraction(1); ...
            diff(comparison.common_explained_fraction)];
        bar(ax2,ranks,100*[rank_sweep_results.incremental_explained_fraction, ...
            common_increment],'grouped');
        legend(ax2,{'Component weights','Common weight'},'Location','best');
    else
        bar(ax2,ranks,100*rank_sweep_results.incremental_explained_fraction);
    end
    grid(ax2,'on'); box(ax2,'on');
    xlabel(ax2,'CP rank R');
    ylabel(ax2,'Incremental explained energy (percentage points)');
    title(ax2,'Actual increment from the saved rank-(R-1) fit');
    xticks(ax2,ranks);
    saveas(fig_summary,paths.explained_png);
    savefig(fig_summary,paths.explained_fig);

    fig_direction = figure('Color','w','Position',[100,100,1050,650], ...
        'Visible',figure_visibility);
    ax = axes(fig_direction);
    hold(ax,'on');
    colors = lines(P);
    labels = cell(P,1);
    for p = 1:P
        plot(ax,ranks,100*rank_sweep_results.per_direction_explained_fraction(p,:), ...
            '-o','Color',colors(p,:),'LineWidth',1.3,'MarkerSize',4);
        labels{p} = sprintf('%d -> %d', ...
            rank_sweep_results.interaction_meta(p).source_id, ...
            rank_sweep_results.interaction_meta(p).target_id);
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'CP rank R');
    ylabel(ax,'Per-direction explained energy (%)');
    title(ax,{'Direction-wise diagnostics', ...
        'Main reported result uses total tensor energy, not a direction average'});
    xticks(ax,ranks);
    legend(ax,labels,'Location','eastoutside');
    saveas(fig_direction,paths.per_direction_png);
    savefig(fig_direction,paths.per_direction_fig);

    fig_weights = figure('Color','w','Position',[100,100,950,650], ...
        'Visible',figure_visibility);
    ax = axes(fig_weights);
    imagesc(ax,maximum_fit.W);
    colormap(ax,make_diverging_colormap(256));
    colorbar(ax);
    xlabel(ax,'Component (ordered by ||W(:,r)||_2)');
    ylabel(ax,'Directed interaction');
    xticks(ax,1:Rmax);
    yticks(ax,1:P);
    yticklabels(ax,labels);
    title(ax,{sprintf('Rank-%d component-specific direction weights',Rmax), ...
        'Component signs and permutations are intrinsically non-identifiable'});
    saveas(fig_weights,paths.max_rank_weights_png);
    savefig(fig_weights,paths.max_rank_weights_fig);

    save(paths.results_mat,'rank_sweep_results','-v7.3');
    if ~opts.keep_figures
        close(fig_summary);
        close(fig_direction);
        close(fig_weights);
    end
end

function cmap = make_diverging_colormap(number_colors)
    half = floor(number_colors/2);
    negative = [linspace(0,0.95,half).', ...
        linspace(0,0.95,half).', linspace(0.85,0.95,half).'];
    positive_count = number_colors-half;
    positive = [linspace(0.95,0.85,positive_count).', ...
        linspace(0.95,0,positive_count).', ...
        linspace(0.95,0,positive_count).'];
    cmap = [negative;positive];
end
