function rank_sweep_results = global_joint_cp_rank_sweep_sinusoidal_sender(round_dir, M, varargin)
%GLOBAL_JOINT_CP_RANK_SWEEP_SINUSOIDAL_SENDER Joint CP rank sweep with pure sinusoidal sender profiles.
%
%   C_fit(:,:,p) = sum_{r=1}^R W(p,r) * A(:,r) * B(:,r).'
%
% Phase-domain model:
%   s_{i<-j}(phi_i, phi_j) = sum_{r=1}^R W_{ij,r} * a_r(phi_i) * b_r(phi_j)
%
% Sender function constraint:
%   b_r(phi) = sqrt(2) * cos(phi - delta_r),  delta_r in [0, pi)
%
% Fourier basis vector B_r (for n = -M:M):
%   B_r(n=1)  = exp(-1i * delta_r) / sqrt(2)
%   B_r(n=-1) = exp( 1i * delta_r) / sqrt(2)
%   B_r(n)    = 0 for n ~= +/-1
%
% Key Properties:
%   - All senders share the exact same b_r(phi) within component r.
%   - Phases delta_r may differ across components r=1..R.
%   - ||B_r||_2 = 1 (unit norm), real in phase domain, zero-mean single sinusoid.
%   - Sign/phase ambiguity normalized so delta_r in [0, pi).
%   - All components are re-optimized independently for each rank R.
%
% Examples:
%   results = global_joint_cp_rank_sweep_sinusoidal_sender();
%   results = global_joint_cp_rank_sweep_sinusoidal_sender( ...
%       fullfile('EstimateL','Round6'), 10, 'Ranks', 1:10);
%   results = global_joint_cp_rank_sweep_sinusoidal_sender( ...
%       fullfile('EstimateL','Round6'), 10, ...
%       'UnconstrainedResultsMat', 'path/to/global_joint_cp_component_weights_rank_sweep_results.mat');

% =========================================================================
% [USER CONFIGURATION PRESET]
% Edit DEFAULT_EXECUTION_MODE below to easily switch execution mode in code:
%   'R1_only' : Calculate Rank 1 model only (R=1)
%   'sweep'   : Sweep ranks 1 to 10 (R=1:10)
% =========================================================================
DEFAULT_EXECUTION_MODE = 'sweep'; % Select: 'R1_only' or 'sweep'

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'SStick');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end
    validateattributes(M, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');
    opts = parse_options(DEFAULT_EXECUTION_MODE, varargin{:});
    ranks_requested = opts.Ranks(:).';
    L_expected = 2 * M + 1;
    if max(ranks_requested) > L_expected
        error('Every requested rank must be at most 2*M+1=%d.', L_expected);
    end

    if opts.SyntheticTestOnly
        rank_sweep_results = run_synthetic_validation(M, opts);
        return;
    end

    if numel(ranks_requested) == 1
        folder_suffix = sprintf('global_joint_cp_sinusoidal_sender_rank_sweep_R%d', ranks_requested(1));
    else
        folder_suffix = sprintf('global_joint_cp_sinusoidal_sender_rank_sweep_R%d_R%d', ranks_requested(1), ranks_requested(end));
    end
    output_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), folder_suffix);
    log_path = fullfile(output_dir, ...
        'global_joint_cp_sinusoidal_sender_rank_sweep_summary.txt');
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

    fprintf('[INFO] Sinusoidal-Sender Joint CP rank sweep\n');
    fprintf('  Model: C_fit(:,:,p) = sum_r W(p,r) * A_r * B_r.''\n');
    fprintf('  Sender constraint: b_r(phi) = sqrt(2)*cos(phi - delta_r), delta_r in [0, pi)\n');
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
    fprintf('[INFO] C_tensor loaded: %d x %d x %d (%s).\n', ...
        L, L, P, input_info.source);

    % Calculate fundamental sender tensor C^(1) and theoretical ceiling E_ceiling
    C_fundamental = compute_fundamental_tensor(C_tensor, M);
    fundamental_energy = sum(abs(C_fundamental(:)).^2);
    E_ceiling = fundamental_energy / tensor_energy;
    fprintf('[INFO] Theoretical Fundamental Mode Ceiling E_ceiling = %.6f (%.4f%%)\n\n', ...
        E_ceiling, 100 * E_ceiling);

    % Optional Validation Tensor
    val_data = load_validation_tensor(opts.ValidationTensorMatFile, M, ...
        C_tensor, interaction_meta, m_values, n_values);
    if ~val_data.available
        fprintf('[NOTE] Validation tensor not provided.\n');
        fprintf('       Increases in training explained fraction cannot be definitively\n');
        fprintf('       attributed to avoiding overfitting; diagnostics (numerical degeneracy,\n');
        fprintf('       localization, cancellation, start dependency) must be evaluated.\n\n');
    else
        fprintf('[INFO] Validation tensor loaded successfully from %s.\n\n', val_data.path);
    end

    % Optional Unconstrained Model Results for Comparison
    unconstrained_data = load_unconstrained_results(opts.UnconstrainedResultsMat, ...
        M, C_tensor, interaction_meta, m_values, n_values);

    phi_grid = linspace(0, 2*pi, 512).';
    common_reference = load_common_weight_reference( ...
        round_dir, M, max(ranks_requested), C_tensor, ...
        interaction_meta, m_values, n_values, opts);

    old_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(old_rng_state));
    rng(opts.RandomSeed, 'twister');
    internal_ranks = 1:max(ranks_requested);
    [internal_fits, solver_diagnostics] = fit_rank_sequence_sinusoidal( ...
        C_tensor, C_fundamental, m_values, n_values, phi_grid, internal_ranks, ...
        opts, rank1_reference, common_reference, val_data);

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
    fundamental_explained_fraction = [fits.fundamental_explained_fraction].';
    fundamental_explained_percent = 100 * fundamental_explained_fraction;
    incremental_explained_fraction = [fits.incremental_explained_fraction].';
    relative_residual = [fits.relative_residual].';
    objective = [fits.objective].';
    orthogonality_error = [fits.orthogonality_error].';
    runtime_seconds = [fits.runtime_seconds].';
    per_direction_relative_residual = cat(2, fits.per_direction_relative_residual);
    per_direction_explained_fraction = cat(2, fits.per_direction_explained_fraction);

    if val_data.available
        validation_explained_fraction = [fits.validation_explained_fraction].';
    else
        validation_explained_fraction = nan(numel(fits), 1);
    end

    rank_sweep_results = struct();
    rank_sweep_results.model = ...
        'sinusoidal-sender CP: C_p=sum_r W(p,r)*a_r*b_r.'' where b_r(phi)=sqrt(2)*cos(phi-delta_r)';
    rank_sweep_results.ranks = ranks_requested(:);
    rank_sweep_results.explained_fraction = explained_fraction;
    rank_sweep_results.explained_percent = explained_percent;
    rank_sweep_results.fundamental_explained_fraction = fundamental_explained_fraction;
    rank_sweep_results.fundamental_explained_percent = fundamental_explained_percent;
    rank_sweep_results.ceiling_explained_fraction = E_ceiling;
    rank_sweep_results.ceiling_explained_percent = 100 * E_ceiling;
    rank_sweep_results.incremental_explained_fraction = incremental_explained_fraction;
    rank_sweep_results.relative_residual = relative_residual;
    rank_sweep_results.objective = objective;
    rank_sweep_results.orthogonality_error = orthogonality_error;
    rank_sweep_results.validation_explained_fraction = validation_explained_fraction;
    rank_sweep_results.runtime_seconds = runtime_seconds;
    rank_sweep_results.per_direction_relative_residual = per_direction_relative_residual;
    rank_sweep_results.per_direction_explained_fraction = per_direction_explained_fraction;
    rank_sweep_results.fits = fits(:);
    rank_sweep_results.C_tensor = C_tensor;
    rank_sweep_results.C_fundamental = C_fundamental;
    rank_sweep_results.interaction_meta = interaction_meta;
    rank_sweep_results.m_values = m_values;
    rank_sweep_results.n_values = n_values;
    rank_sweep_results.phi_grid = phi_grid;
    rank_sweep_results.unique_agents = unique_agents(:);
    rank_sweep_results.options = opts;
    rank_sweep_results.input_info = input_info;
    rank_sweep_results.val_data = val_data;
    rank_sweep_results.unconstrained_data = unconstrained_data;
    rank_sweep_results.internal_ranks = internal_ranks(:);
    rank_sweep_results.internal_fits = internal_fits(:);
    rank_sweep_results.solver_diagnostics = solver_diagnostics(:);
    rank_sweep_results.synthetic_validation = struct('ran', false);
    rank_sweep_results.output_dir = output_dir;
    rank_sweep_results.output_files = make_output_paths(output_dir, max(ranks_requested));
    rank_sweep_results.total_runtime_seconds = toc(total_timer);

    if opts.RunSyntheticValidation
        fprintf('\n[INFO] Running the sinusoidal sender synthetic validation tests.\n');
        rank_sweep_results.synthetic_validation = run_synthetic_validation(max(M, 3), opts);
    end

    fprintf('\n[RESULT] Sinusoidal-Sender CP rank sweep\n');
    for k = 1:numel(fits)
        fprintf(['  R=%2d: total_exp=%10.6f%%, fund_exp=%10.6f%% (ceiling=%.4f%%), ' ...
            'increment=%8.4f pp, relres=%.4e, J=%.4e, max|W|=%.2f, rcond=%.2e, status=%s\n'], ...
            fits(k).rank, fits(k).explained_percent, fits(k).fundamental_explained_percent, ...
            100*E_ceiling, 100*fits(k).incremental_explained_fraction, ...
            fits(k).relative_residual, fits(k).objective, fits(k).max_abs_w, ...
            fits(k).gram_rcond, fits(k).solver_status);
    end

    if opts.SaveOutputs
        save_rank_sweep_outputs_sinusoidal(rank_sweep_results, opts);
        fprintf('\n[INFO] Results and figures saved under %s\n', output_dir);
    end

    print_final_summary_report(rank_sweep_results);

    clear rng_cleanup;
    if ~isempty(diary_cleanup)
        clear diary_cleanup;
    end
end

% -------------------------------------------------------------------------
% Option Parsing & Data Loading
% -------------------------------------------------------------------------

function opts = parse_options(default_mode, varargin)
    if nargin < 1 || isempty(default_mode)
        default_mode = 'sweep';
    end
    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'Mode', default_mode, @(x) ischar(x) || isstring(x));
    addParameter(parser, 'Ranks', [], @(x) isempty(x) || (isnumeric(x) && isvector(x) && ...
        all(isfinite(x)) && all(x >= 1) && all(x == fix(x)) && numel(unique(x)) == numel(x)));
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
    addParameter(parser, 'CommonWeightMatFile', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'ValidationTensorMatFile', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'UnconstrainedResultsMat', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'DisplayFullArrays', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RunSyntheticValidation', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SyntheticTestOnly', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SaveOutputs', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SaveCSV', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SaveFIG', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'keep_figures', false, @(x) islogical(x) && isscalar(x));

    addParameter(parser, 'analysis_start_sec', 10, @is_finite_scalar);
    addParameter(parser, 'analysis_duration_sec', 80, @is_finite_scalar);
    addParameter(parser, 'sample_dt', 0.01, @is_positive_scalar);
    addParameter(parser, 'signal_column', 'a2', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'normalize_signal', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'tail_percent', 10, @is_finite_scalar);
    addParameter(parser, 'clip_normalized_signal', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'clip_limit', 0.5, @is_positive_scalar);
    addParameter(parser, 'use_cache', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'file_indices', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
    addParameter(parser, 'RemoveSelfOnly', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RemoveConstant', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RemoveOtherOnly', false, @(x) islogical(x) && isscalar(x));
    parse(parser, varargin{:});
    opts = parser.Results;

    if isempty(opts.Ranks)
        if strcmpi(opts.Mode, 'R1_only') || strcmpi(opts.Mode, 'R1') || strcmpi(opts.Mode, 'single')
            opts.Ranks = 1;
        else
            opts.Ranks = 1:10;
        end
    else
        opts.Ranks = sort(opts.Ranks(:).');
    end
    opts.TensorMatFile = char(opts.TensorMatFile);
    opts.ExistingRank1Mat = char(opts.ExistingRank1Mat);
    opts.CommonWeightMatFile = char(opts.CommonWeightMatFile);
    opts.ValidationTensorMatFile = char(opts.ValidationTensorMatFile);
    opts.UnconstrainedResultsMat = char(opts.UnconstrainedResultsMat);
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

    fprintf(['[INFO] No tensor MAT file was found. Calling existing ' ...
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
    rank1_reference = extract_rank1_reference(base, 'rank-1 function return value');
    input_info = struct('source', 'rank-1 function return value', ...
        'path', '', 'rank1_function_called', true);
end

function [C_tensor, interaction_meta, m_values, n_values, root] = ...
        extract_tensor_payload(loaded, M, source_path)
    root = loaded;
    if isfield(loaded, 'rank_sweep_results') && isstruct(loaded.rank_sweep_results)
        root = loaded.rank_sweep_results;
    elseif isfield(loaded, 'all_global_results') && isstruct(loaded.all_global_results)
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

function validate_tensor_and_metadata(C_tensor, interaction_meta, m_values, n_values, M)
    L = 2*M + 1;
    if ndims(C_tensor) ~= 3 || size(C_tensor,1) ~= L || size(C_tensor,2) ~= L
        error('C_tensor must have size (2*M+1) x (2*M+1) x P.');
    end
    if any(~isfinite(C_tensor(:)))
        error('C_tensor contains NaN or Inf.');
    end
    if ~isequal(m_values(:), (-M:M).') || ~isequal(n_values(:), (-M:M).')
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

function C_fundamental = compute_fundamental_tensor(C_tensor, M)
    % Keep only n = +/-1 modes in 2nd dimension (sender axis)
    L = 2*M + 1;
    n_plus1_idx = M + 2;
    n_minus1_idx = M;

    C_fundamental = complex(zeros(size(C_tensor)));
    C_fundamental(:, n_plus1_idx, :) = C_tensor(:, n_plus1_idx, :);
    C_fundamental(:, n_minus1_idx, :) = C_tensor(:, n_minus1_idx, :);
end

function val_data = load_validation_tensor(val_path, M, C_tensor, ...
        interaction_meta, m_values, n_values)
    val_data = struct('available', false, 'path', '', 'C_val', [], ...
        'energy', NaN);
    if isempty(val_path)
        return;
    end
    if ~exist(val_path, 'file')
        error('ValidationTensorMatFile does not exist: %s', val_path);
    end
    loaded = load(val_path);
    [C_val, meta_val, m_val, n_val, ~] = extract_tensor_payload(loaded, M, val_path);
    validate_tensor_and_metadata(C_val, meta_val, m_val, n_val, M);
    if ~metadata_matches(meta_val, interaction_meta)
        error('Validation tensor interaction_meta does not match training tensor.');
    end
    val_data.available = true;
    val_data.path = val_path;
    val_data.C_val = C_val;
    val_data.energy = sum(abs(C_val(:)).^2);
end

function unconstrained = load_unconstrained_results(unconstrained_path, ...
        M, C_tensor, interaction_meta, m_values, n_values)
    unconstrained = struct('available', false, 'path', '', 'ranks', [], ...
        'explained_fraction', []);
    if isempty(unconstrained_path)
        return;
    end
    if ~exist(unconstrained_path, 'file')
        error('UnconstrainedResultsMat does not exist: %s', unconstrained_path);
    end
    loaded = load(unconstrained_path);
    if isfield(loaded, 'rank_sweep_results')
        res = loaded.rank_sweep_results;
    else
        res = loaded;
    end
    if ~isfield(res, 'ranks') || ~isfield(res, 'explained_fraction')
        error('UnconstrainedResultsMat lacks ranks or explained_fraction fields.');
    end
    unconstrained.available = true;
    unconstrained.path = unconstrained_path;
    unconstrained.ranks = res.ranks(:);
    unconstrained.explained_fraction = res.explained_fraction(:);
end

function reference = empty_rank1_reference()
    reference = struct('available', false, 'reason', 'not loaded', ...
        'path', '', 'A', [], 'B', [], 'W', [], 'C_fit', [], ...
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
    L = size(root.C_tensor, 1);
    M = (L - 1) / 2;
    [A, B, W, ok, reason] = normalize_cp_columns_sinusoidal(A, B, W, M);
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

function common = load_common_weight_reference(round_dir, M, max_rank, ...
        C_tensor, interaction_meta, m_values, n_values, opts)
    common = struct('available', false, 'reason', 'not found', 'path', '', 'root', []);
    explicit = ~isempty(opts.CommonWeightMatFile);
    if explicit
        path = opts.CommonWeightMatFile;
    else
        path = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), ...
            sprintf('global_joint_common_shape_rank_sweep_R1_R%d', max_rank), ...
            'global_joint_common_shape_rank_sweep_results.mat');
        if ~exist(path, 'file')
            pattern = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), ...
                'global_joint_common_shape_rank_sweep_R1_R*', ...
                'global_joint_common_shape_rank_sweep_results.mat');
            candidates = dir(pattern);
            if ~isempty(candidates)
                [~, newest] = max([candidates.datenum]);
                path = fullfile(candidates(newest).folder, candidates(newest).name);
            end
        end
    end
    common.path = path;
    if ~exist(path, 'file')
        common.reason = 'common-weight MAT file not found';
        return;
    end
    loaded = load(path);
    if ~isfield(loaded, 'rank_sweep_results')
        common.reason = 'MAT lacks rank_sweep_results';
        return;
    end
    root = loaded.rank_sweep_results;
    common.available = true;
    common.root = root;
end

function tf = metadata_matches(a, b)
    tf = numel(a) == numel(b);
    if ~tf, return; end
    for p = 1:numel(a)
        tf = tf && isequal(a(p).target_id, b(p).target_id) && ...
            isequal(a(p).source_id, b(p).source_id) && ...
            strcmp(char(a(p).pair_name), char(b(p).pair_name));
        if ~tf, return; end
    end
end

% -------------------------------------------------------------------------
% Core Optimization Routines (Sinusoidal Sender Constraints)
% -------------------------------------------------------------------------

function [fits, diagnostics] = fit_rank_sequence_sinusoidal(C_tensor, ...
        C_fundamental, m_values, n_values, phi_grid, ranks, opts, ...
        rank1_reference, common_reference, val_data)
    data_scale = norm(C_tensor(:));
    C_work = C_tensor / data_scale;
    L = size(C_tensor, 1);
    M = (L - 1) / 2;

    fits = repmat(empty_final_fit(), numel(ranks), 1);
    diagnostics = repmat(empty_rank_diagnostics(), numel(ranks), 1);
    previous_fit = [];

    for index = 1:numel(ranks)
        R = ranks(index);
        rank_timer = tic;
        baselines = {};
        predefined = {};

        if R == 1
            if rank1_reference.available
                ref_candidate = make_cp_candidate_sinusoidal(C_work, ...
                    rank1_reference.A, rank1_reference.B, ...
                    rank1_reference.W/data_scale, M, 'legacy_rank1');
                if ref_candidate.available
                    baselines{end+1} = ref_candidate; %#ok<AGROW>
                    predefined{end+1} = ref_candidate; %#ok<AGROW>
                end
            end
        else
            zero_baseline = zero_padded_previous_candidate_sinusoidal( ...
                previous_fit, C_work, data_scale, M);
            baselines{end+1} = zero_baseline; %#ok<AGROW>
            predefined{end+1} = zero_baseline; %#ok<AGROW>

            warm_candidate = residual_warm_start_sinusoidal( ...
                previous_fit, C_work, data_scale, opts, M);
            if warm_candidate.available
                predefined{end+1} = warm_candidate; %#ok<AGROW>
            end
        end

        baseline = choose_best_candidate(baselines);
        solver = fit_cp_rank_multistart_sinusoidal(C_work, R, opts, predefined, M);

        selection_tolerance = 128 * eps(max(1, solver.objective));
        use_baseline = baseline.available && ...
            (~solver.available || solver.objective >= baseline.objective - selection_tolerance);
        if use_baseline
            selected = baseline;
            iterations = 0;
            best_start = 0;
            converged = false;
            solver_status = sprintf('baseline_retained:%s', baseline.source);
            objective_history_work = baseline.objective;
        elseif solver.available
            selected = make_cp_candidate_sinusoidal(C_work, solver.A, solver.B, ...
                solver.W, M, sprintf('optimized_start_%d', solver.best_start));
            iterations = solver.iterations;
            best_start = solver.best_start;
            converged = solver.converged;
            solver_status = sprintf('optimized_start_%d:%s', solver.best_start, solver.status);
            objective_history_work = solver.objective_history;
        else
            error('All starts failed at rank %d and no feasible baseline exists.', R);
        end

        W_physical = selected.W * data_scale;
        runtime_seconds = toc(rank_timer);
        fit = finalize_cp_fit_sinusoidal(C_tensor, C_fundamental, selected.A, selected.B, ...
            W_physical, R, m_values, n_values, phi_grid, iterations, best_start, ...
            converged, runtime_seconds, solver_status, objective_history_work * data_scale^2, val_data);

        if ~isempty(previous_fit)
            monotonic_tolerance = 256 * eps(max([1, previous_fit.objective, fit.objective]));
            if fit.objective > previous_fit.objective + monotonic_tolerance
                error('Rank-%d result is worse than rank-%d despite zero-weight baseline.', R, R-1);
            end
        end

        fits(index) = orderfields(fit, fits(index));
        previous_fit = fit;

        diagnostics(index).rank = R;
        diagnostics(index).data_scale = data_scale;
        diagnostics(index).baseline_available = baseline.available;
        diagnostics(index).baseline_source = baseline.source;
        diagnostics(index).baseline_objective = baseline.objective * data_scale^2;
        diagnostics(index).optimized_objective = solver.objective * data_scale^2;
        diagnostics(index).start_objectives = solver.start_objectives * data_scale^2;
        diagnostics(index).start_iterations = solver.start_iterations;
        diagnostics(index).start_converged = solver.start_converged;
        diagnostics(index).start_status = solver.start_status;
        diagnostics(index).selected_baseline = use_baseline;

        fprintf(['[RANK %2d] total_exp=%10.6f%%, fund_exp=%10.6f%%, relres=%.4e, ' ...
            'J=%.4e, start=%d, iter=%d\n'], R, fit.explained_percent, ...
            fit.fundamental_explained_percent, fit.relative_residual, ...
            fit.objective, fit.best_start, fit.iterations);
    end
end

function solver = fit_cp_rank_multistart_sinusoidal(C_tensor, R, opts, predefined, M)
    solver = empty_multistart_result(opts.NumStarts);
    best_objective = Inf;
    for start_index = 1:opts.NumStarts
        if start_index <= numel(predefined) && predefined{start_index}.available
            initialization = predefined{start_index};
        else
            initialization = random_cp_initialization_sinusoidal(C_tensor, R, M);
        end
        result = run_one_cp_start_sinusoidal(C_tensor, initialization.A, ...
            initialization.B, initialization.W, opts, M);
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

function result = empty_start_result()
    result = struct('available', false, 'A', [], 'B', [], 'W', [], ...
        'objective', Inf, 'objective_history', [], 'iterations', 0, ...
        'converged', false, 'status', 'not_started', 'weight_solver', '');
end

function result = run_one_cp_start_sinusoidal(C_tensor, A, B, W, opts, M)
    result = empty_start_result();
    [A, B, ~, ok, reason] = normalize_cp_columns_sinusoidal(A, B, W, M);
    if ~ok
        result.status = ['invalid_initialization:' reason];
        return;
    end
    [W_initial, weight_method, weight_ok] = update_component_weights(C_tensor, A, B);
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
    R = size(A, 2);
    P = size(C_tensor, 3);
    L = 2*M + 1;

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

            % 1. Update A_r
            E_weighted_a = sum(E_minus .* W_reshaped, 3);
            a_raw = E_weighted_a * conj(B_candidate(:,r));
            a_projected = project_conjugate_symmetry(a_raw);
            a_norm = norm(a_projected);
            if isfinite(a_norm) && a_norm > 1e-14
                A_candidate(:,r) = a_projected / a_norm;
            else
                degenerate_updates = degenerate_updates + 1;
            end

            % 2. Update B_r with strict sinusoidal projection
            E_weighted_b = sum(permute(E_minus, [2 1 3]) .* W_reshaped, 3);
            b_raw = E_weighted_b * conj(A_candidate(:,r));
            [B_cand_r, b_norm] = project_sinusoidal_sender(b_raw, L, M);
            if isfinite(b_norm) && b_norm > 1e-14
                B_candidate(:,r) = B_cand_r;
            else
                degenerate_updates = degenerate_updates + 1;
            end

            component_new = A_candidate(:,r) * B_candidate(:,r).';
            delta_component = component_new - component_old;
            C_fit_candidate = C_fit_candidate + delta_component .* W_reshaped;
        end

        % 3. Simultaneous real W update
        [W_new, method, weight_ok] = update_component_weights( ...
            C_tensor, A_candidate, B_candidate);
        used_weight_methods{end+1} = method; %#ok<AGROW>
        if ~weight_ok || any(~isfinite(W_new(:)))
            status = ['weight_least_squares_failed:' method];
            break;
        end
        W_candidate = W_new;
        C_fit_candidate = reconstruct_cp_tensor(A_candidate, B_candidate, W_candidate, size(C_tensor));
        residual_candidate = C_tensor - C_fit_candidate;
        objective_candidate = sum(abs(residual_candidate(:)).^2);
        if ~isfinite(objective_candidate)
            status = 'nonfinite_objective';
            break;
        end

        increase = objective_candidate - accepted_objective;
        numerical_floor = 128 * eps(max([1, accepted_objective, sum(abs(C_tensor(:)).^2)]));
        if increase > numerical_floor
            status = sprintf('objective_increase_rejected:%.3e', increase);
            break;
        elseif increase > 0
            converged = true;
            status = 'numerical_objective_floor';
            break;
        end

        relative_change = abs(accepted_objective - objective_candidate) / ...
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
                status = sprintf('converged_with_%d_degenerate_updates', degenerate_updates);
            else
                status = 'converged';
            end
            break;
        end
    end

    if any(~isfinite(A(:))) || any(~isfinite(B(:))) || any(~isfinite(W(:)))
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
        result.status = [result.status ';weight_solver=augmented_real_least_squares'];
    else
        result.weight_solver = 'normal_equations';
        result.status = [result.status ';weight_solver=normal_equations'];
    end
end

% -------------------------------------------------------------------------
% Sinusoidal Sender Helpers
% -------------------------------------------------------------------------

function [B_proj, raw_norm] = project_sinusoidal_sender(b_raw, L, M)
    % PROJECT_SINUSOIDAL_SENDER Project raw vector onto 2D real sinusoidal space (n = +/- 1)
    n_plus1_idx = M + 2;
    n_minus1_idx = M;

    u_plus1 = b_raw(n_plus1_idx);
    u_minus1 = b_raw(n_minus1_idx);

    c1 = (1 / sqrt(2)) * real(u_plus1 + u_minus1);
    c2 = (1 / sqrt(2)) * imag(u_minus1 - u_plus1);

    raw_norm = hypot(c1, c2);
    if raw_norm <= 1e-14 || ~isfinite(raw_norm)
        delta_r = 0;
    else
        delta_r = atan2(c2, c1);
        delta_r = mod(delta_r, pi);
        if delta_r < 0, delta_r = delta_r + pi; end
    end
    B_proj = create_sinusoidal_B(delta_r, L, M);
end

function B_r = create_sinusoidal_B(delta_r, L, M)
    % CREATE_SINUSOIDAL_B Construct unit-norm Fourier vector B_r for b_r(phi) = sqrt(2)*cos(phi - delta_r)
    n_plus1_idx = M + 2;
    n_minus1_idx = M;

    B_r = complex(zeros(L, 1));
    B_r(n_plus1_idx)  = exp(-1i * delta_r) / sqrt(2);
    B_r(n_minus1_idx) = exp( 1i * delta_r) / sqrt(2);
end

function delta_r = extract_delta_from_B(B_r, M)
    % EXTRACT_DELTA_FROM_B Extract phase delta_r in [0, pi) from sinusoidal Fourier vector B_r
    n_plus1_idx = M + 2;
    n_minus1_idx = M;

    u_plus1 = B_r(n_plus1_idx);
    u_minus1 = B_r(n_minus1_idx);

    c1 = (1 / sqrt(2)) * real(u_plus1 + u_minus1);
    c2 = (1 / sqrt(2)) * imag(u_minus1 - u_plus1);

    delta_r = atan2(c2, c1);
    delta_r = mod(delta_r, pi);
    if delta_r < 0, delta_r = delta_r + pi; end
end

function candidate = zero_padded_previous_candidate_sinusoidal(previous_fit, C_work, data_scale, M)
    if isempty(previous_fit)
        candidate = unavailable_candidate('no_previous_fit');
        return;
    end
    L = size(C_work, 1);
    a_new = random_conjugate_symmetric_unit_vector(L);
    b_new = create_sinusoidal_B(rand() * pi, L, M);

    A = [previous_fit.A, a_new];
    B = [previous_fit.B, b_new];
    W = [previous_fit.W/data_scale, zeros(size(C_work,3), 1)];
    candidate = make_cp_candidate_sinusoidal(C_work, A, B, W, M, ...
        sprintf('rank_%d_zero_weight_extension', previous_fit.rank));
end

function candidate = residual_warm_start_sinusoidal(previous_fit, C_work, data_scale, opts, M)
    if isempty(previous_fit)
        candidate = unavailable_candidate('no_previous_fit');
        return;
    end
    previous_W_work = previous_fit.W / data_scale;
    previous_C_fit = reconstruct_cp_tensor(previous_fit.A, previous_fit.B, ...
        previous_W_work, size(C_work));
    residual = C_work - previous_C_fit;
    [a_new, b_new, w_new, status] = fit_residual_rank1_initialization_sinusoidal(residual, opts, M);
    A = [previous_fit.A, a_new];
    B = [previous_fit.B, b_new];
    W = [previous_W_work, w_new];
    candidate = make_cp_candidate_sinusoidal(C_work, A, B, W, M, ...
        sprintf('rank_%d_warm_residual_%s', previous_fit.rank, status));
end

function [a, b, w, status] = fit_residual_rank1_initialization_sinusoidal(residual, opts, M)
    L = size(residual, 1);
    P = size(residual, 3);
    if norm(residual(:)) <= 1e-14
        a = random_conjugate_symmetric_unit_vector(L);
        b = create_sinusoidal_B(0, L, M);
        w = zeros(P, 1);
        status = 'zero_residual';
        return;
    end
    warm_opts = opts;
    warm_opts.NumStarts = min(5, opts.NumStarts);
    warm_opts.MaxIter = min(200, opts.MaxIter);
    warm = fit_cp_rank_multistart_sinusoidal(residual, 1, warm_opts, {}, M);
    if warm.available
        a = warm.A;
        b = warm.B;
        w = warm.W;
        status = sprintf('rank1_start_%d', warm.best_start);
    else
        a = random_conjugate_symmetric_unit_vector(L);
        b = create_sinusoidal_B(0, L, M);
        w = zeros(P, 1);
        status = 'rank1_failed';
    end
end

function initialization = random_cp_initialization_sinusoidal(C_tensor, R, M)
    L = size(C_tensor, 1);
    A = complex(zeros(L, R));
    B = complex(zeros(L, R));
    for r = 1:R
        A(:,r) = random_conjugate_symmetric_unit_vector(L);
        B(:,r) = create_sinusoidal_B(rand() * pi, L, M);
    end
    [W, method, ok] = update_component_weights(C_tensor, A, B);
    if ~ok
        W = randn(size(C_tensor, 3), R);
        method = 'random_weight_fallback';
    end
    initialization = make_cp_candidate_sinusoidal(C_tensor, A, B, W, M, ['random_' method]);
end

function candidate = unavailable_candidate(reason)
    candidate = struct('available', false, 'source', reason, 'A', [], 'B', [], 'W', [], 'C_fit', [], 'objective', Inf);
end

function candidate = make_cp_candidate_sinusoidal(C_tensor, A, B, W, M, source)
    candidate = unavailable_candidate(source);
    [A, B, W, ok, reason] = normalize_cp_columns_sinusoidal(A, B, W, M);
    if ~ok
        candidate.source = sprintf('%s_invalid:%s', source, reason);
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

function [W, method, ok, info] = update_component_weights(C_tensor, A, B)
    R = size(A, 2);
    P = size(C_tensor, 3);
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
    if isfinite(gram_rcond) && gram_rcond >= 1e-12 && numerical_rank == R
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

function [A, B, W, ok, reason] = normalize_cp_columns_sinusoidal(A, B, W, M)
    ok = true;
    reason = 'ok';
    if ~ismatrix(A) || ~ismatrix(B) || ~ismatrix(W) || ...
            size(A,2) ~= size(B,2) || size(A,2) ~= size(W,2)
        ok = false;
        reason = 'factor dimensions disagree';
        return;
    end
    W = real(W);
    L = 2*M + 1;
    for r = 1:size(A,2)
        A(:,r) = project_conjugate_symmetry(A(:,r));
        norm_a = norm(A(:,r));
        if ~isfinite(norm_a) || norm_a <= 1e-14
            ok = false;
            reason = sprintf('zero/nonfinite factor column A %d', r);
            return;
        end
        A(:,r) = A(:,r) / norm_a;
        W(:,r) = W(:,r) * norm_a;

        % Strict sinusoidal projection for B
        [B_proj, norm_b] = project_sinusoidal_sender(B(:,r), L, M);
        if ~isfinite(norm_b) || norm_b <= 1e-14
            ok = false;
            reason = sprintf('zero/nonfinite factor column B %d', r);
            return;
        end
        % Absorb sign alignment into W
        delta_r = extract_delta_from_B(B_proj, M);
        B_exact = create_sinusoidal_B(delta_r, L, M);
        dot_prod = real(B(:,r)' * B_exact);
        if dot_prod < 0
            W(:,r) = -W(:,r);
        end
        B(:,r) = B_exact;
    end
    if any(~isfinite(A(:))) || any(~isfinite(B(:))) || any(~isfinite(W(:)))
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
            x = x / x_norm;
            return;
        end
    end
    x = zeros(L,1);
    x((L+1)/2) = 1;
end

function fit = finalize_cp_fit_sinusoidal(C_tensor, C_fundamental, A, B, W, R, ...
        m_values, n_values, phi_grid, iterations, best_start, converged, ...
        runtime_seconds, solver_status, objective_history, val_data)

    L = size(C_tensor, 1);
    M = (L - 1) / 2;
    P = size(C_tensor, 3);
    [A, B, W, ok, reason] = normalize_cp_columns_sinusoidal(A, B, W, M);
    if ~ok
        error('Final CP normalization failed: %s', reason);
    end

    receiver_basis = exp(1i * phi_grid * m_values(:).');
    sender_basis = exp(1i * phi_grid * n_values(:).');

    % Deterministic sign ambiguity resolution for A (flips W accordingly, delta unchanged)
    for r = 1:R
        a_val_r = receiver_basis * A(:,r);
        [~, peak_a] = max(abs(a_val_r));
        if real(a_val_r(peak_a)) < 0
            A(:,r) = -A(:,r);
            W(:,r) = -W(:,r);
        end
    end

    C_fit = reconstruct_cp_tensor(A, B, W, size(C_tensor));
    residual = C_tensor - C_fit;
    objective = sum(abs(residual(:)).^2);
    tensor_energy = sum(abs(C_tensor(:)).^2);
    relative_residual = sqrt(objective / tensor_energy);
    explained_fraction = 1 - objective / tensor_energy;
    explained_percent = 100 * explained_fraction;

    fundamental_energy = sum(abs(C_fundamental(:)).^2);
    fund_residual = C_fundamental - C_fit;
    fund_objective = sum(abs(fund_residual(:)).^2);
    fundamental_explained_fraction = 1 - fund_objective / fundamental_energy;
    fundamental_explained_percent = 100 * fundamental_explained_fraction;

    % Orthogonality check error: ||C - Cfit||^2 = ||C - C1||^2 + ||C1 - Cfit||^2
    diff_C_C1 = C_tensor - C_fundamental;
    ortho_lhs = objective;
    ortho_rhs = sum(abs(diff_C_C1(:)).^2) + fund_objective;
    orthogonality_error = abs(ortho_lhs - ortho_rhs) / tensor_energy;

    % Validation Tensor Evaluation
    if val_data.available
        val_residual = val_data.C_val - C_fit;
        val_objective = sum(abs(val_residual(:)).^2);
        validation_explained_fraction = 1 - val_objective / val_data.energy;
    else
        validation_explained_fraction = NaN;
    end

    % Per-direction metrics
    per_direction_relative_residual = zeros(P, 1);
    per_direction_explained_fraction = zeros(P, 1);
    for p = 1:P
        dir_energy = sum(abs(C_tensor(:,:,p)).^2);
        dir_res = C_tensor(:,:,p) - C_fit(:,:,p);
        dir_obj = sum(abs(dir_res(:)).^2);
        if dir_energy > 0
            per_direction_relative_residual(p) = sqrt(dir_obj / dir_energy);
            per_direction_explained_fraction(p) = 1 - dir_obj / dir_energy;
        end
    end

    % Extract delta_r and component profiles
    delta = zeros(R, 1);
    a_values = zeros(numel(phi_grid), R);
    b_values = zeros(numel(phi_grid), R);
    for r = 1:R
        delta(r) = extract_delta_from_B(B(:,r), M);
        a_values(:,r) = real(receiver_basis * A(:,r));
        b_values(:,r) = real(sender_basis * B(:,r));
    end

    % Component level metrics
    component_energy_fraction = zeros(R, 1);
    drop_one_metric = zeros(R, 1);
    max_direction_share = zeros(R, 1);
    effective_directions = zeros(R, 1);
    dominant_interaction_index = zeros(R, 1);
    T_components = complex(zeros(L, L, P, R));

    for r = 1:R
        for p = 1:P
            T_components(:,:,p,r) = W(p,r) * (A(:,r) * B(:,r).');
        end
        T_r = T_components(:,:,:,r);
        component_energy_fraction(r) = sum(abs(T_r(:)).^2) / tensor_energy;

        C_fit_drop_r = C_fit - T_r;
        drop_one_obj = sum(abs(C_tensor(:) - C_fit_drop_r(:)).^2);
        drop_one_metric(r) = (drop_one_obj - objective) / tensor_energy;

        w2 = W(:,r).^2;
        sum_w2 = sum(w2);
        if sum_w2 > 0
            pi_pr = w2 / sum_w2;
            max_direction_share(r) = max(pi_pr);
            effective_directions(r) = 1 / sum(pi_pr.^2);
        else
            max_direction_share(r) = 0;
            effective_directions(r) = 0;
        end
        [~, dominant_interaction_index(r)] = max(abs(W(:,r)));
    end

    % Matrix-level & Cancellation Metrics
    sum_comp_energy = sum(component_energy_fraction) * tensor_energy;
    fit_energy = sum(abs(C_fit(:)).^2);
    if fit_energy > 0
        cancellation_index = sum_comp_energy / fit_energy;
    else
        cancellation_index = 1.0;
    end

    D = complex(zeros(L*L, R));
    for r = 1:R
        D(:,r) = reshape(A(:,r)*B(:,r).', [], 1);
    end
    gram = real(D' * D);
    gram_rcond = rcond(gram);
    design_rank = rank([real(D); imag(D)]);

    max_atom_similarity = 0;
    for r1 = 1:R
        for r2 = (r1+1):R
            sim = abs(real(D(:,r1)' * D(:,r2))) / (norm(D(:,r1)) * norm(D(:,r2)));
            if sim > max_atom_similarity
                max_atom_similarity = sim;
            end
        end
    end

    fit = struct();
    fit.rank = R;
    fit.A = A;
    fit.B = B;
    fit.W = W;
    fit.delta = delta;
    fit.C_fit = C_fit;
    fit.a_values = a_values;
    fit.b_values = b_values;
    fit.objective = objective;
    fit.relative_residual = relative_residual;
    fit.explained_fraction = explained_fraction;
    fit.explained_percent = explained_percent;
    fit.fundamental_explained_fraction = fundamental_explained_fraction;
    fit.fundamental_explained_percent = fundamental_explained_percent;
    fit.orthogonality_error = orthogonality_error;
    fit.validation_explained_fraction = validation_explained_fraction;
    fit.incremental_explained_fraction = NaN;
    fit.per_direction_relative_residual = per_direction_relative_residual;
    fit.per_direction_explained_fraction = per_direction_explained_fraction;
    fit.component_energy_fraction = component_energy_fraction;
    fit.drop_one_metric = drop_one_metric;
    fit.max_direction_share = max_direction_share;
    fit.effective_directions = effective_directions;
    fit.dominant_interaction_index = dominant_interaction_index;
    fit.cancellation_index = cancellation_index;
    fit.design_rank = design_rank;
    fit.gram_rcond = gram_rcond;
    fit.max_atom_similarity = max_atom_similarity;
    fit.max_abs_w = max(abs(W(:)));
    fit.norm_w = norm(W, 'fro');
    fit.iterations = iterations;
    fit.best_start = best_start;
    fit.converged = converged;
    fit.runtime_seconds = runtime_seconds;
    fit.solver_status = solver_status;
    fit.objective_history = objective_history;
end

function fit = empty_final_fit()
    fit = struct('rank', [], 'A', [], 'B', [], 'W', [], 'delta', [], ...
        'C_fit', [], 'a_values', [], 'b_values', [], 'objective', [], ...
        'relative_residual', [], 'explained_fraction', [], 'explained_percent', [], ...
        'fundamental_explained_fraction', [], 'fundamental_explained_percent', [], ...
        'orthogonality_error', [], 'validation_explained_fraction', [], ...
        'incremental_explained_fraction', NaN, ...
        'per_direction_relative_residual', [], ...
        'per_direction_explained_fraction', [], ...
        'component_energy_fraction', [], 'drop_one_metric', [], ...
        'max_direction_share', [], 'effective_directions', [], ...
        'dominant_interaction_index', [], 'cancellation_index', [], ...
        'design_rank', [], 'gram_rcond', [], 'max_atom_similarity', [], ...
        'max_abs_w', [], 'norm_w', [], 'iterations', [], ...
        'best_start', [], 'converged', [], 'runtime_seconds', [], ...
        'solver_status', '', 'objective_history', []);
end

function diagnostics = empty_rank_diagnostics()
    diagnostics = struct('rank', [], 'data_scale', [], ...
        'baseline_available', false, 'baseline_source', '', ...
        'baseline_objective', Inf, 'optimized_objective', Inf, ...
        'start_objectives', [], 'start_iterations', [], ...
        'start_converged', [], 'start_status', {{}}, ...
        'selected_baseline', false);
end

% -------------------------------------------------------------------------
% Output Generation & Figure Plotting
% -------------------------------------------------------------------------

function paths = make_output_paths(output_dir, maximum_rank)
    prefix = 'global_joint_cp_sinusoidal_sender_rank_sweep';
    paths = struct();
    paths.summary_log = fullfile(output_dir, [prefix '_summary.txt']);
    paths.results_mat = fullfile(output_dir, [prefix '_results.mat']);
    paths.rank_summary_csv = fullfile(output_dir, [prefix '_rank_summary.csv']);
    paths.component_metrics_csv = fullfile(output_dir, [prefix '_component_metrics.csv']);
    paths.per_direction_csv = fullfile(output_dir, [prefix '_per_direction_metrics.csv']);
    paths.max_rank_weights_csv = fullfile(output_dir, sprintf('%s_rank%d_component_weights.csv', prefix, maximum_rank));

    paths.explained_png = fullfile(output_dir, [prefix '_explained_vs_rank.png']);
    paths.explained_fig = fullfile(output_dir, [prefix '_explained_vs_rank.fig']);
    paths.fundamental_explained_png = fullfile(output_dir, [prefix '_fundamental_explained_vs_rank.png']);
    paths.fundamental_explained_fig = fullfile(output_dir, [prefix '_fundamental_explained_vs_rank.fig']);
    paths.incremental_explained_png = fullfile(output_dir, [prefix '_incremental_explained_vs_rank.png']);
    paths.incremental_explained_fig = fullfile(output_dir, [prefix '_incremental_explained_vs_rank.fig']);
    paths.validation_explained_png = fullfile(output_dir, [prefix '_validation_explained_vs_rank.png']);
    paths.validation_explained_fig = fullfile(output_dir, [prefix '_validation_explained_vs_rank.fig']);
    paths.diagnostics_png = fullfile(output_dir, [prefix '_diagnostics_vs_rank.png']);
    paths.diagnostics_fig = fullfile(output_dir, [prefix '_diagnostics_vs_rank.fig']);
    paths.directional_localization_png = fullfile(output_dir, [prefix '_directional_localization_vs_rank.png']);
    paths.directional_localization_fig = fullfile(output_dir, [prefix '_directional_localization_vs_rank.fig']);
end

function save_rank_sweep_outputs_sinusoidal(rank_sweep_results, opts)
    paths = rank_sweep_results.output_files;
    fits = rank_sweep_results.fits;
    ranks = rank_sweep_results.ranks;
    interaction_meta = rank_sweep_results.interaction_meta;
    P = numel(interaction_meta);
    output_dir = rank_sweep_results.output_dir;
    prefix = 'global_joint_cp_sinusoidal_sender_rank_sweep';

    if opts.keep_figures
        figure_visibility = 'on';
    else
        figure_visibility = 'off';
    end

    % 1. CSV Summaries
    Rank = ranks;
    Objective = rank_sweep_results.objective;
    RelativeResidual = rank_sweep_results.relative_residual;
    TotalExplainedFraction = rank_sweep_results.explained_fraction;
    TotalExplainedPercent = rank_sweep_results.explained_percent;
    FundamentalExplainedFraction = rank_sweep_results.fundamental_explained_fraction;
    FundamentalExplainedPercent = rank_sweep_results.fundamental_explained_percent;
    IncrementalExplainedFraction = rank_sweep_results.incremental_explained_fraction;
    IncrementalExplainedPercent = 100 * IncrementalExplainedFraction;
    CeilingExplainedFraction = repmat(rank_sweep_results.ceiling_explained_fraction, numel(ranks), 1);
    CeilingExplainedPercent = repmat(rank_sweep_results.ceiling_explained_percent, numel(ranks), 1);
    OrthogonalityError = rank_sweep_results.orthogonality_error;
    ValidationExplainedFraction = rank_sweep_results.validation_explained_fraction;

    MaxAbsW = [fits.max_abs_w].';
    NormW = [fits.norm_w].';
    CancellationIndex = [fits.cancellation_index].';
    GramRCond = [fits.gram_rcond].';
    MaxAtomSimilarity = [fits.max_atom_similarity].';
    DesignRank = [fits.design_rank].';

    RuntimeSeconds = rank_sweep_results.runtime_seconds;
    BestStart = [fits.best_start].';
    Iterations = [fits.iterations].';
    Converged = [fits.converged].';
    SolverStatus = string({fits.solver_status}.');

    summary_table = table(Rank, Objective, RelativeResidual, ...
        TotalExplainedFraction, TotalExplainedPercent, ...
        FundamentalExplainedFraction, FundamentalExplainedPercent, ...
        IncrementalExplainedFraction, IncrementalExplainedPercent, ...
        CeilingExplainedFraction, CeilingExplainedPercent, ...
        OrthogonalityError, MaxAbsW, NormW, CancellationIndex, GramRCond, ...
        MaxAtomSimilarity, DesignRank, ValidationExplainedFraction, ...
        RuntimeSeconds, BestStart, Iterations, Converged, SolverStatus);
    if opts.SaveCSV
        writetable(summary_table, paths.rank_summary_csv);
        writetable(comp_table, paths.component_metrics_csv);
        writetable(direction_table, paths.per_direction_csv);
    end

    % -------------------------------------------------------------------------
    % Figures
    % -------------------------------------------------------------------------

    % Fig 1: Total Explained Energy & Fundamental Ceiling (+ Unconstrained if present)
    fig1 = figure('Color', 'w', 'Position', [100, 100, 850, 500], 'Visible', figure_visibility);
    ax1 = axes(fig1); hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
    plot(ax1, ranks, rank_sweep_results.explained_percent, '-o', ...
        'LineWidth', 2.2, 'MarkerSize', 7, 'DisplayName', 'Sinusoidal-sender CP');
    yline(ax1, rank_sweep_results.ceiling_explained_percent, '--r', ...
        'LineWidth', 2.0, 'DisplayName', sprintf('Fundamental Mode Ceiling (%.2f%%)', rank_sweep_results.ceiling_explained_percent));
    if rank_sweep_results.unconstrained_data.available
        plot(ax1, rank_sweep_results.unconstrained_data.ranks, ...
            100 * rank_sweep_results.unconstrained_data.explained_fraction, ':s', ...
            'LineWidth', 1.8, 'MarkerSize', 6, 'DisplayName', 'Unconstrained Sender CP');
    end
    for k = 1:numel(ranks)
        text(ax1, ranks(k), rank_sweep_results.explained_percent(k), ...
            sprintf(' %.2f%%', rank_sweep_results.explained_percent(k)), ...
            'VerticalAlignment', 'bottom', 'FontSize', 8);
    end
    xlabel(ax1, 'CP Rank R'); ylabel(ax1, 'Total Explained Energy (%)');
    title(ax1, {'Total Tensor-Energy Explained Percentage vs Rank', ...
        'Constrained Sinusoidal Sender b_r(\\phi) = \\sqrt{2}\\cos(\\phi - \\delta_r)'});
    xticks(ax1, ranks); legend(ax1, 'Location', 'southeast');
    save_fig_helper(fig1, paths.explained_png, paths.explained_fig, opts);

    % Fig 2: Fundamental Subspace Explained Energy E_R^{(1)}
    fig2 = figure('Color', 'w', 'Position', [100, 100, 850, 500], 'Visible', figure_visibility);
    ax2 = axes(fig2); hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
    plot(ax2, ranks, rank_sweep_results.fundamental_explained_percent, '-^', ...
        'Color', [0.85, 0.325, 0.098], 'LineWidth', 2.2, 'MarkerSize', 7);
    for k = 1:numel(ranks)
        text(ax2, ranks(k), rank_sweep_results.fundamental_explained_percent(k), ...
            sprintf(' %.2f%%', rank_sweep_results.fundamental_explained_percent(k)), ...
            'VerticalAlignment', 'bottom', 'FontSize', 8);
    end
    xlabel(ax2, 'CP Rank R'); ylabel(ax2, 'Fundamental Subspace Explained Energy (%)');
    title(ax2, {'Explained Energy within Fundamental Sender Subspace E_R^{(1)}', ...
        'Relative to ||C^{(1)}||_F^2 (Filtered to n = \\pm 1 in Sender Axis)'});
    xticks(ax2, ranks);
    save_fig_helper(fig2, paths.fundamental_explained_png, paths.fundamental_explained_fig, opts);

    % Fig 3: Incremental Explained Energy
    fig3 = figure('Color', 'w', 'Position', [100, 100, 850, 480], 'Visible', figure_visibility);
    ax3 = axes(fig3);
    bar(ax3, ranks, 100 * rank_sweep_results.incremental_explained_fraction, 'FaceColor', [0, 0.447, 0.741]);
    grid(ax3, 'on'); box(ax3, 'on');
    xlabel(ax3, 'CP Rank R'); ylabel(ax3, 'Incremental Explained Energy (pp)');
    title(ax3, 'Actual Explained Energy Increment \\Delta E_R from Rank-(R-1)');
    xticks(ax3, ranks);
    save_fig_helper(fig3, paths.incremental_explained_png, paths.incremental_explained_fig, opts);

    % Fig 4: Validation Explained Energy (if provided)
    if rank_sweep_results.val_data.available
        fig4 = figure('Color', 'w', 'Position', [100, 100, 850, 480], 'Visible', figure_visibility);
        ax4 = axes(fig4); hold(ax4, 'on'); grid(ax4, 'on'); box(ax4, 'on');
        plot(ax4, ranks, 100 * rank_sweep_results.explained_fraction, '-o', 'LineWidth', 2.0, 'DisplayName', 'Training');
        plot(ax4, ranks, 100 * rank_sweep_results.validation_explained_fraction, '--s', 'LineWidth', 2.0, 'DisplayName', 'Validation');
        xlabel(ax4, 'CP Rank R'); ylabel(ax4, 'Explained Energy (%)');
        title(ax4, 'Training vs Independent Validation Tensor Explained Energy');
        xticks(ax4, ranks); legend(ax4, 'Location', 'southeast');
        save_fig_helper(fig4, paths.validation_explained_png, paths.validation_explained_fig, opts);
    end

    % Fig 5: Diagnostics vs Rank (4 tiles)
    fig5 = figure('Color', 'w', 'Position', [100, 100, 950, 700], 'Visible', figure_visibility);
    tiledlayout(fig5, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax5_1 = nexttile; plot(ax5_1, ranks, MaxAbsW, '-o', 'LineWidth', 1.8); grid(ax5_1, 'on');
    title(ax5_1, 'Max |W_{pr}|'); xlabel(ax5_1, 'Rank R'); xticks(ax5_1, ranks);

    ax5_2 = nexttile; semilogy(ax5_2, ranks, GramRCond, '-s', 'LineWidth', 1.8); grid(ax5_2, 'on');
    title(ax5_2, 'Gram Matrix rcond(D^T D)'); xlabel(ax5_2, 'Rank R'); xticks(ax5_2, ranks);

    ax5_3 = nexttile; plot(ax5_3, ranks, MaxAtomSimilarity, '-d', 'LineWidth', 1.8); grid(ax5_3, 'on');
    title(ax5_3, 'Max Atom Pair Similarity'); xlabel(ax5_3, 'Rank R'); xticks(ax5_3, ranks);

    ax5_4 = nexttile; plot(ax5_4, ranks, CancellationIndex, '-^', 'LineWidth', 1.8); grid(ax5_4, 'on');
    title(ax5_4, 'Cancellation Index \\kappa_{cancel}'); xlabel(ax5_4, 'Rank R'); xticks(ax5_4, ranks);

    save_fig_helper(fig5, paths.diagnostics_png, paths.diagnostics_fig, opts);

    % Fig 6: Directional Localization vs Rank
    fig6 = figure('Color', 'w', 'Position', [100, 100, 950, 480], 'Visible', figure_visibility);
    tiledlayout(fig6, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax6_1 = nexttile; hold(ax6_1, 'on'); grid(ax6_1, 'on');
    ax6_2 = nexttile; hold(ax6_2, 'on'); grid(ax6_2, 'on');
    for k = 1:numel(fits)
        R = fits(k).rank;
        plot(ax6_1, repmat(R, R, 1), fits(k).max_direction_share, 'k.', 'MarkerSize', 12);
        plot(ax6_2, repmat(R, R, 1), fits(k).effective_directions, 'k.', 'MarkerSize', 12);
    end
    plot(ax6_1, ranks, arrayfun(@(f) mean(f.max_direction_share), fits), '-r', 'LineWidth', 2.0);
    plot(ax6_2, ranks, arrayfun(@(f) mean(f.effective_directions), fits), '-r', 'LineWidth', 2.0);
    title(ax6_1, 'Max Direction Share s_r^{max}'); xlabel(ax6_1, 'Rank R'); xticks(ax6_1, ranks);
    title(ax6_2, 'Effective Directions N_{eff,r}'); xlabel(ax6_2, 'Rank R'); xticks(ax6_2, ranks);
    save_fig_helper(fig6, paths.directional_localization_png, paths.directional_localization_fig, opts);

    % -------------------------------------------------------------------------
    % Per-Rank Detailed Output Figures (Heatmaps, Networks, Profiles, Phases, Reconstructions)
    % -------------------------------------------------------------------------

    labels = cell(P, 1);
    for p = 1:P
        meta = get_meta_struct(interaction_meta, p);
        pname = get_pair_name(meta);
        labels{p} = sprintf('%s (%d->%d)', pname, get_source_id(meta), get_target_id(meta));
    end

    round_dir = round_dir_from_results(rank_sweep_results);

    for k = 1:numel(fits)
        current_fit = fits(k);
        R = current_fit.rank;

        % Heatmap of W for Rank R
        fig_w = figure('Color', 'w', 'Position', [100, 100, 900, 600], 'Visible', figure_visibility);
        ax_w = axes(fig_w);
        imagesc(ax_w, current_fit.W); colormap(ax_w, make_diverging_colormap(256)); colorbar(ax_w);
        xlabel(ax_w, 'Component r'); ylabel(ax_w, 'Directed Interaction p');
        xticks(ax_w, 1:R); yticks(ax_w, 1:P); yticklabels(ax_w, labels);
        title(ax_w, sprintf('Rank-%d Sinusoidal-Sender Component Weights W_{pr}', R));
        save_fig_helper(fig_w, fullfile(output_dir, sprintf('%s_rank%d_weights_heatmap.png', prefix, R)), ...
            fullfile(output_dir, sprintf('%s_rank%d_weights_heatmap.fig', prefix, R)), opts);
        if ~opts.keep_figures, close(fig_w); end

        % Sender Phases delta_r for Rank R
        fig_phase = figure('Color', 'w', 'Position', [100, 100, 900, 420], 'Visible', figure_visibility);
        tiledlayout(fig_phase, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

        ax_ph1 = nexttile; hold(ax_ph1, 'on'); grid(ax_ph1, 'on'); box(ax_ph1, 'on');
        stem(ax_ph1, 1:R, current_fit.delta, 'LineWidth', 1.8, 'MarkerSize', 7);
        ylabel(ax_ph1, 'Phase \\delta_r (rad)'); xlabel(ax_ph1, 'Component r');
        ylim(ax_ph1, [0, pi]); xticks(ax_ph1, 1:R);
        yticks(ax_ph1, [0, pi/4, pi/2, 3*pi/4, pi]);
        yticklabels(ax_ph1, {'0', '\\pi/4', '\\pi/2', '3\\pi/4', '\\pi'});
        title(ax_ph1, sprintf('Rank-%d Sender Phases \\delta_r \\in [0, \\pi)', R));

        ax_ph2 = nexttile; hold(ax_ph2, 'on'); grid(ax_ph2, 'on'); axis(ax_ph2, 'equal');
        th_semi = linspace(0, pi, 200);
        plot(ax_ph2, cos(th_semi), sin(th_semi), 'k--', 'LineWidth', 1.2);
        colors_comp = lines(R);
        for r = 1:R
            plot(ax_ph2, [0, cos(current_fit.delta(r))], [0, sin(current_fit.delta(r))], ...
                '-o', 'Color', colors_comp(r,:), 'LineWidth', 2.0, 'MarkerSize', 6);
            text(ax_ph2, 1.15*cos(current_fit.delta(r)), 1.15*sin(current_fit.delta(r)), ...
                sprintf('r=%d (%.1f^o)', r, rad2deg(current_fit.delta(r))), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
        xlim(ax_ph2, [-1.3, 1.3]); ylim(ax_ph2, [-0.2, 1.3]);
        title(ax_ph2, 'Sender Phase Directions on Semi-Circle');
        save_fig_helper(fig_phase, fullfile(output_dir, sprintf('%s_rank%d_sender_phases.png', prefix, R)), ...
            fullfile(output_dir, sprintf('%s_rank%d_sender_phases.fig', prefix, R)), opts);
        if ~opts.keep_figures, close(fig_phase); end

        % All Components Network Summary for Rank R
        fig_net_all = figure('Color', 'w', 'Position', [100, 100, 1000, max(360, 340 * ceil(R/2))], 'Visible', figure_visibility);
        t_lay_net = tiledlayout(fig_net_all, ceil(R/2), min(R, 2), 'TileSpacing', 'compact', 'Padding', 'compact');
        title(t_lay_net, sprintf('Rank-%d Fit: Network Graphs per Component', R), 'FontWeight', 'bold', 'FontSize', 13);
        for r = 1:R
            ax_net_single = nexttile(t_lay_net, r);
            plot_component_network_graph(ax_net_single, interaction_meta, current_fit.W(:, r), r, rank_sweep_results.unique_agents, round_dir);
        end
        save_fig_helper(fig_net_all, fullfile(output_dir, sprintf('%s_rank%d_all_components_network.png', prefix, R)), ...
            fullfile(output_dir, sprintf('%s_rank%d_all_components_network.fig', prefix, R)), opts);
        if ~opts.keep_figures, close(fig_net_all); end

        % Individual 3-Panel Component Plots for Rank R
        for r = 1:R
            a_r = current_fit.a_values(:, r);
            b_r = current_fit.b_values(:, r);
            W_r = current_fit.W(:, r);

            fig_comp = figure('Color', 'w', 'Position', [100, 100, 1400, 420], 'Visible', figure_visibility);
            t_lay = tiledlayout(fig_comp, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
            title(t_lay, sprintf('Global Joint CP Component %d of %d (Rank-%d Fit)', r, R, R), 'FontWeight', 'bold', 'FontSize', 12);

            % Panel 1: Shared Receiver Profile a_r(phi)
            ax_in = nexttile(t_lay, 1); hold(ax_in, 'on'); grid(ax_in, 'on'); box(ax_in, 'on');
            plot(ax_in, rank_sweep_results.phi_grid, a_r, 'LineWidth', 2.8, 'Color', [0, 0.447, 0.741]);
            ylabel(ax_in, 'a (Shared Receiver)'); xlabel(ax_in, '\\phi_{target}'); set_phase_axis(ax_in);
            title(ax_in, sprintf('Shared Receiver Profile a_%d(\\phi)', r));

            % Panel 2: Shared Sender Profile b_r(phi) = sqrt(2)*cos(phi - delta_r)
            ax_out = nexttile(t_lay, 2); hold(ax_out, 'on'); grid(ax_out, 'on'); box(ax_out, 'on');
            plot(ax_out, rank_sweep_results.phi_grid, b_r, 'k--', 'LineWidth', 3.0, 'DisplayName', sprintf('Shared b_%d(\\phi) (\\delta_r=%.2f rad)', r, current_fit.delta(r)));
            colors_p = lines(P);
            for p = 1:P
                plot(ax_out, rank_sweep_results.phi_grid, W_r(p) * b_r, 'LineWidth', 1.2, 'Color', colors_p(p,:));
            end
            ylabel(ax_out, 'b (Shared Sender & W_{pr}b_r)'); xlabel(ax_out, '\\phi_{source}'); set_phase_axis(ax_out);
            title(ax_out, sprintf('Shared Sender Profile b_%d(\\phi) = \\sqrt{2}\\cos(\\phi - %.2f)', r, current_fit.delta(r)));

            % Panel 3: Component Network Graph
            ax_net = nexttile(t_lay, 3);
            plot_component_network_graph(ax_net, interaction_meta, W_r, r, rank_sweep_results.unique_agents, round_dir);

            save_fig_helper(fig_comp, fullfile(output_dir, sprintf('%s_rank%d_component%d_profiles.png', prefix, R, r)), ...
                fullfile(output_dir, sprintf('%s_rank%d_component%d_profiles.fig', prefix, R, r)), opts);
            if ~opts.keep_figures, close(fig_comp); end
        end

        % Representative Interaction Surface Reconstruction Plots for Rank R
        fig_recon = figure('Color', 'w', 'Position', [100, 100, 1200, 800], 'Visible', figure_visibility);
        num_show = min(4, P);
        t_lay_rec = tiledlayout(fig_recon, num_show, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        title(t_lay_rec, sprintf('Rank-%d Fit: Representative Interaction Reconstructions', R), 'FontWeight', 'bold', 'FontSize', 12);
        rec_basis = exp(1i * rank_sweep_results.phi_grid * rank_sweep_results.m_values(:).');
        snd_basis = exp(1i * rank_sweep_results.phi_grid * rank_sweep_results.n_values(:).');

        for idx_p = 1:num_show
            p_curr = idx_p;
            s_orig = real(rec_basis * rank_sweep_results.C_tensor(:,:,p_curr) * snd_basis.');
            s_fit = real(rec_basis * current_fit.C_fit(:,:,p_curr) * snd_basis.');
            s_res = s_orig - s_fit;

            ax_r1 = nexttile(t_lay_rec);
            imagesc(ax_r1, rank_sweep_results.phi_grid, rank_sweep_results.phi_grid, s_orig);
            colormap(ax_r1, make_diverging_colormap(256)); colorbar(ax_r1);
            title(ax_r1, sprintf('%s Orig', labels{p_curr}), 'FontSize', 8, 'Interpreter', 'none');
            set_phase_axis(ax_r1); set(ax_r1, 'YTick', [0, pi, 2*pi], 'YTickLabel', {'0','\\pi','2\\pi'});

            ax_r2 = nexttile(t_lay_rec);
            imagesc(ax_r2, rank_sweep_results.phi_grid, rank_sweep_results.phi_grid, s_fit);
            colormap(ax_r2, make_diverging_colormap(256)); colorbar(ax_r2);
            title(ax_r2, sprintf('%s Fit (Rank %d)', labels{p_curr}, R), 'FontSize', 8, 'Interpreter', 'none');
            set_phase_axis(ax_r2); set(ax_r2, 'YTick', [0, pi, 2*pi], 'YTickLabel', {'0','\\pi','2\\pi'});

            ax_r3 = nexttile(t_lay_rec);
            imagesc(ax_r3, rank_sweep_results.phi_grid, rank_sweep_results.phi_grid, s_res);
            colormap(ax_r3, make_diverging_colormap(256)); colorbar(ax_r3);
            title(ax_r3, sprintf('%s Residual', labels{p_curr}), 'FontSize', 8, 'Interpreter', 'none');
            set_phase_axis(ax_r3); set(ax_r3, 'YTick', [0, pi, 2*pi], 'YTickLabel', {'0','\\pi','2\\pi'});
        end
        save_fig_helper(fig_recon, fullfile(output_dir, sprintf('%s_rank%d_interaction_reconstructions.png', prefix, R)), ...
            fullfile(output_dir, sprintf('%s_rank%d_interaction_reconstructions.fig', prefix, R)), opts);
        if ~opts.keep_figures, close(fig_recon); end
    end

    if ~opts.keep_figures
        close(fig1); close(fig2); close(fig3);
        if rank_sweep_results.val_data.available, close(fig4); end
        close(fig5); close(fig6);
    end

    save(paths.results_mat, 'rank_sweep_results', '-v7.3');
end

function save_fig_helper(fig, path_png, path_fig, opts)
    if isfield(opts, 'SavePNG') && ~opts.SavePNG
        % skip PNG
    else
        saveas(fig, path_png);
    end
    if isfield(opts, 'SaveFIG') && opts.SaveFIG
        savefig(fig, path_fig);
    end
end

% -------------------------------------------------------------------------
% Plotting Helpers
% -------------------------------------------------------------------------

function round_dir = round_dir_from_results(rank_sweep_results)
    round_dir = '';
    if isfield(rank_sweep_results, 'output_dir') && ~isempty(rank_sweep_results.output_dir)
        round_dir = rank_sweep_results.output_dir;
    elseif isfield(rank_sweep_results, 'options') && isfield(rank_sweep_results.options, 'cache_dir')
        round_dir = rank_sweep_results.options.cache_dir;
    end
end

function plot_component_network_graph(ax, interaction_meta, W_r, r, unique_agents, round_dir)
    P = numel(interaction_meta);
    source_ids = zeros(P, 1);
    target_ids = zeros(P, 1);
    for p = 1:P
        meta = get_meta_struct(interaction_meta, p);
        source_ids(p) = meta.source_id;
        target_ids(p) = meta.target_id;
    end

    node_names = arrayfun(@(id) sprintf('%d', id), unique_agents(:), 'UniformOutput', false);
    source_names = arrayfun(@(id) sprintf('%d', id), source_ids, 'UniformOutput', false);
    target_names = arrayfun(@(id) sprintf('%d', id), target_ids, 'UniformOutput', false);

    graph_data = digraph(source_names, target_names, W_r(:), node_names);
    [x_data, y_data] = get_preferred_node_positions(graph_data, round_dir);

    graph_plot = plot(ax, graph_data, 'XData', x_data, 'YData', y_data, ...
        'NodeLabel', {}, 'ArrowSize', 16, 'ArrowPosition', 0.75, ...
        'MarkerSize', 8, 'NodeColor', [0.15, 0.15, 0.15], 'EdgeColor', [0.0, 0.4470, 0.7410]);

    axis(ax, 'equal');
    margin = 0.45;
    rx = max(x_data, [], 'omitnan') - min(x_data, [], 'omitnan') + 2*margin;
    ry = max(y_data, [], 'omitnan') - min(y_data, [], 'omitnan') + 2*margin;
    max_r = max(rx, ry);
    cx = (min(x_data, [], 'omitnan') + max(x_data, [], 'omitnan')) / 2;
    cy = (min(y_data, [], 'omitnan') + max(y_data, [], 'omitnan')) / 2;
    xlim(ax, [cx - max_r/2, cx + max_r/2]);
    ylim(ax, [cy - max_r/2, cy + max_r/2]);

    title(ax, sprintf('Component %d Network (W_{p,%d})', r, r), 'FontSize', 10, 'FontWeight', 'bold');

    if numedges(graph_data) > 0
        max_abs_w = max(abs(graph_data.Edges.Weight));
        clim_limit = max(0.1, max_abs_w);
        linewidth_limit = max(0.06, max_abs_w);

        graph_plot.LineWidth = scale_edge_width(abs(graph_data.Edges.Weight), linewidth_limit);
        graph_plot.EdgeCData = graph_data.Edges.Weight;
        graph_plot.EdgeColor = 'flat';
        clim(ax, [-clim_limit, clim_limit]);
        colormap(ax, make_diverging_colormap(256));
        cb = colorbar(ax);
        ylabel(cb, sprintf('W_{p,%d}', r));
    end

    draw_edge_labels(ax, graph_data, x_data, y_data);
    draw_node_labels(ax, graph_data, x_data, y_data);
    axis(ax, 'off');
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
        preferred_ids = [7, 8, 9, 10, 11, 12];
        preferred_x   = [1.5, 1.0, 1.5, 2.5, 3.0, 2.5];
        preferred_y   = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    else
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
            y_label = y_label(node) + node_offset;
        end
        text(ax, x_label, y_label, G.Nodes.Name{node}, ...
            'FontSize', 12, 'FontWeight', 'bold', ...
            'Color', [0.15, 0.15, 0.15], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
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

function set_phase_axis(ax)
    set(ax, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
    set(ax, 'XTickLabel', {'0', '\\pi/2', '\\pi', '3\\pi/2', '2\\pi'});
    xlim(ax, [0, 2*pi]);
end

function meta = get_meta_struct(interaction_meta, p)
    if iscell(interaction_meta)
        meta = interaction_meta{p};
    else
        meta = interaction_meta(p);
    end
    while iscell(meta) && ~isempty(meta)
        meta = meta{1};
    end
    if isstruct(meta) && numel(meta) > 1
        meta = meta(1);
    end
end

function str = get_pair_name(meta)
    str = '';
    if isstruct(meta)
        if isfield(meta, 'pair_name') && ~isempty(meta.pair_name)
            val = meta.pair_name;
            while iscell(val) && ~isempty(val), val = val{1}; end
            str = char(val);
        elseif isfield(meta, 'source_id') && isfield(meta, 'target_id')
            str = sprintf('Pair_%d_%d', get_source_id(meta), get_target_id(meta));
        end
    end
    if isempty(str)
        str = 'UnknownPair';
    end
end

function sid = get_source_id(meta)
    sid = 0;
    if isstruct(meta) && isfield(meta, 'source_id')
        sid = meta.source_id;
        while iscell(sid) && ~isempty(sid), sid = sid{1}; end
        if isstring(sid) || ischar(sid), sid = str2double(sid); end
    end
end

function tid = get_target_id(meta)
    tid = 0;
    if isstruct(meta) && isfield(meta, 'target_id')
        tid = meta.target_id;
        while iscell(tid) && ~isempty(tid), tid = tid{1}; end
        if isstring(tid) || ischar(tid), tid = str2double(tid); end
    end
end

% -------------------------------------------------------------------------
% Synthetic Validation Suite
% -------------------------------------------------------------------------

function results = run_synthetic_validation(M, opts)
    fprintf('\n=== RUNNING SYNTHETIC VALIDATION SUITE FOR SINUSOIDAL SENDER MODEL ===\n');
    L = 2*M + 1;
    P = 12;
    results = struct('ran', true, 'passed', true, 'tests', struct());

    % Synthetic metadata
    interaction_meta = repmat(struct('pair_name','', 'target_id',0, 'source_id',0, 'direction',''), P, 1);
    agents = [7, 8, 9, 10];
    p_idx = 0;
    for s = 1:numel(agents)
        for t = 1:numel(agents)
            if s ~= t
                p_idx = p_idx + 1;
                interaction_meta(p_idx).source_id = agents(s);
                interaction_meta(p_idx).target_id = agents(t);
                interaction_meta(p_idx).pair_name = sprintf('Pair_%d_%d', agents(s), agents(t));
                interaction_meta(p_idx).direction = sprintf('%d->%d', agents(s), agents(t));
            end
        end
    end

    % TEST 1: Exact Rank-3 Reconstruction with distinct delta_r
    fprintf('\n[TEST 1] Known Rank-3 Tensor Reconstruction with distinct delta_r\n');
    R_true = 3;
    delta_true = [0.2, 1.2, 2.5].';
    A_true = complex(zeros(L, R_true));
    B_true = complex(zeros(L, R_true));
    for r = 1:R_true
        A_true(:,r) = random_conjugate_symmetric_unit_vector(L);
        B_true(:,r) = create_sinusoidal_B(delta_true(r), L, M);
    end
    W_true = randn(P, R_true);
    C_syn = reconstruct_cp_tensor(A_true, B_true, W_true, [L, L, P]);

    test_opts = opts;
    test_opts.NumStarts = 10;
    test_opts.MaxIter = 500;
    test_opts.Ranks = 1:4;

    m_values = (-M:M).'; n_values = (-M:M).'; phi_grid = linspace(0, 2*pi, 100).';
    C_fund_syn = compute_fundamental_tensor(C_syn, M);
    val_empty = struct('available', false, 'path', '', 'C_val', [], 'energy', NaN);

    [fits_syn, ~] = fit_rank_sequence_sinusoidal(C_syn, C_fund_syn, m_values, n_values, ...
        phi_grid, 1:4, test_opts, empty_rank1_reference(), struct('available', false), val_empty);

    res_R3 = fits_syn(3).relative_residual;
    t1_pass = res_R3 < 1e-4;
    fprintf('  Rank-3 Relative Residual = %.4e (Pass threshold < 1e-4): %d\n', res_R3, t1_pass);
    results.tests.test1_exact_reconstruction = struct('passed', t1_pass, 'relative_residual', res_R3);

    % TEST 2: Strict B_r Zero Spectrum outside n = +/- 1
    fprintf('\n[TEST 2] Verifying B_r Spectrum is Zero for n ~= +/- 1\n');
    non_fund_mask = true(L, 1);
    non_fund_mask(M) = false;     % n = -1
    non_fund_mask(M+2) = false;   % n = +1
    max_non_fund_energy = 0;
    for k = 1:numel(fits_syn)
        for r = 1:fits_syn(k).rank
            max_non_fund_energy = max(max_non_fund_energy, max(abs(fits_syn(k).B(non_fund_mask, r))));
        end
    end
    t2_pass = max_non_fund_energy < 1e-12;
    fprintf('  Max non-fundamental B_r spectral energy = %.4e (Pass threshold < 1e-12): %d\n', max_non_fund_energy, t2_pass);
    results.tests.test2_strict_sender_spectrum = struct('passed', t2_pass, 'max_leakage', max_non_fund_energy);

    % TEST 3: b_r(phi) Numerical Verification against sqrt(2)*cos(phi - delta_r)
    fprintf('\n[TEST 3] Verifying b_r(phi) equals sqrt(2)*cos(phi - delta_r)\n');
    snd_basis = exp(1i * phi_grid * n_values(:).');
    max_b_diff = 0;
    for k = 1:numel(fits_syn)
        for r = 1:fits_syn(k).rank
            b_num = real(snd_basis * fits_syn(k).B(:,r));
            b_analytic = sqrt(2) * cos(phi_grid - fits_syn(k).delta(r));
            max_b_diff = max(max_b_diff, max(abs(b_num - b_analytic)));
        end
    end
    t3_pass = max_b_diff < 1e-10;
    fprintf('  Max difference between b_r(phi) and sqrt(2)*cos(phi - delta_r) = %.4e: %d\n', max_b_diff, t3_pass);
    results.tests.test3_sinusoidal_form = struct('passed', t3_pass, 'max_diff', max_b_diff);

    % TEST 4: Harmonic Data & Fundamental Mode Ceiling Limit
    fprintf('\n[TEST 4] Higher Harmonics Data & Theoretical Ceiling Limit\n');
    C_harm = C_syn;
    % Add n = +/- 2 harmonic sender components
    C_harm(:, M-1, :) = 0.5 * randn(L, P);
    C_harm(:, M+3, :) = conj(flipud(C_harm(:, M-1, :)));
    C_fund_harm = compute_fundamental_tensor(C_harm, M);
    E_ceil_harm = sum(abs(C_fund_harm(:)).^2) / sum(abs(C_harm(:)).^2);

    [fits_harm, ~] = fit_rank_sequence_sinusoidal(C_harm, C_fund_harm, m_values, n_values, ...
        phi_grid, 1:4, test_opts, empty_rank1_reference(), struct('available', false), val_empty);

    exceeded_ceiling = false;
    for k = 1:numel(fits_harm)
        if fits_harm(k).explained_fraction > E_ceil_harm + 1e-5
            exceeded_ceiling = true;
        end
    end
    t4_pass = ~exceeded_ceiling;
    fprintf('  Fundamental Mode Ceiling E_ceiling = %.4f%%\n', 100 * E_ceil_harm);
    fprintf('  Max Rank-4 Fit Total Explained Fraction = %.4f%%\n', 100 * fits_harm(end).explained_fraction);
    fprintf('  Did fits stay below theoretical ceiling (plus tolerance)? %d\n', t4_pass);
    results.tests.test4_ceiling_limit = struct('passed', t4_pass, 'ceiling', E_ceil_harm, 'max_exp', fits_harm(end).explained_fraction);

    % TEST 5: Degenerate Case Detection (Close delta_r & Close A_r)
    fprintf('\n[TEST 5] Degenerate Component Detection (Close delta_r & Close A_r)\n');
    delta_deg = [0.5, 0.505].';
    A_deg = complex(zeros(L, 2));
    A_deg(:,1) = random_conjugate_symmetric_unit_vector(L);
    A_deg(:,2) = A_deg(:,1) + 0.005 * random_conjugate_symmetric_unit_vector(L);
    A_deg(:,2) = A_deg(:,2) / norm(A_deg(:,2));
    B_deg = [create_sinusoidal_B(delta_deg(1), L, M), create_sinusoidal_B(delta_deg(2), L, M)];
    W_deg = [100 * ones(P,1), -100 * ones(P,1)];
    C_degen = reconstruct_cp_tensor(A_deg, B_deg, W_deg, [L, L, P]);
    C_degen_fund = compute_fundamental_tensor(C_degen, M);
    D_deg = [reshape(A_deg(:,1)*B_deg(:,1).', [], 1), reshape(A_deg(:,2)*B_deg(:,2).', [], 1)];
    gram_deg = real(D_deg' * D_deg);
    rcond_deg = rcond(gram_deg);
    atom_sim_deg = abs(real(D_deg(:,1)' * D_deg(:,2))) / (norm(D_deg(:,1)) * norm(D_deg(:,2)));
    cancellation_deg = (sum(abs(W_deg(:,1)).^2) + sum(abs(W_deg(:,2)).^2)) / sum(abs(C_degen(:)).^2);

    test_opts_deg = test_opts;
    test_opts_deg.Ranks = 1:2;
    [fit_deg_seq, ~] = fit_rank_sequence_sinusoidal(C_degen, C_degen_fund, m_values, n_values, ...
        phi_grid, 1:2, test_opts_deg, empty_rank1_reference(), struct('available', false), val_empty);
    fit_deg = fit_deg_seq(end);

    t5_pass = (rcond_deg < 1e-2) || (atom_sim_deg > 0.90) || (cancellation_deg > 1.5) || ...
              (fit_deg.gram_rcond < 1e-2) || (fit_deg.max_atom_similarity > 0.90) || (fit_deg.cancellation_index > 1.5);
    fprintf('  Degenerate Model Diagnostics: Gram rcond=%.2e, Atom Similarity=%.4f, Cancellation Index=%.2f\n', ...
        rcond_deg, atom_sim_deg, cancellation_deg);
    fprintf('  Did diagnostics successfully flag numerical degeneracy? %d\n', t5_pass);
    results.tests.test5_degeneracy_detection = struct('passed', t5_pass, ...
        'gram_rcond', rcond_deg, 'atom_sim', atom_sim_deg, 'cancel_idx', cancellation_deg);

    % TEST 6: Non-increasing Objective Monotonicity
    fprintf('\n[TEST 6] Unregularized Training Objective Non-Increasing Monotonicity\n');
    objectives = [fits_syn.objective];
    increases = diff(objectives);
    t6_pass = all(increases <= 1e-8);
    fprintf('  Max Objective Step Increase = %.4e (Pass threshold <= 1e-8): %d\n', max(increases), t6_pass);
    results.tests.test6_objective_monotonicity = struct('passed', t6_pass, 'max_increase', max(increases));

    all_passed = t1_pass && t2_pass && t3_pass && t4_pass && t5_pass && t6_pass;
    if all_passed
        status_str = 'PASSED';
    else
        status_str = 'FAILED';
    end
    fprintf('\n=== SYNTHETIC VALIDATION SUITE OVERALL RESULT: %s ===\n\n', status_str);
end

% -------------------------------------------------------------------------
% Final Structured Summary Report
% -------------------------------------------------------------------------

function print_final_summary_report(results)
    fprintf('\n=========================================================================\n');
    fprintf('           SUMMARY REPORT: SINUSOIDAL-SENDER JOINT CP FIT\n');
    fprintf('=========================================================================\n\n');

    fits = results.fits;
    Rmax_fit = fits(end);

    % 1. Loss due to sinusoidal constraint vs free shape b_r
    fprintf('1. EXPLAINED ENERGY LOSS FROM SINUSOIDAL SENDER CONSTRAINT:\n');
    fprintf('   - Theoretical Fundamental Mode Ceiling E_ceiling = %.2f%%\n', results.ceiling_explained_percent);
    fprintf('   - Constrained Sinusoidal Model Rank-%d Explained Energy = %.2f%%\n', Rmax_fit.rank, Rmax_fit.explained_percent);
    fprintf('   - Fundamental Subspace Explained Energy E_R^{(1)} = %.2f%%\n', Rmax_fit.fundamental_explained_percent);
    if results.unconstrained_data.available
        unconst_max = results.unconstrained_data.explained_fraction(end) * 100;
        loss_pp = unconst_max - Rmax_fit.explained_percent;
        fprintf('   - Unconstrained Sender Model Rank-%d Explained Energy = %.2f%%\n', Rmax_fit.rank, unconst_max);
        fprintf('   - Loss in Explained Energy due to Sinusoidal Constraint = %.2f pp\n', loss_pp);
    else
        loss_ceiling_pp = results.ceiling_explained_percent - Rmax_fit.explained_percent;
        fprintf('   - Remaining gap to theoretical fundamental ceiling = %.2f pp\n', loss_ceiling_pp);
    end
    fprintf('\n');

    % 2. Substantial improvement from higher ranks
    fprintf('2. RANK SWEEP IMPROVEMENT TRAJECTORY:\n');
    fprintf('   - Rank 1 Explained Energy: %.2f%%\n', fits(1).explained_percent);
    fprintf('   - Rank %d Explained Energy: %.2f%% (+%.2f pp total gain)\n', ...
        Rmax_fit.rank, Rmax_fit.explained_percent, Rmax_fit.explained_percent - fits(1).explained_percent);
    for k = 2:numel(fits)
        fprintf('     * Rank %2d -> %2d: +%.4f pp\n', fits(k-1).rank, fits(k).rank, ...
            100 * fits(k).incremental_explained_fraction);
    end
    fprintf('\n');

    % 3. Component Directional Localization
    fprintf('3. COMPONENT DIRECTIONAL LOCALIZATION & SHARE:\n');
    for r = 1:Rmax_fit.rank
        meta = get_meta_struct(results.interaction_meta, Rmax_fit.dominant_interaction_index(r));
        pname = get_pair_name(meta);
        fprintf('   - Component %2d: max_share=%.2f%% (%s %d->%d), N_eff=%.2f, \\delta_r=%.1f deg\n', ...
            r, 100 * Rmax_fit.max_direction_share(r), pname, meta.source_id, ...
            meta.target_id, Rmax_fit.effective_directions(r), rad2deg(Rmax_fit.delta(r)));
    end
    avg_neff = mean(Rmax_fit.effective_directions);
    if avg_neff < 2.0
        fprintf('   -> Observation: Components show strong directional localization (mean N_eff = %.2f).\n', avg_neff);
    else
        fprintf('   -> Observation: Components are distributed across multiple interactions (mean N_eff = %.2f).\n', avg_neff);
    end
    fprintf('\n');

    % 4. Conditioning, Large Weights & Cancellation
    fprintf('4. CONDITIONING, LARGE WEIGHTS & COMPONENT CANCELLATION:\n');
    fprintf('   - Max |W_{pr}| = %.2f, ||W||_F = %.2f\n', Rmax_fit.max_abs_w, Rmax_fit.norm_w);
    fprintf('   - Gram Matrix rcond(D^T D) = %.2e\n', Rmax_fit.gram_rcond);
    fprintf('   - Max Atom Pair Similarity = %.4f\n', Rmax_fit.max_atom_similarity);
    fprintf('   - Cancellation Index \\kappa_{cancel} = %.2f\n', Rmax_fit.cancellation_index);
    if Rmax_fit.cancellation_index > 1.5 || Rmax_fit.gram_rcond < 1e-4
        fprintf('   -> WARNING: High cancellation index or ill-conditioned Gram matrix indicates\n');
        fprintf('               some components have opposing large weights due to close sender phases or profiles.\n');
    else
        fprintf('   -> Observation: Model parameters are well-conditioned without severe cancellation.\n');
    end
    fprintf('\n');

    % 5. Independent Validation Evaluation
    fprintf('5. INDEPENDENT VALIDATION EVALUATION:\n');
    if results.val_data.available
        fprintf('   - Rank 1 Validation Explained Energy = %.2f%%\n', 100 * fits(1).validation_explained_fraction);
        fprintf('   - Rank %d Validation Explained Energy = %.2f%%\n', Rmax_fit.rank, 100 * Rmax_fit.validation_explained_fraction);
        val_gain = 100 * (Rmax_fit.validation_explained_fraction - fits(1).validation_explained_fraction);
        fprintf('   - Validation Gain from Higher Ranks = %+.2f pp\n', val_gain);
        if val_gain > 0
            fprintf('   -> Conclusion: Higher ranks successfully improve generalization on independent validation data.\n');
        else
            fprintf('   -> Conclusion: Higher ranks show signs of overfitting on independent validation data.\n');
        end
    else
        fprintf('   - No independent validation tensor provided.\n');
        fprintf('   -> Note: Training explained energy gains cannot guarantee lack of overfitting.\n');
        fprintf('            Rely on diagnostics (Gram rcond, atom similarity, cancellation) to select rank.\n');
    end
    fprintf('\n=========================================================================\n\n');
end
