function results = global_joint_cp_rank1_profile_free_network_svd(round_dir, M, varargin)
%GLOBAL_JOINT_CP_RANK1_PROFILE_FREE_NETWORK_SVD
% Joint CP fitting with Rank-1 Profile (R_profile = 1) and Free Directed Signed Network W,
% followed by network decomposition (SVD or Witten et al. 2009 Sparse PMD) and collective signal analysis.
%
% Model:
%   C(:,:,p) approx W_{i_p, j_p} * A * B.'
%   s_{i<-j}(phi_i, phi_j) approx W_{ij} * a(phi_i) * b(phi_j),  i ~= j (W_{ii} = 0)
%   b(phi) = sqrt(2) * cos(phi - delta), delta in [0, pi)
%   B_{+1} = exp(-1i * delta) / sqrt(2),  B_{-1} = exp(1i * delta) / sqrt(2)
%   W in R^{N x N} is an unconstrained signed directed network (W_{ii} = 0).
%
% Network Decompositions:
%   1. SVD:        W = sum_{l=1}^N sigma_l * u_l * v_l.'
%   2. Sparse PMD: W approx sum_{l=1}^K d_l * p_l * q_l.'
%      subject to ||p_l||_2 = 1, ||q_l||_2 = 1, ||p_l||_1 <= c_p, ||q_l||_1 <= c_q.
%
% Saved Output PNGs:
%   1. rank1_profile_free_network_[svd|pmd]_summary.png
%   2. rank1_profile_free_network_[svd|pmd]_modes.png
%   3. rank1_profile_free_network_[svd|pmd]_approximation.png
%   4. rank1_profile_free_network_collective_signals.png (if phase time series available)

    % =========================================================================
    % USER CONFIGURATION: TARGET DATASET & DECOMPOSITION (For F5 / Run without args)
    % =========================================================================
    % Change these variables to switch datasets or decomposition methods in code:
    %
    % Target Dataset / Directory:
    %   'SStickFlat': SStickFlat experiment (Agents 8, 9, 11, 12)
    %   'SStick'    : SStick experiment (Agents 7, 8, 9, 10)
    %   'Round'     : Round experiment (Agents 7, 8, 9, 10)
    %   'Round6'    : Round6 experiment (Agents 7, 8, 9, 10, 11, 12)
    %   'Stick'     : Stick experiment (Agents 7, 8, 9, 10)
    % Or specify any relative/absolute path.
    DEFAULT_DATASET = 'SStickFlat';

    % Network Decomposition Method: 'svd' or 'sparse_pmd'
    DEFAULT_NETWORK_DECOMPOSITION = 'svd';

    % Sparsity Parameter rho in [0, 1] (0: strongest L1 constraint, 1: unconstrained)
    % L1 bounds are mapped via:
    %   c_p = 1 + ReceiverSparsity * (sqrt(N) - 1)
    %   c_q = 1 + SenderSparsity   * (sqrt(N) - 1)
    % Examples:
    %   - Both sides sparse:             ReceiverSparsity = 0.5, SenderSparsity = 0.5
    %   - Receiver only sparse:          ReceiverSparsity = 0.5, SenderSparsity = 1.0
    %   - Sender only sparse:            ReceiverSparsity = 1.0, SenderSparsity = 0.5
    %   - Unconstrained (reduces to SVD): ReceiverSparsity = 1.0, SenderSparsity = 1.0
    DEFAULT_RECEIVER_SPARSITY = 0.5;  % 0: strongest constraint, 1: no constraint
    DEFAULT_SENDER_SPARSITY   = 0.5;  % 0: strongest constraint, 1: no constraint

    % Number of Network Modes K:
    %   - []: default, uses N components.
    %   - Positive integer: number of components (in sparse_pmd, K can exceed N).
    DEFAULT_NUM_NETWORK_MODES = [];
    % =========================================================================

    if nargin < 1 || isempty(round_dir)
        round_dir = DEFAULT_DATASET;
    end
    round_dir = resolve_dataset_directory(round_dir);
    if nargin < 2 || isempty(M)
        M = 10;
    end
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');

    opts = parse_options(DEFAULT_NETWORK_DECOMPOSITION, DEFAULT_RECEIVER_SPARSITY, ...
        DEFAULT_SENDER_SPARSITY, DEFAULT_NUM_NETWORK_MODES, varargin{:});

    % 1. Run rigorous numerical Synthetic Validation Suite
    if opts.RunSyntheticValidation || opts.SyntheticTestOnly
        synth_results = run_rigorous_synthetic_validation(M, opts);
        if opts.SyntheticTestOnly
            results = synth_results;
            return;
        end
    end

    % 2. Setup output directory (separate directories for SVD vs Sparse PMD)
    if strcmpi(opts.NetworkDecomposition, 'sparse_pmd')
        output_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), 'global_joint_cp_rank1_profile_free_network_sparse_pmd');
    else
        output_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), 'global_joint_cp_rank1_profile_free_network_svd');
    end
    if opts.SaveOutputs && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    fprintf('[INFO] Starting Rank-1 Profile Free-Network CP Fit & %s Analysis\n', upper(opts.NetworkDecomposition));
    fprintf('  Round Dir: %s | Fourier M: %d | NumStarts: %d | Seed: %d\n', ...
        round_dir, M, opts.NumStarts, opts.RandomSeed);
    if strcmpi(opts.NetworkDecomposition, 'sparse_pmd')
        fprintf('  Sparse PMD Config: ReceiverSparsity = %.2f | SenderSparsity = %.2f | NumModes = %s\n', ...
            opts.ReceiverSparsity, opts.SenderSparsity, mat2str(opts.NumNetworkModes));
    end

    total_timer = tic;

    % 3. Call existing global_joint_cp_rank_sweep_sinusoidal_sender to fit free W_edge weights
    % (Existing file is called strictly as a read-only dependency, without modification)
    base = global_joint_cp_rank_sweep_sinusoidal_sender( ...
        round_dir, M, ...
        'Ranks', 1, ...
        'NumStarts', opts.NumStarts, ...
        'MaxIter', opts.MaxIter, ...
        'Tol', opts.Tol, ...
        'RandomSeed', opts.RandomSeed, ...
        'analysis_start_sec', opts.analysis_start_sec, ...
        'analysis_duration_sec', opts.analysis_duration_sec, ...
        'SaveOutputs', false);

    fit_R1 = base.fits([base.fits.rank] == 1);
    if isfield(base, 'phi_grid')
        fit_R1.phi_grid = base.phi_grid;
    else
        fit_R1.phi_grid = linspace(0, 2*pi, size(fit_R1.a_values, 1)).';
    end
    W_edge = fit_R1.W(:, 1); % Independent signed real edge weights

    interaction_meta = base.interaction_meta;
    C_tensor = base.C_tensor;
    P_edges = size(C_tensor, 3);
    L = size(C_tensor, 1);

    [agent_ids, target_indices, source_indices] = extract_agent_mappings(interaction_meta);
    N = numel(agent_ids);

    fprintf('[INFO] Loaded C_tensor: %dx%dx%d, N=%d agents.\n', L, L, P_edges, N);

    % 4. Construct N x N Network Matrix W with zero diagonal (Unmodified identification)
    W = zeros(N, N);
    for p = 1:P_edges
        i = target_indices(p);
        j = source_indices(p);
        W(i, j) = W_edge(p);
    end
    W(1:N+1:end) = 0; % Force diagonal explicitly to 0

    % Verification of edge indexing
    for p = 1:P_edges
        assert(W(target_indices(p), source_indices(p)) == W_edge(p), ...
            'Edge index mismatch between W_edge and W matrix!');
    end
    has_negative_edges = any(W(:) < 0);

    % 5. Perform SVD Analysis of Signed Network W (always computed as baseline reference)
    svd_res = analyze_network_svd(W, C_tensor, target_indices, source_indices, fit_R1.A(:,1), fit_R1.B(:,1));

    % 6. Perform Selected Network Decomposition (SVD or Sparse PMD)
    if strcmpi(opts.NetworkDecomposition, 'sparse_pmd')
        network_modes = analyze_network_sparse_pmd(W, C_tensor, target_indices, source_indices, ...
            fit_R1.A(:,1), fit_R1.B(:,1), opts, svd_res);
    else
        network_modes = wrap_svd_as_network_modes(svd_res, opts);
    end

    % 7. Load and verify Collective Phase Time Series if available
    phase_signals = load_collective_phase_signals(round_dir, agent_ids, fit_R1, network_modes, opts);

    % Assemble results struct
    results = struct();
    results.round_dir = round_dir;
    results.M = M;
    results.N = N;
    results.agent_ids = agent_ids;
    results.interaction_meta = interaction_meta;
    results.fit_R1 = fit_R1;
    results.W = W; % Identified signed network matrix (never overwritten)
    results.has_negative_edges = has_negative_edges;
    results.svd = svd_res; % True SVD result intact
    results.network_modes = network_modes; % Selected decomposition (SVD or PMD)
    results.phase_signals = phase_signals;
    results.output_dir = output_dir;
    results.opts = opts;
    results.runtime_seconds = toc(total_timer);

    % 8. Save PNG outputs, CSV summary tables, and optional compact MAT
    if opts.SaveOutputs
        save_all_figures(results, phase_signals, opts);
        save_analysis_csv_files(results, output_dir, opts);
        if isfield(opts, 'SaveCompactMat') && opts.SaveCompactMat
            save_compact_mat_file(results, output_dir, opts);
        end
    end

    % 9. Print concise summary
    print_final_summary(results);
end

% =========================================================================
% SVD DECOMPOSITION & DUAL ERROR METRIC CALCULATIONS
% =========================================================================

function svd_res = analyze_network_svd(W, C_tensor, target_indices, source_indices, A, B)
    N = size(W, 1);
    P_edges = size(C_tensor, 3);

    [U, S_mat, V] = svd(W);
    singular_values = diag(S_mat);
    num_rank = sum(singular_values > 1e-12 * (singular_values(1) + eps));

    % Fix sign orientation of singular vectors deterministically
    for l = 1:N
        [~, max_idx] = max(abs(U(:, l)));
        if U(max_idx, l) < 0
            U(:, l) = -U(:, l);
            V(:, l) = -V(:, l);
        end
    end

    sq_sv = singular_values.^2;
    total_sq = sum(sq_sv);
    if total_sq > 0
        singular_value_energy_share = sq_sv / total_sq;
    else
        singular_value_energy_share = zeros(N, 1);
    end
    cumulative_singular_value_energy_share = cumsum(singular_value_energy_share);

    norm_W_fro = norm(W, 'fro');
    norm_C_fro = norm(C_tensor(:));

    raw_svd_relative_error = zeros(N, 1);
    offdiag_svd_relative_error = zeros(N, 1);
    tensor_explained_fraction_K = zeros(N, 1);

    W_raw_K = zeros(N, N);
    AB_outer = A * B.';

    for K = 1:N
        W_raw_K = W_raw_K + singular_values(K) * (U(:, K) * V(:, K).');
        
        W_offdiag_K = W_raw_K;
        W_offdiag_K(1:N+1:end) = 0;

        % 1. Raw SVD relative error
        if norm_W_fro > 0
            raw_svd_relative_error(K) = norm(W - W_raw_K, 'fro') / norm_W_fro;
        else
            raw_svd_relative_error(K) = 0;
        end

        % 2. Off-diagonal relative error P_off(W - W_raw_K)
        if norm_W_fro > 0
            offdiag_svd_relative_error(K) = norm(W - W_offdiag_K, 'fro') / norm_W_fro;
        else
            offdiag_svd_relative_error(K) = 0;
        end

        % 3. Tensor explained energy using W_offdiag_K
        C_hat_K = zeros(size(C_tensor));
        for p = 1:P_edges
            w_val = W_offdiag_K(target_indices(p), source_indices(p));
            C_hat_K(:,:,p) = w_val * AB_outer;
        end

        if norm_C_fro > 0
            tensor_explained_fraction_K(K) = 1 - (norm(C_tensor(:) - C_hat_K(:))^2 / (norm_C_fro^2));
        else
            tensor_explained_fraction_K(K) = 0;
        end
    end

    % Rank K selection criteria
    K_90 = find(cumulative_singular_value_energy_share >= 0.90, 1); if isempty(K_90), K_90 = N; end
    K_95 = find(cumulative_singular_value_energy_share >= 0.95, 1); if isempty(K_95), K_95 = N; end
    K_99 = find(cumulative_singular_value_energy_share >= 0.99, 1); if isempty(K_99), K_99 = N; end

    svd_res = struct();
    svd_res.W = W;
    svd_res.U = U;
    svd_res.singular_values = singular_values;
    svd_res.V = V;
    svd_res.numerical_rank = num_rank;
    svd_res.singular_value_energy_share = singular_value_energy_share;
    svd_res.cumulative_singular_value_energy_share = cumulative_singular_value_energy_share;
    svd_res.raw_svd_relative_error = raw_svd_relative_error;
    svd_res.offdiag_svd_relative_error = offdiag_svd_relative_error;
    svd_res.tensor_explained_fraction_K = tensor_explained_fraction_K;
    svd_res.K_90 = K_90;
    svd_res.K_95 = K_95;
    svd_res.K_99 = K_99;
    svd_res.selected_K = K_95;
end

function modes = wrap_svd_as_network_modes(svd_res, opts)
    N = size(svd_res.W, 1);
    if isempty(opts.NumNetworkModes)
        K = N;
    else
        K = min(opts.NumNetworkModes, N);
    end
    modes = struct();
    modes.method = 'svd';
    modes.P = svd_res.U(:, 1:K);
    modes.Q = svd_res.V(:, 1:K);
    modes.d = svd_res.singular_values(1:K);
    modes.num_modes = K;
    modes.relative_error_K = svd_res.raw_svd_relative_error(1:K);
    modes.offdiag_relative_error_K = svd_res.offdiag_svd_relative_error(1:K);
    modes.reconstruction_fraction_K = 1 - (modes.relative_error_K).^2;
    modes.tensor_explained_fraction_K = svd_res.tensor_explained_fraction_K(1:K);
    modes.receiver_sparsity = 1.0;
    modes.sender_sparsity = 1.0;
    modes.c_p = sqrt(N);
    modes.c_q = sqrt(N);
    modes.L1_P = sum(abs(modes.P), 1).';
    modes.L2_P = sqrt(sum(modes.P.^2, 1)).';
    modes.L1_Q = sum(abs(modes.Q), 1).';
    modes.L2_Q = sqrt(sum(modes.Q.^2, 1)).';
    modes.nnz_P = sum(abs(modes.P) > 1e-10, 1).';
    modes.nnz_Q = sum(abs(modes.Q) > 1e-10, 1).';
    modes.reached_95 = any(modes.reconstruction_fraction_K >= 0.95);
    if modes.reached_95
        modes.K_95 = find(modes.reconstruction_fraction_K >= 0.95, 1);
        modes.selected_K = modes.K_95;
    else
        modes.K_95 = [];
        modes.selected_K = K;
    end
end

% =========================================================================
% SPARSE PMD DECOMPOSITION (Witten, Tibshirani, and Hastie 2009)
% =========================================================================

function pmd_res = analyze_network_sparse_pmd(W, C_tensor, target_indices, source_indices, A, B, opts, svd_res)
    N = size(W, 1);
    P_edges = size(C_tensor, 3);
    AB_outer = A * B.';
    norm_W_fro = norm(W, 'fro');
    norm_C_fro = norm(C_tensor(:));

    c_p = 1 + opts.ReceiverSparsity * (sqrt(N) - 1);
    c_q = 1 + opts.SenderSparsity   * (sqrt(N) - 1);

    if isempty(opts.NumNetworkModes)
        max_K = N;
    else
        max_K = opts.NumNetworkModes;
    end

    % If both sparsities are 1, PMD unconstrained optimization is identical to SVD
    if abs(opts.ReceiverSparsity - 1.0) < 1e-9 && abs(opts.SenderSparsity - 1.0) < 1e-9
        K = min(max_K, N);
        P = svd_res.U(:, 1:K);
        Q = svd_res.V(:, 1:K);
        d = svd_res.singular_values(1:K);
    else
        % PMD(L1, L1) Algorithm (Witten et al. 2009)
        P = zeros(N, max_K);
        Q = zeros(N, max_K);
        d = zeros(max_K, 1);

        R = W;
        actual_K = 0;

        for l = 1:max_K
            norm_R = norm(R, 'fro');
            if norm_R < 1e-10 * (norm_W_fro + eps)
                break;
            end

            % Find optimal rank-1 factor (p_l, q_l) using multi-start alternating updates
            [p_best, q_best, d_best] = pmd_rank1_multi_start(R, c_p, c_q, opts, l);

            if d_best <= 1e-12 * (norm_W_fro + eps) || norm(p_best) == 0 || norm(q_best) == 0
                break;
            end

            % Fix sign deterministically (maximum absolute element of p is positive)
            [~, max_idx] = max(abs(p_best));
            if p_best(max_idx) < 0
                p_best = -p_best;
                q_best = -q_best;
            end

            actual_K = actual_K + 1;
            P(:, actual_K) = p_best;
            Q(:, actual_K) = q_best;
            d(actual_K) = d_best;

            % Deflation: R_{l} = R_{l-1} - d_l * p_l * q_l.'
            % (Raw residual matrix is deflated; diagonal is not altered during PMD extraction)
            R = R - d_best * (p_best * q_best.');
        end

        P = P(:, 1:actual_K);
        Q = Q(:, 1:actual_K);
        d = d(1:actual_K);
        K = actual_K;
    end

    % Compute metrics for each K = 1:K
    relative_error_K = zeros(K, 1);
    offdiag_relative_error_K = zeros(K, 1);
    reconstruction_fraction_K = zeros(K, 1);
    tensor_explained_fraction_K = zeros(K, 1);

    W_raw_K = zeros(N, N);
    for l = 1:K
        % W_raw_K = sum_{j=1}^l d_j * p_j * q_j.'
        W_raw_K = W_raw_K + d(l) * (P(:, l) * Q(:, l).');

        % Note: W_l = d_l * p_l * q_l.' is rank-1, but W_{l, offdiag} is generally NOT rank-1.
        W_offdiag_K = W_raw_K;
        W_offdiag_K(1:N+1:end) = 0;

        if norm_W_fro > 0
            relative_error_K(l) = norm(W - W_raw_K, 'fro') / norm_W_fro;
            offdiag_relative_error_K(l) = norm(W - W_offdiag_K, 'fro') / norm_W_fro;
        else
            relative_error_K(l) = 0;
            offdiag_relative_error_K(l) = 0;
        end
        reconstruction_fraction_K(l) = 1 - relative_error_K(l)^2;

        C_hat_K = zeros(size(C_tensor));
        for p = 1:P_edges
            w_val = W_offdiag_K(target_indices(p), source_indices(p));
            C_hat_K(:,:,p) = w_val * AB_outer;
        end
        if norm_C_fro > 0
            tensor_explained_fraction_K(l) = 1 - (norm(C_tensor(:) - C_hat_K(:))^2 / (norm_C_fro^2));
        else
            tensor_explained_fraction_K(l) = 0;
        end
    end

    reached_95 = any(reconstruction_fraction_K >= 0.95);
    if reached_95
        K_95 = find(reconstruction_fraction_K >= 0.95, 1);
        selected_K = K_95;
    else
        K_95 = [];
        selected_K = K;
    end

    pmd_res = struct();
    pmd_res.method = 'sparse_pmd';
    pmd_res.P = P;
    pmd_res.Q = Q;
    pmd_res.d = d;
    pmd_res.num_modes = K;
    pmd_res.relative_error_K = relative_error_K;
    pmd_res.offdiag_relative_error_K = offdiag_relative_error_K;
    pmd_res.reconstruction_fraction_K = reconstruction_fraction_K;
    pmd_res.tensor_explained_fraction_K = tensor_explained_fraction_K;
    pmd_res.receiver_sparsity = opts.ReceiverSparsity;
    pmd_res.sender_sparsity = opts.SenderSparsity;
    pmd_res.c_p = c_p;
    pmd_res.c_q = c_q;
    pmd_res.L1_P = sum(abs(P), 1).';
    pmd_res.L2_P = sqrt(sum(P.^2, 1)).';
    pmd_res.L1_Q = sum(abs(Q), 1).';
    pmd_res.L2_Q = sqrt(sum(Q.^2, 1)).';
    pmd_res.nnz_P = sum(abs(P) > 1e-10, 1).';
    pmd_res.nnz_Q = sum(abs(Q) > 1e-10, 1).';
    pmd_res.reached_95 = reached_95;
    pmd_res.K_95 = K_95;
    pmd_res.selected_K = selected_K;
end

function [p_best, q_best, d_best] = pmd_rank1_multi_start(R, c_p, c_q, opts, mode_index)
    N = size(R, 1);
    p_best = zeros(N, 1);
    q_best = zeros(N, 1);
    d_best = -inf;

    num_starts = opts.PMDNumStarts;
    max_iter = opts.PMDMaxIter;
    tol = opts.PMDTol;

    % Build candidate starting vectors for q
    candidates = cell(num_starts, 1);
    c_idx = 0;

    % 1. Leading right singular vector of R
    try
        [~, ~, v_svd] = svds(R, 1);
        if ~isempty(v_svd) && norm(v_svd) > 0
            c_idx = c_idx + 1;
            candidates{c_idx} = v_svd / norm(v_svd);
        end
    catch
    end

    % 2. Standard coordinate unit vectors
    for j = 1:min(N, num_starts - c_idx)
        ej = zeros(N, 1); ej(j) = 1;
        c_idx = c_idx + 1;
        candidates{c_idx} = ej;
    end

    % 3. Deterministic pseudo-random starts
    saved_rng = rng;
    cleaner = onCleanup(@() rng(saved_rng));
    rng(opts.PMDSeed + mode_index * 1000, 'twister');

    while c_idx < num_starts
        c_idx = c_idx + 1;
        q_rand = randn(N, 1);
        if norm(q_rand) > 0
            candidates{c_idx} = q_rand / norm(q_rand);
        else
            candidates{c_idx} = ones(N, 1) / sqrt(N);
        end
    end

    % Run alternating soft-thresholded updates for each candidate
    for s = 1:num_starts
        q_init = candidates{s};
        if isempty(q_init), continue; end

        q = constrained_l1_l2_update(q_init, c_q);
        if norm(q) == 0, q = q_init / (norm(q_init) + eps); end

        p = zeros(N, 1);
        for it = 1:max_iter
            p_next = constrained_l1_l2_update(R * q, c_p);
            if norm(p_next) == 0, break; end

            q_next = constrained_l1_l2_update(R.' * p_next, c_q);
            if norm(q_next) == 0, break; end

            if norm(p_next - p) < tol && norm(q_next - q) < tol
                p = p_next; q = q_next;
                break;
            end
            p = p_next; q = q_next;
        end

        if norm(p) > 0 && norm(q) > 0
            d_val = p.' * R * q;
            if d_val < 0
                p = -p;
                d_val = -d_val;
            end
            if d_val > d_best
                d_best = d_val;
                p_best = p;
                q_best = q;
            end
        end
    end

    if d_best < 0
        d_best = 0;
        p_best = zeros(N, 1);
        q_best = zeros(N, 1);
    end
end

function u = constrained_l1_l2_update(z, c)
% CONSTRAINED_L1_L2_UPDATE
% Finds unit vector u (||u||_2 = 1) such that ||u||_1 <= c maximizing u.'*z.
% Formulated via soft thresholding S(z, Delta) = sign(z) .* max(|z| - Delta, 0).
    norm_z = norm(z, 2);
    if norm_z == 0 || ~isfinite(norm_z)
        u = zeros(size(z));
        return;
    end
    N_dim = numel(z);
    c = min(max(c, 1.0), sqrt(N_dim));

    % Case c == 1: exactly 1 non-zero element (maximum absolute entry)
    if abs(c - 1.0) < 1e-12
        u = zeros(size(z));
        [~, max_idx] = max(abs(z));
        s = sign(z(max_idx));
        if s == 0, s = 1; end
        u(max_idx) = s;
        return;
    end

    % Check if unconstrained unit vector satisfies L1 bound
    u_unconstrained = z / norm_z;
    if sum(abs(u_unconstrained)) <= c + 1e-12
        u = u_unconstrained;
        return;
    end

    % Search Delta in [0, max(|z|)) via bisection
    abs_z = abs(z);
    max_val = max(abs_z);
    low = 0;
    high = max_val;
    for iter = 1:60
        mid = (low + high) / 2;
        s_mid = sign(z) .* max(abs_z - mid, 0);
        norm_s = norm(s_mid, 2);
        if norm_s == 0
            high = mid;
            continue;
        end
        l1_ratio = sum(abs(s_mid)) / norm_s;
        if l1_ratio > c
            low = mid;
        else
            high = mid;
        end
        if (high - low) < 1e-14 * max_val
            break;
        end
    end
    delta = (low + high) / 2;
    s = sign(z) .* max(abs_z - delta, 0);
    norm_s = norm(s, 2);
    if norm_s > 0
        u = s / norm_s;
    else
        u = zeros(size(z));
        [~, max_idx] = max(abs(z));
        s_val = sign(z(max_idx));
        if s_val == 0, s_val = 1; end
        u(max_idx) = s_val;
    end
end

% =========================================================================
% RIGOROUS SYNTHETIC VALIDATION SUITE (11 Tests, Zero Hardcoded String Faking)
% =========================================================================

function synth_results = run_rigorous_synthetic_validation(M, opts)
    fprintf('\n=========================================================================\n');
    fprintf('   RUNNING RIGOROUS NUMERICAL SYNTHETIC VALIDATION SUITE (11 TESTS)\n');
    fprintf('=========================================================================\n');

    N = 4;
    L = 2 * M + 1;

    % 1. Create true non-rank-1 W matrix with mixed positive and negative values
    W_true = [ 0.0,  0.4, -0.3,  0.2; ...
              -0.5,  0.0,  0.6, -0.1; ...
               0.2, -0.4,  0.0,  0.5; ...
              -0.1,  0.3, -0.2,  0.0];
    
    assert(rank(W_true) > 1, 'Validation error: W_true must have rank > 1.');
    assert(any(W_true(:) < 0), 'Validation error: W_true must contain negative entries.');
    fprintf('  TEST 1: Non-rank-1 signed W_true handling       ... PASSED\n');

    % SVD of W_true
    [U_true, S_true, V_true] = svd(W_true);
    sigma_true = diag(S_true);

    % 2. Verify K=N SVD error is machine precision
    W_rec_N = U_true * S_true * V_true.';
    err_N = norm(W_true - W_rec_N, 'fro') / norm(W_true, 'fro');
    assert(err_N < 1e-14, 'Validation error: K=N SVD reconstruction error exceeds machine precision.');
    fprintf('  TEST 2: K=N SVD machine precision error (%.2e) ... PASSED\n', err_N);

    % 3. Verify rank-K SVD error matches theoretical remaining singular values
    for K = 1:N-1
        W_rec_K = U_true(:, 1:K) * S_true(1:K, 1:K) * V_true(:, 1:K).';
        err_empirical = norm(W_true - W_rec_K, 'fro');
        err_theoretical = sqrt(sum(sigma_true(K+1:end).^2));
        assert(abs(err_empirical - err_theoretical) < 1e-12, ...
            'Validation error: Rank-K SVD error does not match theoretical value from singular values!');
    end
    fprintf('  TEST 3: Theoretical vs empirical SVD error       ... PASSED\n');

    % 4. Verify SVD Collective Signal Mode Sum matches direct W_offdiag_K matrix product
    T_pts = 100;
    rng(42, 'twister');
    phi_ts = rand(T_pts, N) * 2 * pi;
    delta_true = 0.35;
    b_ts = sqrt(2) * cos(phi_ts - delta_true); % T_pts x N

    for K = 1:N
        W_raw_K = U_true(:, 1:K) * S_true(1:K, 1:K) * V_true(:, 1:K).';
        W_offdiag_K = W_raw_K;
        W_offdiag_K(1:N+1:end) = 0;

        direct_input = zeros(T_pts, N);
        for i = 1:N
            for j = 1:N
                if i ~= j
                    direct_input(:, i) = direct_input(:, i) + W_offdiag_K(i, j) * b_ts(:, j);
                end
            end
        end

        mode_input_sum = zeros(T_pts, N);
        for l = 1:K
            v_l = V_true(:, l);
            u_l = U_true(:, l);
            sigma_l = sigma_true(l);
            X_l = b_ts * v_l;
            for i = 1:N
                F_il = sigma_l * u_l(i) * (X_l - v_l(i) * b_ts(:, i));
                mode_input_sum(:, i) = mode_input_sum(:, i) + F_il;
            end
        end

        max_diff = max(abs(direct_input(:) - mode_input_sum(:)));
        assert(max_diff < 1e-12, 'Validation error: SVD collective signal mode sum mismatch!');
    end
    fprintf('  TEST 4: SVD Mode sum vs Direct matrix product    ... PASSED (diff < 1e-12)\n');

    % 5. Verify Sparse PMD with Sparsity=1 reduces exactly to SVD
    C_dummy = zeros(L, L, N*(N-1));
    [targets, sources] = find(~eye(N));
    A_dummy = ones(L, 1); B_dummy = ones(L, 1);
    opts_unconstrained = opts;
    opts_unconstrained.ReceiverSparsity = 1.0;
    opts_unconstrained.SenderSparsity = 1.0;
    opts_unconstrained.NumNetworkModes = N;

    svd_test = analyze_network_svd(W_true, C_dummy, targets, sources, A_dummy, B_dummy);
    pmd_unconstrained = analyze_network_sparse_pmd(W_true, C_dummy, targets, sources, ...
        A_dummy, B_dummy, opts_unconstrained, svd_test);

    W_pmd_rec = pmd_unconstrained.P * diag(pmd_unconstrained.d) * pmd_unconstrained.Q.';
    diff_svd_pmd = norm(W_true - W_pmd_rec, 'fro');
    assert(diff_svd_pmd < 1e-12, 'Validation error: PMD with Sparsity=1 does not match SVD!');
    fprintf('  TEST 5: PMD(1.0, 1.0) exact SVD equivalence      ... PASSED (diff < 1e-12)\n');

    % 6. Verify Strong L1/L2 constraint enforcement in Sparse PMD
    opts_sparse = opts;
    opts_sparse.ReceiverSparsity = 0.2;
    opts_sparse.SenderSparsity = 0.3;
    opts_sparse.NumNetworkModes = 3;
    pmd_sparse = analyze_network_sparse_pmd(W_true, C_dummy, targets, sources, ...
        A_dummy, B_dummy, opts_sparse, svd_test);

    for l = 1:pmd_sparse.num_modes
        p_l = pmd_sparse.P(:, l);
        q_l = pmd_sparse.Q(:, l);
        assert(abs(norm(p_l, 2) - 1.0) < 1e-10, 'Validation error: ||p_l||_2 must be 1.');
        assert(abs(norm(q_l, 2) - 1.0) < 1e-10, 'Validation error: ||q_l||_2 must be 1.');
        assert(norm(p_l, 1) <= pmd_sparse.c_p + 1e-10, 'Validation error: ||p_l||_1 exceeds c_p.');
        assert(norm(q_l, 1) <= pmd_sparse.c_q + 1e-10, 'Validation error: ||q_l||_1 exceeds c_q.');
    end
    fprintf('  TEST 6: Strong L1 and L2 constraint satisfaction ... PASSED\n');

    % 7. Verify PMD Raw and Offdiag Collective Signal Mode Sums match Direct Matrix Products
    for K = 1:pmd_sparse.num_modes
        P_K = pmd_sparse.P(:, 1:K);
        Q_K = pmd_sparse.Q(:, 1:K);
        d_K = pmd_sparse.d(1:K);

        W_raw_K = P_K * diag(d_K) * Q_K.';
        W_offdiag_K = W_raw_K;
        W_offdiag_K(1:N+1:end) = 0;

        % Raw direct vs mode sum
        raw_direct = b_ts * W_raw_K.';
        raw_mode_sum = zeros(T_pts, N);
        for l = 1:K
            X_l = b_ts * Q_K(:, l);
            for i = 1:N
                raw_mode_sum(:, i) = raw_mode_sum(:, i) + d_K(l) * P_K(i, l) * X_l;
            end
        end
        assert(max(abs(raw_direct(:) - raw_mode_sum(:))) < 1e-12, ...
            'Validation error: PMD raw mode sum does not match direct matrix product!');

        % Offdiag direct vs mode sum
        offdiag_direct = b_ts * W_offdiag_K.';
        offdiag_mode_sum = zeros(T_pts, N);
        for l = 1:K
            X_l = b_ts * Q_K(:, l);
            for i = 1:N
                offdiag_mode_sum(:, i) = offdiag_mode_sum(:, i) + d_K(l) * P_K(i, l) * (X_l - Q_K(i, l) * b_ts(:, i));
            end
        end
        assert(max(abs(offdiag_direct(:) - offdiag_mode_sum(:))) < 1e-12, ...
            'Validation error: PMD offdiag mode sum does not match direct matrix product!');
    end
    fprintf('  TEST 7: PMD Raw & Offdiag collective signal sums ... PASSED (diff < 1e-12)\n');

    % 8. Verify Signed weights handling (W with negative entries, d >= 0)
    assert(all(pmd_sparse.d >= 0), 'Validation error: PMD d coefficients must be non-negative.');
    fprintf('  TEST 8: Signed network weights and d >= 0       ... PASSED\n');

    % 9. Verify Edge cases: Zero matrix, c=1, tied coefficients do not produce NaN or Inf
    u_zero = constrained_l1_l2_update(zeros(4, 1), 1.5);
    assert(~any(isnan(u_zero(:))) && ~any(isinf(u_zero(:))), 'Edge case failed: zero vector update.');

    u_c1 = constrained_l1_l2_update([0.5; -0.8; 0.3; -0.1], 1.0);
    assert(abs(norm(u_c1, 1) - 1.0) < 1e-12 && sum(abs(u_c1) > 1e-10) == 1, 'Edge case failed: c=1.');

    u_tied = constrained_l1_l2_update([1.0; 1.0; -1.0; -1.0], 1.2);
    assert(~any(isnan(u_tied(:))) && ~any(isinf(u_tied(:))), 'Edge case failed: tied maximal coefficients.');

    pmd_zero_mat = analyze_network_sparse_pmd(zeros(4, 4), C_dummy, targets, sources, ...
        A_dummy, B_dummy, opts_sparse, svd_test);
    assert(pmd_zero_mat.num_modes == 0 || all(pmd_zero_mat.d == 0), 'Edge case failed: zero matrix decomposition.');
    fprintf('  TEST 9: Edge cases (zeros, c=1, ties, zero W)   ... PASSED (no NaN/Inf)\n');

    % 10. Verify Deterministic reproducibility with same seed
    pmd_rep1 = analyze_network_sparse_pmd(W_true, C_dummy, targets, sources, A_dummy, B_dummy, opts_sparse, svd_test);
    pmd_rep2 = analyze_network_sparse_pmd(W_true, C_dummy, targets, sources, A_dummy, B_dummy, opts_sparse, svd_test);
    assert(max(abs(pmd_rep1.P(:) - pmd_rep2.P(:))) < 1e-14, 'Validation error: PMD is not reproducible!');
    assert(max(abs(pmd_rep1.Q(:) - pmd_rep2.Q(:))) < 1e-14, 'Validation error: PMD is not reproducible!');
    assert(max(abs(pmd_rep1.d - pmd_rep2.d)) < 1e-14, 'Validation error: PMD is not reproducible!');
    fprintf('  TEST 10: Deterministic reproducibility with seed ... PASSED (diff < 1e-14)\n');

    % 11. Verify Fourier reconstruction of b(phi) matches sqrt(2)*cos(phi - delta)
    phi_grid = linspace(0, 2*pi, 256).';
    B_fourier = zeros(L, 1);
    n_vals = (-M:M).';
    B_fourier(n_vals == 1) = exp(-1i * delta_true) / sqrt(2);
    B_fourier(n_vals == -1) = exp(1i * delta_true) / sqrt(2);
    b_recon = real(exp(1i * phi_grid * n_vals.') * B_fourier);
    b_exact = sqrt(2) * cos(phi_grid - delta_true);
    assert(max(abs(b_recon - b_exact)) < 1e-12, 'Validation error: Fourier reconstruction of b(phi) mismatch.');
    fprintf('  TEST 11: Fourier b(phi) exactness                ... PASSED\n');

    fprintf('  SUMMARY: All 11 numerical synthetic tests PASSED (100%%).\n');
    fprintf('=========================================================================\n\n');

    synth_results = struct('synthetic_passed', true, 'num_passed', 11);
end

% =========================================================================
% CONSOLIDATED FIGURES GENERATION (4 PNGs Only)
% =========================================================================

function save_all_figures(results, phase_signals, opts)
    output_dir = results.output_dir;
    N = results.N;
    agent_ids = results.agent_ids;
    fit_R1 = results.fit_R1;
    svd_res = results.svd;
    network_modes = results.network_modes;
    W = results.W;
    is_pmd = strcmpi(network_modes.method, 'sparse_pmd');

    max_abs_W = max(abs(W(:)));
    if max_abs_W == 0, max_abs_W = 1.0; end

    % ---------------------------------------------------------------------
    % Figure 1: Model & Decomposition Summary
    % ---------------------------------------------------------------------
    fig1 = figure('Color', 'w', 'Position', [100, 100, 1150, 720], 'Visible', 'off');
    t_lay1 = tiledlayout(fig1, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    if is_pmd
        method_title = sprintf('Rank-1 Profile Free-Network Sparse PMD Summary (Rec: %.2f, Send: %.2f)', ...
            network_modes.receiver_sparsity, network_modes.sender_sparsity);
    else
        method_title = sprintf('Rank-1 Profile Free-Network Model & SVD Summary (R=1 Explained: %.2f%%)', ...
            100 * fit_R1.explained_fraction);
    end
    title(t_lay1, method_title, 'FontWeight', 'bold', 'FontSize', 12);

    % Panel 1: a(phi)
    ax1 = nexttile(t_lay1, 1); hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
    plot(ax1, fit_R1.phi_grid, fit_R1.a_values(:,1), 'LineWidth', 2.2, 'Color', [0, 0.447, 0.741]);
    xlabel(ax1, '\phi_{target}'); ylabel(ax1, 'a(\phi)'); title(ax1, 'Shared Receiver Profile a(\phi)');
    set_phase_axis(ax1);

    % Panel 2: b(phi)
    ax2 = nexttile(t_lay1, 2); hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
    plot(ax2, fit_R1.phi_grid, fit_R1.b_values(:,1), 'LineWidth', 2.2, 'Color', [0.85, 0.325, 0.098]);
    xlabel(ax2, '\phi_{source}'); ylabel(ax2, 'b(\phi)');
    delta_deg = rad2deg(fit_R1.delta(1));
    title(ax2, sprintf('Shared Sender Profile b(\\phi) (\\delta = %.1f^o)', delta_deg));
    set_phase_axis(ax2);

    % Panel 3: Signed Network W Heatmap
    ax3 = nexttile(t_lay1, 3);
    imagesc(ax3, W); colormap(ax3, make_symmetric_diverging_colormap(256));
    clim(ax3, [-max_abs_W, max_abs_W]); colorbar(ax3);
    xticks(ax3, 1:N); yticks(ax3, 1:N); xticklabels(ax3, agent_ids); yticklabels(ax3, agent_ids);
    xlabel(ax3, 'Sender Agent j'); ylabel(ax3, 'Receiver Agent i');
    title(ax3, 'Signed Effective Network W (W_{ii}=0)');

    % Panel 4: Signed Directed Graph W
    ax4 = nexttile(t_lay1, 4);
    plot_signed_directed_graph(ax4, W, 'Free Network W', agent_ids, results.round_dir);

    % Panel 5: Decomposition Coefficients
    ax5 = nexttile(t_lay1, 5); hold(ax5, 'on'); grid(ax5, 'on'); box(ax5, 'on');
    num_m = network_modes.num_modes;
    stem(ax5, 1:num_m, network_modes.d, 'LineWidth', 1.8, 'MarkerSize', 7, 'Color', [0.466, 0.674, 0.188]);
    if is_pmd
        xlabel(ax5, 'PMD Mode l'); ylabel(ax5, 'Component Coefficient d_l');
        title(ax5, sprintf('PMD Coefficients (Modes = %d)', num_m));
    else
        xlabel(ax5, 'SVD Mode l'); ylabel(ax5, 'Singular Value \sigma_l');
        title(ax5, sprintf('Singular Values (Rank = %d)', svd_res.numerical_rank));
    end
    xticks(ax5, 1:num_m);

    % Panel 6: Energy Share or Reconstruction Fraction
    ax6 = nexttile(t_lay1, 6); hold(ax6, 'on'); grid(ax6, 'on'); box(ax6, 'on');
    if is_pmd
        plot(ax6, 1:num_m, 100 * network_modes.reconstruction_fraction_K, '-bo', 'LineWidth', 1.8, 'MarkerSize', 6);
        yline(ax6, 95, 'r--', '95% Reconstruction', 'LineWidth', 1.2);
        xlabel(ax6, 'PMD Modes K'); ylabel(ax6, 'Reconstruction Fraction 1 - err_K^2 (%)');
        if network_modes.reached_95
            title(ax6, sprintf('PMD Reconstruction (95%% reached at K=%d)', network_modes.K_95));
        else
            title(ax6, 'PMD Reconstruction (95% 未達 / Not Reached)');
        end
        xticks(ax6, 1:num_m); ylim(ax6, [0, 105]);
    else
        yyaxis(ax6, 'left');
        bar(ax6, 1:N, 100 * svd_res.singular_value_energy_share, 0.5, 'FaceColor', [0.301, 0.745, 0.933]);
        ylabel(ax6, 'SVD Energy Share (%)');
        yyaxis(ax6, 'right');
        plot(ax6, 1:N, 100 * svd_res.cumulative_singular_value_energy_share, '-ro', 'LineWidth', 1.8, 'MarkerSize', 6);
        ylabel(ax6, 'Cumulative SVD Energy (%)');
        xlabel(ax6, 'SVD Mode l'); title(ax6, 'SVD Singular Value Energy Share');
        xticks(ax6, 1:N); ylim(ax6, [0, 105]);
    end

    if is_pmd
        saveas(fig1, fullfile(output_dir, 'rank1_profile_free_network_pmd_summary.png'));
    else
        saveas(fig1, fullfile(output_dir, 'rank1_profile_free_network_svd_summary.png'));
    end
    close(fig1);

    % ---------------------------------------------------------------------
    % Figure 2: Modes Visualization
    % ---------------------------------------------------------------------
    num_plot_modes = min(network_modes.num_modes, 6);
    fig2 = figure('Color', 'w', 'Position', [100, 100, 1250, max(260 * num_plot_modes, 500)], 'Visible', 'off');
    t_lay2 = tiledlayout(fig2, num_plot_modes, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    if is_pmd
        title(t_lay2, sprintf('Sparse PMD Modes (ReceiverSparsity=%.2f, SenderSparsity=%.2f): P_l, Q_l, Rank-1 W_l, & Off-diagonal Graph', ...
            network_modes.receiver_sparsity, network_modes.sender_sparsity), 'FontWeight', 'bold', 'FontSize', 11);
    else
        title(t_lay2, 'SVD Modes: Left Vector u_l, Right Vector v_l, Rank-1 Matrix W_l, & Off-diagonal Graph', ...
            'FontWeight', 'bold', 'FontSize', 12);
    end

    agent_str_labels = arrayfun(@(id) sprintf('%d', id), agent_ids, 'UniformOutput', false);

    for l = 1:num_plot_modes
        % Col 1: Receiver Weights P(:, l)
        ax_u = nexttile(t_lay2);
        bar(ax_u, 1:N, network_modes.P(:, l), 'FaceColor', [0, 0.447, 0.741]);
        xticks(ax_u, 1:N); xticklabels(ax_u, agent_str_labels); grid(ax_u, 'on'); box(ax_u, 'on');
        ylabel(ax_u, sprintf('Mode %d', l), 'FontWeight', 'bold');
        if is_pmd
            title(ax_u, sprintf('P_%d (nnz=%d, L1=%.2f)', l, network_modes.nnz_P(l), network_modes.L1_P(l)), 'FontSize', 8.5);
        else
            title(ax_u, sprintf('u_%d (Receiver Sensitivity)', l), 'FontSize', 9);
        end

        % Col 2: Sender Weights Q(:, l)
        ax_v = nexttile(t_lay2);
        bar(ax_v, 1:N, network_modes.Q(:, l), 'FaceColor', [0.85, 0.325, 0.098]);
        xticks(ax_v, 1:N); xticklabels(ax_v, agent_str_labels); grid(ax_v, 'on'); box(ax_v, 'on');
        if is_pmd
            title(ax_v, sprintf('Q_%d (nnz=%d, L1=%.2f)', l, network_modes.nnz_Q(l), network_modes.L1_Q(l)), 'FontSize', 8.5);
        else
            title(ax_v, sprintf('v_%d (Sender Contribution)', l), 'FontSize', 9);
        end

        % Col 3: True Rank-1 Mode Matrix W_l = d_l * P(:, l) * Q(:, l)^T
        W_l = network_modes.d(l) * (network_modes.P(:, l) * network_modes.Q(:, l).');
        max_abs_Wl = max(abs(W_l(:))); if max_abs_Wl == 0, max_abs_Wl = 1.0; end

        ax_hm = nexttile(t_lay2);
        imagesc(ax_hm, W_l); colormap(ax_hm, make_symmetric_diverging_colormap(256));
        clim(ax_hm, [-max_abs_Wl, max_abs_Wl]); colorbar(ax_hm);
        xticks(ax_hm, 1:N); yticks(ax_hm, 1:N); xticklabels(ax_hm, agent_ids); yticklabels(ax_hm, agent_ids);
        if is_pmd
            title(ax_hm, sprintf('Rank-1 Matrix W_%d = d_%d P_%d Q_%d^T (d_%d = %.3f)', ...
                l, l, l, l, l, network_modes.d(l)), 'FontSize', 8.5);
        else
            title(ax_hm, sprintf('Rank-1 Matrix W_%d = \\sigma_%d u_%d v_%d^T (\\sigma_%d = %.3f)', ...
                l, l, l, l, l, network_modes.d(l)), 'FontSize', 8.5);
        end

        % Col 4: Off-diagonal Contribution Network Graph
        % (Note: zeroing the diagonal of a rank-1 matrix produces a matrix that is generally NOT rank-1)
        W_l_off = W_l; W_l_off(1:N+1:end) = 0;
        ax_net = nexttile(t_lay2);
        if is_pmd
            plot_signed_directed_graph(ax_net, W_l_off, sprintf('Mode %d Off-diag Graph (Not Rank-1)', l), agent_ids, results.round_dir);
        else
            plot_signed_directed_graph(ax_net, W_l_off, sprintf('Mode %d Off-diagonal Graph', l), agent_ids, results.round_dir);
        end
    end

    if is_pmd
        saveas(fig2, fullfile(output_dir, 'rank1_profile_free_network_pmd_modes.png'));
    else
        saveas(fig2, fullfile(output_dir, 'rank1_profile_free_network_svd_modes.png'));
    end
    close(fig2);

    % ---------------------------------------------------------------------
    % Figure 3: Approximation & Comparative Error Analysis
    % ---------------------------------------------------------------------
    fig3 = figure('Color', 'w', 'Position', [100, 100, 1150, 720], 'Visible', 'off');
    t_lay3 = tiledlayout(fig3, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    if is_pmd
        title(t_lay3, sprintf('Sparse PMD Rank-K Approximation & SVD Baseline Comparison (Rec: %.2f, Send: %.2f)', ...
            network_modes.receiver_sparsity, network_modes.sender_sparsity), 'FontWeight', 'bold', 'FontSize', 11);
    else
        title(t_lay3, 'SVD Rank-K Approximation & Dual Relative Error Metrics', 'FontWeight', 'bold', 'FontSize', 12);
    end

    num_m = network_modes.num_modes;

    % Panel 1: Reconstruction Fraction
    ax_b1 = nexttile(t_lay3, 1); hold(ax_b1, 'on'); grid(ax_b1, 'on'); box(ax_b1, 'on');
    if is_pmd
        plot(ax_b1, 1:num_m, 100 * network_modes.reconstruction_fraction_K, '-bo', 'LineWidth', 2.0, 'DisplayName', 'PMD 1 - err_K^2');
        plot(ax_b1, 1:N, 100 * svd_res.cumulative_singular_value_energy_share, '--k', 'LineWidth', 1.5, 'DisplayName', 'SVD Cumulative Energy');
        yline(ax_b1, 95, 'r--', '95% Target', 'LineWidth', 1.2);
        xlabel(ax_b1, 'Modes K'); ylabel(ax_b1, 'Reconstruction Fraction (%)');
        if network_modes.reached_95
            title(ax_b1, sprintf('PMD Reconstruction (95%% at K=%d)', network_modes.K_95));
        else
            max_rec = max(network_modes.reconstruction_fraction_K);
            title(ax_b1, sprintf('PMD Reconstruction (95%% 未達, Max=%.1f%%)', 100 * max_rec));
        end
        xticks(ax_b1, 1:num_m); ylim(ax_b1, [0, 105]); legend(ax_b1, 'Location', 'southeast', 'FontSize', 7);
    else
        plot(ax_b1, 1:N, 100 * svd_res.cumulative_singular_value_energy_share, '-o', 'LineWidth', 2.0, 'Color', [0.85, 0.325, 0.098]);
        yline(ax_b1, 95, 'r--', '95% Energy', 'LineWidth', 1.2);
        xlabel(ax_b1, 'SVD Modes K'); ylabel(ax_b1, 'Cumulative SVD Energy (%)');
        title(ax_b1, 'SVD Singular Value Energy Share'); xticks(ax_b1, 1:N); ylim(ax_b1, [0, 105]);
    end

    % Panel 2: Relative Error Comparison (PMD vs SVD Baseline)
    ax_b2 = nexttile(t_lay3, 2); hold(ax_b2, 'on'); grid(ax_b2, 'on'); box(ax_b2, 'on');
    plot(ax_b2, 1:num_m, 100 * network_modes.relative_error_K, '-s', 'LineWidth', 2.0, ...
        'Color', [0, 0.447, 0.741], 'DisplayName', sprintf('%s Raw ||W - W_{raw,K}||_F', upper(network_modes.method)));
    plot(ax_b2, 1:num_m, 100 * network_modes.offdiag_relative_error_K, '-^', 'LineWidth', 2.0, ...
        'Color', [0.494, 0.184, 0.556], 'DisplayName', sprintf('%s Off-diag ||P_{off}(W - W_{raw,K})||_F', upper(network_modes.method)));
    if is_pmd
        % Show SVD baseline relative error for exact comparison
        plot(ax_b2, 1:min(num_m, N), 100 * svd_res.raw_svd_relative_error(1:min(num_m, N)), '--ko', ...
            'LineWidth', 1.5, 'DisplayName', 'SVD Baseline Raw Error');
    end
    xlabel(ax_b2, 'Modes K'); ylabel(ax_b2, 'Relative Error (%)');
    title(ax_b2, 'Relative Error Metrics & SVD Baseline'); xticks(ax_b2, 1:num_m);
    legend(ax_b2, 'Location', 'northeast', 'FontSize', 7);

    % Panel 3: Tensor Explained Energy for W_offdiag_K
    ax_b3 = nexttile(t_lay3, 3); hold(ax_b3, 'on'); grid(ax_b3, 'on'); box(ax_b3, 'on');
    plot(ax_b3, 1:num_m, 100 * network_modes.tensor_explained_fraction_K, '-d', 'LineWidth', 2.0, 'Color', [0.466, 0.674, 0.188]);
    yline(ax_b3, 100 * fit_R1.explained_fraction, 'b--', 'Full Free W Fit', 'LineWidth', 1.2);
    xlabel(ax_b3, 'Modes K'); ylabel(ax_b3, 'Tensor Explained Energy (%)');
    title(ax_b3, 'Tensor Explained Energy for W_{offdiag,K}'); xticks(ax_b3, 1:num_m);

    % Panel 4: Original Signed W
    ax_b4 = nexttile(t_lay3, 4);
    imagesc(ax_b4, W); colormap(ax_b4, make_symmetric_diverging_colormap(256));
    clim(ax_b4, [-max_abs_W, max_abs_W]); colorbar(ax_b4);
    xticks(ax_b4, 1:N); yticks(ax_b4, 1:N); xticklabels(ax_b4, agent_ids); yticklabels(ax_b4, agent_ids);
    xlabel(ax_b4, 'Sender j'); ylabel(ax_b4, 'Receiver i');
    title(ax_b4, 'Original Signed W');

    % Panel 5: Selected Rank-K W_offdiag_K
    K_sel = network_modes.selected_K;
    if K_sel > num_m, K_sel = num_m; end
    W_raw_sel = zeros(N, N);
    for l = 1:K_sel
        W_raw_sel = W_raw_sel + network_modes.d(l) * (network_modes.P(:, l) * network_modes.Q(:, l).');
    end
    W_off_sel = W_raw_sel; W_off_sel(1:N+1:end) = 0;

    ax_b5 = nexttile(t_lay3, 5);
    imagesc(ax_b5, W_off_sel); colormap(ax_b5, make_symmetric_diverging_colormap(256));
    clim(ax_b5, [-max_abs_W, max_abs_W]); colorbar(ax_b5);
    xticks(ax_b5, 1:N); yticks(ax_b5, 1:N); xticklabels(ax_b5, agent_ids); yticklabels(ax_b5, agent_ids);
    xlabel(ax_b5, 'Sender j'); ylabel(ax_b5, 'Receiver i');
    title(ax_b5, sprintf('Selected W_{offdiag,%d} (K=%d)', K_sel, K_sel));

    % Panel 6: Residual Matrix (W - W_offdiag_K)
    W_res = W - W_off_sel;
    ax_b6 = nexttile(t_lay3, 6);
    imagesc(ax_b6, W_res); colormap(ax_b6, make_symmetric_diverging_colormap(256));
    clim(ax_b6, [-max_abs_W, max_abs_W]); colorbar(ax_b6);
    xticks(ax_b6, 1:N); yticks(ax_b6, 1:N); xticklabels(ax_b6, agent_ids); yticklabels(ax_b6, agent_ids);
    xlabel(ax_b6, 'Sender j'); ylabel(ax_b6, 'Receiver i');
    title(ax_b6, sprintf('Residual (Off-diag Error: %.2f%%)', 100 * network_modes.offdiag_relative_error_K(K_sel)));

    if is_pmd
        saveas(fig3, fullfile(output_dir, 'rank1_profile_free_network_pmd_approximation.png'));
    else
        saveas(fig3, fullfile(output_dir, 'rank1_profile_free_network_svd_approximation.png'));
    end
    close(fig3);

    % ---------------------------------------------------------------------
    % Figure 4: Collective Signals (If available)
    % ---------------------------------------------------------------------
    if phase_signals.available
        fig4 = figure('Color', 'w', 'Position', [100, 100, 1150, 780], 'Visible', 'off');
        t_lay4 = tiledlayout(fig4, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        title(t_lay4, sprintf('Collective Sender Signals X_l(t), Complex Order Z_l(t) & Off-diagonal Receiver Input (%s)', ...
            upper(network_modes.method)), 'FontWeight', 'bold', 'FontSize', 12);

        t_vec = phase_signals.time_sec;
        num_sig_modes = size(phase_signals.X_l, 2);
        colors_l = lines(num_sig_modes);

        % Subplot 1: Collective Signals X_l(t) = sum_j Q(j,l) b(phi_j(t))
        ax_s1 = nexttile(t_lay4, 1); hold(ax_s1, 'on'); grid(ax_s1, 'on'); box(ax_s1, 'on');
        for l = 1:num_sig_modes
            plot(ax_s1, t_vec, phase_signals.X_l(:, l), 'LineWidth', 1.2, 'Color', colors_l(l,:), ...
                'DisplayName', sprintf('X_%d(t) (Mode %d)', l, l));
        end
        ylabel(ax_s1, 'X_l(t)'); legend(ax_s1, 'Location', 'eastoutside', 'FontSize', 7);
        title(ax_s1, 'Collective Sender Signals X_l(t) = \sum_j q_{jl} b(\phi_j(t))');

        % Subplot 2: Order Parameter Magnitude |Z_l(t)|
        ax_s2 = nexttile(t_lay4, 2); hold(ax_s2, 'on'); grid(ax_s2, 'on'); box(ax_s2, 'on');
        for l = 1:num_sig_modes
            plot(ax_s2, t_vec, abs(phase_signals.Z_l(:, l)), 'LineWidth', 1.2, 'Color', colors_l(l,:), ...
                'DisplayName', sprintf('|Z_%d(t)|', l));
        end
        ylabel(ax_s2, '|Z_l(t)|'); legend(ax_s2, 'Location', 'eastoutside', 'FontSize', 7);
        title(ax_s2, 'Mode Order Parameter Magnitude |Z_l(t)| = |\sum_j q_{jl} e^{i \phi_j(t)}|');

        % Subplot 3: Phase arg Z_l(t)
        ax_s3 = nexttile(t_lay4, 3); hold(ax_s3, 'on'); grid(ax_s3, 'on'); box(ax_s3, 'on');
        for l = 1:num_sig_modes
            plot(ax_s3, t_vec, angle(phase_signals.Z_l(:, l)), 'LineWidth', 1.0, 'Color', colors_l(l,:), ...
                'DisplayName', sprintf('arg Z_%d(t)', l));
        end
        ylabel(ax_s3, 'arg Z_l(t) (rad)'); legend(ax_s3, 'Location', 'eastoutside', 'FontSize', 7);
        title(ax_s3, 'Mode Order Parameter Phase Angle arg Z_l(t)');

        % Subplot 4: Mode Off-diagonal Contributions F_{il}(t) to Receiver Input
        ax_s4 = nexttile(t_lay4, 4); hold(ax_s4, 'on'); grid(ax_s4, 'on'); box(ax_s4, 'on');
        for l = 1:num_sig_modes
            mean_F_l = mean(phase_signals.F_offdiag(:, :, l), 2);
            plot(ax_s4, t_vec, mean_F_l, 'LineWidth', 1.2, 'Color', colors_l(l,:), ...
                'DisplayName', sprintf('Mode %d Mean F_{offdiag,l}(t)', l));
        end
        xlabel(ax_s4, 'Time (s)'); ylabel(ax_s4, 'Receiver Contribution F_{il}(t)');
        legend(ax_s4, 'Location', 'eastoutside', 'FontSize', 7);
        title(ax_s4, 'Off-diagonal Mode Receiver Inputs F_{il}(t) = d_l p_{il} [ X_l(t) - q_{il} b(\phi_i(t)) ]');

        saveas(fig4, fullfile(output_dir, 'rank1_profile_free_network_collective_signals.png'));
        close(fig4);
    else
        fprintf('[INFO] Collective signals figure skipped: %s\n', phase_signals.reason);
    end
end

% =========================================================================
% LIGHTWEIGHT COMPACT MAT FILE SAVER (Only if SaveCompactMat = true)
% =========================================================================

function save_compact_mat_file(results, output_dir, opts)
    is_pmd = strcmpi(results.network_modes.method, 'sparse_pmd');
    if is_pmd
        mat_path = fullfile(output_dir, 'rank1_profile_free_network_sparse_pmd_results.mat');
    else
        mat_path = fullfile(output_dir, 'rank1_profile_free_network_svd_results.mat');
    end
    fit_R1 = results.fit_R1;
    modes = results.network_modes;

    compact_struct = struct( ...
        'A', fit_R1.A(:,1), ...
        'B', fit_R1.B(:,1), ...
        'delta', fit_R1.delta(1), ...
        'W_edge', fit_R1.W(:,1), ...
        'W', results.W, ...
        'P', modes.P, ...
        'Q', modes.Q, ...
        'd', modes.d, ...
        'method', modes.method, ...
        'relative_error_K', modes.relative_error_K, ...
        'offdiag_relative_error_K', modes.offdiag_relative_error_K, ...
        'tensor_explained_fraction_K', modes.tensor_explained_fraction_K, ...
        'agent_ids', results.agent_ids, ...
        'opts', opts ...
    );

    save(mat_path, '-struct', 'compact_struct', '-v7.3');
end

% =========================================================================
% DETAILED CSV EXPORTER FOR DOWNSTREAM ANALYSIS (a(phi), delta, u, v, W)
% =========================================================================

function save_analysis_csv_files(results, output_dir, opts)
    fit_R1 = results.fit_R1;
    modes = results.network_modes;
    agent_ids = results.agent_ids(:);
    N = numel(agent_ids);
    is_pmd = strcmpi(modes.method, 'sparse_pmd');

    try
        % 1. Target Receiver Profile a(phi)
        phi_grid = fit_R1.phi_grid(:);
        a_phi = real(exp(1i * phi_grid * (-results.M : results.M)) * fit_R1.A(:, 1));
        t_a = table(phi_grid, a_phi, 'VariableNames', {'phi', 'a_phi'});
        writetable(t_a, fullfile(output_dir, 'target_receiver_profile_a_phi.csv'));

        % 2. Sender Phase Shift delta
        delta_rad = fit_R1.delta(1);
        delta_deg = rad2deg(delta_rad);
        delta_over_pi = delta_rad / pi;
        t_delta = table(delta_rad, delta_deg, delta_over_pi, 'VariableNames', {'delta_rad', 'delta_deg', 'delta_over_pi'});
        writetable(t_delta, fullfile(output_dir, 'sender_phase_shift_delta.csv'));

        % 3. Network Modes Summary CSV (Method-specific naming & format)
        if is_pmd
            mode_l = (1:modes.num_modes).';
            component_d = modes.d;
            reconstruction_fraction = modes.reconstruction_fraction_K;
            raw_relative_error = modes.relative_error_K;
            offdiag_relative_error = modes.offdiag_relative_error_K;
            receiver_l1 = modes.L1_P;
            receiver_l2 = modes.L2_P;
            receiver_nnz = modes.nnz_P;
            sender_l1 = modes.L1_Q;
            sender_l2 = modes.L2_Q;
            sender_nnz = modes.nnz_Q;
            t_pmd_modes = table(mode_l, component_d, reconstruction_fraction, raw_relative_error, ...
                offdiag_relative_error, receiver_l1, receiver_l2, receiver_nnz, sender_l1, sender_l2, sender_nnz, ...
                'VariableNames', {'mode_l', 'component_d', 'reconstruction_fraction', 'raw_relative_error', ...
                'offdiag_relative_error', 'receiver_l1', 'receiver_l2', 'receiver_nnz', 'sender_l1', 'sender_l2', 'sender_nnz'});
            writetable(t_pmd_modes, fullfile(output_dir, 'network_pmd_modes_summary.csv'));
        else
            svd_res = results.svd;
            mode_l = (1:N).';
            singular_value_sigma = svd_res.singular_values;
            energy_share_eta = svd_res.singular_value_energy_share;
            cum_energy_share = svd_res.cumulative_singular_value_energy_share;
            t_modes = table(mode_l, singular_value_sigma, energy_share_eta, cum_energy_share, ...
                'VariableNames', {'mode_l', 'singular_value_sigma', 'energy_share_eta', 'cum_energy_share'});
            writetable(t_modes, fullfile(output_dir, 'network_svd_modes_summary.csv'));
        end

        % 4. Agent Contributions (Sender & Receiver)
        t_agent = table(agent_ids, 'VariableNames', {'agent_id'});
        if is_pmd
            for l = 1:modes.num_modes
                t_agent.(sprintf('receiver_P_mode%d', l)) = modes.P(:, l);
                t_agent.(sprintf('sender_Q_mode%d', l))   = modes.Q(:, l);
            end
            if modes.num_modes >= 1
                t_agent.receiver_mode1_weighted = modes.d(1) * abs(modes.P(:, 1));
                t_agent.sender_mode1_weighted   = modes.d(1) * abs(modes.Q(:, 1));
            end
            t_agent.receiver_total_weight = sqrt(sum(results.W.^2, 2));
            t_agent.sender_total_weight   = sqrt(sum(results.W.^2, 1)).';
            writetable(t_agent, fullfile(output_dir, 'agent_pmd_contributions.csv'));
        else
            svd_res = results.svd;
            for l = 1:N
                t_agent.(sprintf('receiver_u_mode%d', l)) = svd_res.U(:, l);
                t_agent.(sprintf('sender_v_mode%d', l))   = svd_res.V(:, l);
            end
            t_agent.receiver_mode1_weighted = svd_res.singular_values(1) * abs(svd_res.U(:, 1));
            t_agent.sender_mode1_weighted   = svd_res.singular_values(1) * abs(svd_res.V(:, 1));
            t_agent.receiver_total_weight   = sqrt(sum(results.W.^2, 2));
            t_agent.sender_total_weight     = sqrt(sum(results.W.^2, 1)).';
            writetable(t_agent, fullfile(output_dir, 'agent_svd_contributions.csv'));
        end

        % 5. Network Coupling Matrix W (Row: Receiver, Column: Sender)
        W_mat = results.W;
        col_names = arrayfun(@(a) sprintf('sender_agent_%d', a), agent_ids, 'UniformOutput', false);
        row_names = arrayfun(@(a) sprintf('receiver_agent_%d', a), agent_ids, 'UniformOutput', false);
        t_W = table(row_names, 'VariableNames', {'receiver_agent'});
        for j = 1:N
            t_W.(col_names{j}) = W_mat(:, j);
        end
        writetable(t_W, fullfile(output_dir, 'network_coupling_matrix_W.csv'));

        % 6. Fourier Coefficients Vector A
        m_vals = (-results.M : results.M).';
        A_vec = fit_R1.A(:, 1);
        t_A = table(m_vals, real(A_vec), imag(A_vec), abs(A_vec), angle(A_vec), ...
            'VariableNames', {'m', 'Re_A', 'Im_A', 'Abs_A', 'Angle_A'});
        writetable(t_A, fullfile(output_dir, 'fourier_coefficients_A.csv'));

        fprintf('[INFO] Saved detailed analysis CSV files to:\n  %s\n', output_dir);
    catch ME
        fprintf('[WARNING] Failed to save CSV files: %s\n', ME.message);
    end
end

% =========================================================================
% SIGNED DIRECTED GRAPH PLOTTING HELPER
% =========================================================================

function plot_signed_directed_graph(ax, W_mat, title_str, agent_ids, round_dir)
    if nargin < 5
        round_dir = '';
    end
    N = size(W_mat, 1);
    s_idx = []; t_idx = []; weights = [];
    for i = 1:N
        for j = 1:N
            if i ~= j && abs(W_mat(i, j)) > 1e-10
                t_idx = [t_idx; i];
                s_idx = [s_idx; j];
                weights = [weights; W_mat(i, j)];
            end
        end
    end
    if isempty(s_idx)
        s_idx = 1; t_idx = 2; weights = 0;
    end

    G = digraph(s_idx, t_idx, abs(weights), N);

    layout_agent_ids = agent_ids;
    has_9_10 = any(agent_ids == 9) && any(agent_ids == 10) && ~any(agent_ids == 12);

    if contains(lower(round_dir), 'sstick') || (has_9_10 && ~contains(lower(round_dir), 'round6'))
        % SStick: Swap node 9 and node 10, rotate 45 degrees clockwise (-45 deg)
        idx9 = find(layout_agent_ids == 9, 1);
        idx10 = find(layout_agent_ids == 10, 1);
        if ~isempty(idx9) && ~isempty(idx10)
            layout_agent_ids(idx9) = 10;
            layout_agent_ids(idx10) = 9;
        end
        rot_offset_deg = -45;
    else
        % Round6: Swap node 10 and node 12, rotate 30 degrees clockwise (-30 deg)
        idx10 = find(layout_agent_ids == 10, 1);
        idx12 = find(layout_agent_ids == 12, 1);
        if ~isempty(idx10) && ~isempty(idx12)
            layout_agent_ids(idx10) = 12;
            layout_agent_ids(idx12) = 10;
        end
        rot_offset_deg = -30;
    end

    th = linspace(0, 2*pi, N+1) + deg2rad(rot_offset_deg);
    th(end) = [];

    x_pos = zeros(1, N);
    y_pos = zeros(1, N);
    for k = 1:N
        pos_idx = find(layout_agent_ids == agent_ids(k), 1);
        x_pos(k) = cos(th(pos_idx));
        y_pos(k) = sin(th(pos_idx));
    end

    node_labels = arrayfun(@num2str, agent_ids, 'UniformOutput', false);

    num_edges = numel(weights);
    abs_weights = abs(weights);
    max_w = max(abs_weights);
    if max_w == 0, max_w = 1.0; end
    rel_weights = abs_weights / max_w;

    edge_line_widths = 0.5 + 4.5 * rel_weights;
    edge_arrow_sizes = 6 + 10 * rel_weights;

    h_g = plot(ax, G, 'XData', x_pos, 'YData', y_pos, 'NodeLabel', node_labels, ...
        'NodeColor', [0.2 0.6 0.8], 'MarkerSize', 8);

    h_g.LineWidth = edge_line_widths;
    h_g.ArrowSize = edge_arrow_sizes;

    edge_colors = zeros(num_edges, 3);
    for e = 1:num_edges
        if weights(e) >= 0
            edge_colors(e, :) = [0.85, 0.325, 0.098]; % Red for positive
        else
            edge_colors(e, :) = [0.0, 0.447, 0.741];  % Blue for negative
        end
    end
    h_g.EdgeColor = edge_colors;

    title(ax, title_str, 'FontSize', 9);
    axis(ax, 'equal'); axis(ax, 'off');
end

% =========================================================================
% PHASE TIME SERIES & COLLECTIVE SIGNALS LOADER
% =========================================================================

function phase_signals = load_collective_phase_signals(round_dir, agent_ids, fit_R1, network_modes, opts)
    phase_signals = struct('available', false, 'reason', 'Phase time series data not found in cache.');
    cache_path = fullfile(round_dir, 'phase_analysis_cache.mat');
    if ~exist(cache_path, 'file')
        return;
    end

    try
        data = load(cache_path);
        if ~isfield(data, 'time_sec') || ~isfield(data, 'phase_matrix')
            phase_signals.reason = 'Missing time_sec or phase_matrix in cache.';
            return;
        end

        t_raw = data.time_sec(:);
        phases_raw = data.phase_matrix; % T x N_cache

        % Check agent ID correspondence
        N = numel(agent_ids);
        if isfield(data, 'agent_ids')
            cache_agent_ids = data.agent_ids(:).';
            if ~isequal(sort(cache_agent_ids), sort(agent_ids))
                phase_signals.reason = sprintf('Agent IDs mismatch between cache (%s) and model (%s).', ...
                    mat2str(cache_agent_ids), mat2str(agent_ids));
                return;
            end
            % Permute columns of phase_matrix to match model agent_ids order
            perm_indices = zeros(1, N);
            for k = 1:N
                perm_indices(k) = find(cache_agent_ids == agent_ids(k), 1);
            end
            phases_aligned = phases_raw(:, perm_indices);
        else
            if size(phases_raw, 2) ~= N
                phase_signals.reason = sprintf('Phase matrix column count (%d) does not match N (%d).', ...
                    size(phases_raw, 2), N);
                return;
            end
            phases_aligned = phases_raw;
        end

        step = max(1, floor(numel(t_raw) / 800));
        idx_sub = 1:step:numel(t_raw);
        t_sub = t_raw(idx_sub);
        phi_sub = phases_aligned(idx_sub, :); % T x N

        T = numel(t_sub);
        K = network_modes.num_modes;
        P = network_modes.P;
        Q = network_modes.Q;
        d = network_modes.d;
        delta = fit_R1.delta(1);

        % b_i(t) = sqrt(2) * cos(phi_i(t) - delta)
        b_ts = sqrt(2) * cos(phi_sub - delta); % T x N

        Z_l = zeros(T, K);
        X_l = zeros(T, K);
        F_raw = zeros(T, N, K);
        F_offdiag = zeros(T, N, K);

        for l = 1:K
            q_l = Q(:, l);
            p_l = P(:, l);
            d_l = d(l);

            % Z_l(t) = exp(1i * phi_ts) * Q(:, l)
            Z_l(:, l) = exp(1i * phi_sub) * q_l;

            % X_l(t) = b_ts * Q(:, l)
            X_l(:, l) = b_ts * q_l;

            % F_raw(:, i, l) = d(l) * P(i, l) * X_l
            % F_offdiag(:, i, l) = d(l) * P(i, l) * (X_l - Q(i, l) * b_ts(:, i))
            for i = 1:N
                F_raw(:, i, l) = d_l * p_l(i) * X_l(:, l);
                F_offdiag(:, i, l) = d_l * p_l(i) * (X_l(:, l) - q_l(i) * b_ts(:, i));
            end
        end

        % Verify analytical identity: sum_{l=1}^K F_offdiag(:,:,l) == b_ts * W_offdiag_K.'
        W_raw_K = P * diag(d) * Q.';
        W_offdiag_K = W_raw_K;
        W_offdiag_K(1:N+1:end) = 0;
        expected_offdiag = b_ts * W_offdiag_K.';
        actual_offdiag = sum(F_offdiag, 3);
        max_diff = max(abs(actual_offdiag(:) - expected_offdiag(:)));
        if max_diff > 1e-10
            warning('Collective signal mode sum vs direct product mismatch: diff = %.2e', max_diff);
        end

        phase_signals.available = true;
        phase_signals.time_sec = t_sub;
        phase_signals.X_l = X_l;
        phase_signals.Z_l = Z_l;
        phase_signals.F_raw = F_raw;
        phase_signals.F_offdiag = F_offdiag;
        phase_signals.F_il_tensor = F_offdiag; % for backward compatibility
        phase_signals.max_diff_identity = max_diff;
    catch ME
        phase_signals.available = false;
        phase_signals.reason = sprintf('Exception during phase signal processing: %s', ME.message);
    end
end

% =========================================================================
% CONSOLE REPORTING & OPTIONS PARSING HELPERS
% =========================================================================

function print_final_summary(results)
    fit_R1 = results.fit_R1;
    modes = results.network_modes;
    is_pmd = strcmpi(modes.method, 'sparse_pmd');

    fprintf('\n=========================================================================\n');
    if is_pmd
        fprintf('  SUMMARY REPORT: RANK-1 PROFILE FREE-NETWORK SPARSE PMD ANALYSIS\n');
    else
        fprintf('  SUMMARY REPORT: RANK-1 PROFILE FREE-NETWORK SVD ANALYSIS\n');
    end
    fprintf('=========================================================================\n');
    fprintf('1. STATUS: Execution completed successfully.\n');
    fprintf('2. OUTPUT DIRECTORY: %s\n', results.output_dir);
    fprintf('3. DECOMPOSITION METHOD: %s\n', upper(modes.method));
    if is_pmd
        fprintf('   - Receiver Sparsity Parameter: %.2f (c_p = %.2f)\n', modes.receiver_sparsity, modes.c_p);
        fprintf('   - Sender Sparsity Parameter:   %.2f (c_q = %.2f)\n', modes.sender_sparsity, modes.c_q);
    end
    fprintf('4. FREE NETWORK EXPLAINED FRACTION: %.2f%%\n', 100 * fit_R1.explained_fraction);
    fprintf('5. NETWORK PROPERTIES:\n');
    fprintf('   - Matrix Size: %dx%d | Active Components K: %d\n', results.N, results.N, modes.num_modes);
    fprintf('   - Signed Edges: Negative edges present = %s\n', mat2str(results.has_negative_edges));
    fprintf('6. COMPONENT COEFFICIENTS & METRICS:\n');
    for l = 1:modes.num_modes
        if is_pmd
            fprintf('   - Mode %2d: d = %7.4f | Rec Fraction = %5.2f%% | Rel Err = %5.2f%% | nnz(P)=%d/%d, nnz(Q)=%d/%d\n', ...
                l, modes.d(l), 100 * modes.reconstruction_fraction_K(l), 100 * modes.relative_error_K(l), ...
                modes.nnz_P(l), results.N, modes.nnz_Q(l), results.N);
        else
            svd_res = results.svd;
            fprintf('   - Mode %2d: sigma = %7.4f | Energy Share = %5.2f%% | Cumulative = %5.2f%%\n', ...
                l, svd_res.singular_values(l), 100 * svd_res.singular_value_energy_share(l), ...
                100 * svd_res.cumulative_singular_value_energy_share(l));
        end
    end
    fprintf('7. MODE SELECTION (95%% THRESHOLD):\n');
    if is_pmd
        if modes.reached_95
            fprintf('   - 95%% Reconstruction Fraction Reached: K = %d\n', modes.K_95);
        else
            fprintf('   - 95%% Reconstruction Fraction: 未達 (Not Reached) [Max = %.2f%% at K=%d]\n', ...
                100 * max(modes.reconstruction_fraction_K), modes.num_modes);
        end
    else
        fprintf('   - 95%% Cumulative Energy : K = %d (Selected K)\n', results.svd.K_95);
    end
    fprintf('8. DUAL RELATIVE ERROR & TENSOR EXPLAINED FRACTION (Selected K=%d):\n', modes.selected_K);
    fprintf('   - Raw Relative Error ||W - W_{raw,K}||_F / ||W||_F     = %.2f%%\n', ...
        100 * modes.relative_error_K(modes.selected_K));
    fprintf('   - Off-diag Relative Error ||P_{off}(W - W_{raw,K})||_F / ||W||_F = %.2f%%\n', ...
        100 * modes.offdiag_relative_error_K(modes.selected_K));
    fprintf('   - Tensor Energy Explained for W_{offdiag,K}               = %.2f%%\n', ...
        100 * modes.tensor_explained_fraction_K(modes.selected_K));
    fprintf('=========================================================================\n\n');
end

function opts = parse_options(default_decomp, default_rec_sp, default_send_sp, default_num_modes, varargin)
    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'NetworkDecomposition', default_decomp, @(x) ischar(x) || isstring(x));
    addParameter(parser, 'ReceiverSparsity', default_rec_sp, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(parser, 'SenderSparsity', default_send_sp, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(parser, 'NumNetworkModes', default_num_modes, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));

    % PMD specific options
    addParameter(parser, 'PMDMaxIter', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(parser, 'PMDTol', 1e-7, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'PMDNumStarts', 10, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(parser, 'PMDSeed', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);

    % Optimization & Run options
    addParameter(parser, 'NumStarts', 20, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(parser, 'MaxIter', 1000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(parser, 'Tol', 1e-10, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'RandomSeed', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(parser, 'SaveOutputs', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SaveCompactMat', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'analysis_start_sec', 6.5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
    addParameter(parser, 'analysis_duration_sec', 80, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(parser, 'RunSyntheticValidation', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SyntheticTestOnly', false, @(x) islogical(x) && isscalar(x));

    parse(parser, varargin{:});
    opts = parser.Results;
    opts.NetworkDecomposition = lower(char(opts.NetworkDecomposition));
    assert(ismember(opts.NetworkDecomposition, {'svd', 'sparse_pmd'}), ...
        'Invalid NetworkDecomposition: must be ''svd'' or ''sparse_pmd''.');
end

function [agent_ids, target_indices, source_indices] = extract_agent_mappings(interaction_meta)
    P = numel(interaction_meta);
    raw_targets = zeros(P, 1);
    raw_sources = zeros(P, 1);
    for p = 1:P
        meta = get_meta_struct(interaction_meta, p);
        raw_targets(p) = get_target_id(meta);
        raw_sources(p) = get_source_id(meta);
    end
    agent_ids = unique([raw_targets; raw_sources]);
    agent_ids = agent_ids(:).';
    target_indices = zeros(P, 1);
    source_indices = zeros(P, 1);
    for p = 1:P
        target_indices(p) = find(agent_ids == raw_targets(p), 1);
        source_indices(p) = find(agent_ids == raw_sources(p), 1);
    end
end

function meta = get_meta_struct(interaction_meta, p)
    if iscell(interaction_meta)
        meta = interaction_meta{p};
    else
        meta = interaction_meta(p);
    end
end

function target_id = get_target_id(meta)
    if isfield(meta, 'target_id')
        target_id = meta.target_id;
    elseif isfield(meta, 'target')
        target_id = meta.target;
    elseif isfield(meta, 'i')
        target_id = meta.i;
    else
        error('Cannot determine target agent ID.');
    end
end

function source_id = get_source_id(meta)
    if isfield(meta, 'source_id')
        source_id = meta.source_id;
    elseif isfield(meta, 'source')
        source_id = meta.source;
    elseif isfield(meta, 'j')
        source_id = meta.j;
    else
        error('Cannot determine source agent ID.');
    end
end

function set_phase_axis(ax)
    xlim(ax, [0, 2*pi]);
    set(ax, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi], ...
        'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
end

function cmap = make_symmetric_diverging_colormap(N)
    if nargin < 1, N = 256; end
    r = [linspace(0.2, 1, N/2), linspace(1, 0.8, N/2)].';
    g = [linspace(0.2, 1, N/2), linspace(1, 0.8, N/2)].';
    b = [linspace(0.8, 1, N/2), linspace(1, 0.2, N/2)].';
    cmap = [r, g, b];
end

function resolved_dir = resolve_dataset_directory(d)
    if ischar(d) || isstring(d)
        d_str = char(d);
    else
        resolved_dir = d;
        return;
    end

    if exist(d_str, 'dir')
        resolved_dir = d_str;
        return;
    end

    base_root = fileparts(fileparts(mfilename('fullpath')));
    estimate_dir = fileparts(mfilename('fullpath'));
    candidates = {
        fullfile(estimate_dir, d_str), ...
        fullfile(base_root, 'EstimateL', d_str), ...
        fullfile('EstimateL', d_str), ...
        fullfile(base_root, d_str)
    };
    for c = 1:numel(candidates)
        if exist(candidates{c}, 'dir')
            resolved_dir = candidates{c};
            return;
        end
    end
    resolved_dir = d_str;
end
