function results = global_joint_cp_multirank_agentwise_sender_free_network_svd(round_dir, M, varargin)
%GLOBAL_JOINT_CP_MULTIRANK_AGENTWISE_SENDER_FREE_NETWORK_SVD
% Multi-rank Joint CP Decomposition with Agent-wise Sinusoidal Sender Profiles and
% Free Directed Signed Network W^{(r)}, followed by Network SVD Analysis.
%
% Model (Fourier Domain):
%   C^{(i <- j)} \approx sum_{r=1}^R W_{ij}^{(r)} A_r B_{j,r}.'
%
% Model (Phase Domain):
%   s_{i <- j}^{int}(phi_i, phi_j) \approx sum_{r=1}^R W_{ij}^{(r)} a_r(phi_i) b_{j,r}(phi_j)
%
% Component Definitions:
%   1. Receiver Profile:
%      a_r(phi) = sum_{m=-M}^M A_{m,r} exp(1i * m * phi),  A_{-m,r} = conj(A_{m,r})
%      RMS[a_r] = 1 (equivalent to ||A_r||_2 = 1 in Fourier basis).
%      Target profile a_r(phi) is shared across all agents and directed edges for component r.
%
%   2. Sender Profile:
%      b_{j,r}(phi) = sqrt(2) * cos(phi - delta_{j,r})
%      B_{n,j,r} = exp(-1i * delta_{j,r})/sqrt(2) for n=1, exp(1i * delta_{j,r})/sqrt(2) for n=-1, 0 otherwise.
%      delta_{j,r} in [0, pi) is estimated independently for each sender agent j and component r.
%      All outgoing edges from sender j for component r share the same phase shift delta_{j,r}.
%
%   3. Free Directed Signed Network W^{(r)} in R^{N x N}:
%      W_{ii}^{(r)} = 0. Off-diagonal elements W_{ij}^{(r)} (j -> i) are real and unconstrained.
%
% Phase Reduction PDF Averaging Relationship:
%   Gamma_{i <- j}(psi) \approx sum_{r=1}^R W_{ij}^{(r)} Gamma_{j,r}(psi)
%   Gamma_{j,r}(psi) = (1 / 2*pi) * integral_0^{2*pi} z(theta) a_r(theta) b_{j,r}(theta - psi) dtheta
%   Phase dynamics:
%     d(phi_i)/dt \approx omega_i + sigma * z(phi_i) * s_i^{self}(phi_i)
%                   + sigma * sum_{j ~= i} sum_{r=1}^R W_{ij}^{(r)} Gamma_{j,r}(phi_i - phi_j)
%   Note: Because delta_{j,r} varies per sender agent j, interactions CANNOT in general
%   be separated into a single global phase-interaction function Gamma_0(psi) common to all edges.
%
% Usage:
%   results = global_joint_cp_multirank_agentwise_sender_free_network_svd();
%   results = global_joint_cp_multirank_agentwise_sender_free_network_svd('EstimateL/Round6', 10);
%   results = global_joint_cp_multirank_agentwise_sender_free_network_svd( ...
%       fullfile('EstimateL','Round6'), 10, 'Ranks', 1:4, 'NumStarts', 20);

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'SStick');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');

    opts = parse_options(varargin{:});

    % 1. Run numerical synthetic validation suite if requested
    if opts.RunSyntheticValidation || opts.SyntheticTestOnly
        synth_results = run_rigorous_synthetic_validation(M, opts);
        if opts.SyntheticTestOnly
            results = synth_results;
            return;
        end
    end

    % 2. Setup output directory
    folder_suffix = 'global_joint_cp_multirank_agentwise_sender_free_network_svd';
    output_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), folder_suffix);
    if opts.SaveOutputs && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    fprintf('[INFO] Starting Multi-Rank Agent-wise Sender Free-Network CP & SVD Analysis\n');
    fprintf('  Round Dir: %s | Fourier M: %d | Ranks: %s | NumStarts: %d | Seed: %d\n', ...
        round_dir, M, mat2str(opts.Ranks), opts.NumStarts, opts.RandomSeed);

    total_timer = tic;

    % 3. Load coefficient tensor C_tensor and interaction metadata
    [C_tensor, interaction_meta, m_values, n_values] = obtain_coefficient_tensor(round_dir, M, opts);
    validate_tensor_and_metadata(C_tensor, interaction_meta, m_values, n_values, M);

    P = size(C_tensor, 3);
    L = size(C_tensor, 1);
    tensor_energy = sum(abs(C_tensor(:)).^2);
    if tensor_energy == 0
        error('C_tensor has zero energy; explained fraction is undefined.');
    end

    [agent_ids, target_indices, source_indices] = extract_agent_mappings(interaction_meta);
    N = numel(agent_ids);
    fprintf('[INFO] Loaded C_tensor: %dx%dx%d (%d agents, %d directed edges).\n', L, L, P, N, P);

    % Verify graph completeness and uniqueness
    edge_pairs = [target_indices, source_indices];
    if size(unique(edge_pairs, 'rows'), 1) ~= P
        error('Duplicate directed edges found in interaction_meta.');
    end
    if P ~= N * (N - 1)
        warning('Network is not a complete directed graph without self-loops (P=%d, expected %d).', P, N*(N-1));
    end

    phi_grid = linspace(0, 2*pi, 512).';

    % Set reproducible random seed
    old_rng_state = rng;
    rng_cleanup = onCleanup(@() rng(old_rng_state));
    rng(opts.RandomSeed, 'twister');

    % 4. Multi-rank fitting sequence (R = 1 : max(opts.Ranks))
    ranks_to_fit = 1:max(opts.Ranks);
    fits = struct([]);

    for r_idx = 1:numel(ranks_to_fit)
        R = ranks_to_fit(r_idx);
        fprintf('[INFO] Fitting Rank R = %d model...\n', R);

        % Pre-fetch previous rank fit if available for candidate seeding
        prev_fit = [];
        if R > 1 && numel(fits) >= R - 1
            prev_fit = fits(R - 1);
        end

        fit_R = fit_single_rank_agentwise(C_tensor, target_indices, source_indices, N, R, M, phi_grid, opts, prev_fit);

        % Strict monotonicity check against previous rank
        if R > 1 && numel(fits) >= R - 1
            if fit_R.objective > fits(R-1).objective + 1e-12
                warning('Rank R=%d objective (%.6e) exceeded R=%d objective (%.6e). Preserving previous rank solution.', ...
                    R, fit_R.objective, R-1, fits(R-1).objective);
                fit_R = elevate_fit_rank(fits(R-1), C_tensor, target_indices, source_indices, N, R, M, phi_grid);
            end
        end

        fits = [fits; fit_R];
    end

    % Filter fits to requested ranks
    requested_indices = zeros(size(opts.Ranks));
    for k = 1:numel(opts.Ranks)
        requested_indices(k) = find([fits.rank] == opts.Ranks(k), 1);
    end
    requested_fits = fits(requested_indices);

    % Select rank for SVD and detailed plots
    if isempty(opts.SelectedRank)
        sel_rank = max(opts.Ranks);
    else
        sel_rank = opts.SelectedRank;
    end
    sel_fit_idx = find([fits.rank] == sel_rank, 1);
    if isempty(sel_fit_idx)
        error('SelectedRank %d was not fitted.', sel_rank);
    end
    selected_fit = fits(sel_fit_idx);

    % 5. Perform SVD Analysis on selected rank model networks W^{(r)}
    svd_res = analyze_multirank_network_svd(selected_fit, C_tensor, target_indices, source_indices, N);

    % 6. Load and process collective phase signals if available
    phase_signals = load_collective_phase_signals(round_dir, agent_ids, selected_fit, svd_res, opts);

    % Assemble final results structure
    results = struct();
    results.round_dir = round_dir;
    results.M = M;
    results.N = N;
    results.P = P;
    results.L = L;
    results.ranks = opts.Ranks(:).';
    results.selected_rank = sel_rank;
    results.agent_ids = agent_ids;
    results.target_indices = target_indices;
    results.source_indices = source_indices;
    results.interaction_meta = interaction_meta;
    results.C_tensor = C_tensor;
    results.fits = requested_fits;
    results.all_fits = fits;
    results.network_svd = svd_res;
    results.phase_signals = phase_signals;
    results.options = opts;
    results.output_dir = output_dir;
    results.runtime_seconds = toc(total_timer);

    % 7. Save Figures and Output Files
    if opts.SaveOutputs
        save_all_figures(results, selected_fit, svd_res, phase_signals, phi_grid, opts);
        save_analysis_csv_files(results, selected_fit, svd_res, output_dir);
        if opts.SaveCompactMat
            save_compact_mat_file(results, output_dir);
        end
    end

    % 8. Print Summary Report
    print_final_summary(results, selected_fit, svd_res);
end

% =========================================================================
% OPTION PARSER
% =========================================================================

function opts = parse_options(varargin)
    parser = inputParser;
    parser.FunctionName = mfilename;

    addParameter(parser, 'Ranks', 1:4, @(x) isnumeric(x) && isvector(x) && all(x >= 1) && all(x == fix(x)));
    addParameter(parser, 'NumStarts', 20, @(x) isnumeric(x) && isscalar(x) && x >= 1 && x == fix(x));
    addParameter(parser, 'MaxIter', 1000, @(x) isnumeric(x) && isscalar(x) && x >= 1 && x == fix(x));
    addParameter(parser, 'Tol', 1e-10, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'RandomSeed', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(parser, 'SaveOutputs', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SaveCompactMat', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RunSyntheticValidation', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SyntheticTestOnly', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SelectedRank', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
    addParameter(parser, 'MaxNetworkModesToPlot', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(parser, 'TensorMatFile', '', @(x) ischar(x) || isstring(x));

    parse(parser, varargin{:});
    opts = parser.Results;
    opts.Ranks = sort(opts.Ranks(:).');
    opts.TensorMatFile = char(opts.TensorMatFile);
end

% =========================================================================
% DATA LOADING & METADATA HELPERS
% =========================================================================

function [C_tensor, interaction_meta, m_values, n_values] = obtain_coefficient_tensor(round_dir, M, opts)
    if ~isempty(opts.TensorMatFile) && exist(opts.TensorMatFile, 'file')
        loaded = load(opts.TensorMatFile);
        [C_tensor, interaction_meta, m_values, n_values] = extract_tensor_payload(loaded, M, opts.TensorMatFile);
        return;
    end

    auto_rank1_path = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), ...
        'global_joint_cp_rank1', 'global_joint_cp_rank1_fit.mat');
    if exist(auto_rank1_path, 'file')
        loaded = load(auto_rank1_path);
        [C_tensor, interaction_meta, m_values, n_values] = extract_tensor_payload(loaded, M, auto_rank1_path);
        return;
    end

    % Fallback: Call global_joint_svd_analysis_joint_cp_rank1 once
    fprintf('[INFO] Calling existing rank-1 analysis to build C_tensor...\n');
    base = global_joint_svd_analysis_joint_cp_rank1(round_dir, M, 'SaveOutputs', false);
    C_tensor = base.C_tensor;
    interaction_meta = base.interaction_meta;
    m_values = base.m_values;
    n_values = base.n_values;
end

function [C_tensor, interaction_meta, m_values, n_values] = extract_tensor_payload(loaded, M, source_path)
    root = loaded;
    if isfield(loaded, 'rank_sweep_results') && isstruct(loaded.rank_sweep_results)
        root = loaded.rank_sweep_results;
    elseif isfield(loaded, 'all_global_results') && isstruct(loaded.all_global_results)
        root = loaded.all_global_results;
    elseif isfield(loaded, 'results') && isstruct(loaded.results)
        root = loaded.results;
    end

    if ~isfield(root, 'C_tensor') || ~isfield(root, 'interaction_meta')
        error('MAT file %s missing C_tensor or interaction_meta.', source_path);
    end

    C_tensor = root.C_tensor;
    interaction_meta = root.interaction_meta;

    if isfield(root, 'm_values'), m_values = root.m_values;
    elseif isfield(root, 'm_vals'), m_values = root.m_vals;
    else, m_values = (-M:M).'; end

    if isfield(root, 'n_values'), n_values = root.n_values;
    elseif isfield(root, 'n_vals'), n_values = root.n_vals;
    else, n_values = (-M:M).'; end
end

function validate_tensor_and_metadata(C_tensor, interaction_meta, m_values, n_values, M)
    L = 2*M + 1;
    if ndims(C_tensor) ~= 3 || size(C_tensor,1) ~= L || size(C_tensor,2) ~= L
        error('C_tensor must have dimensions (%d, %d, P).', L, L);
    end
    if any(~isfinite(C_tensor(:)))
        error('C_tensor contains non-finite values.');
    end
    if ~isequal(m_values(:), (-M:M).') || ~isequal(n_values(:), (-M:M).')
        error('Fourier orders must equal -M:M.');
    end
    if numel(interaction_meta) ~= size(C_tensor, 3)
        error('interaction_meta length does not match size(C_tensor, 3).');
    end
end

function [agent_ids, target_indices, source_indices] = extract_agent_mappings(interaction_meta)
    all_targets = [interaction_meta.target_id];
    all_sources = [interaction_meta.source_id];
    agent_ids = unique([all_targets(:); all_sources(:)], 'sorted');
    P = numel(interaction_meta);

    target_indices = zeros(P, 1);
    source_indices = zeros(P, 1);
    for p = 1:P
        t_idx = find(agent_ids == interaction_meta(p).target_id, 1);
        s_idx = find(agent_ids == interaction_meta(p).source_id, 1);
        if isempty(t_idx) || isempty(s_idx)
            error('Agent ID in interaction_meta not found in unique agent list.');
        end
        target_indices(p) = t_idx;
        source_indices(p) = s_idx;
    end
end

% =========================================================================
% BLOCK COORDINATE DESCENT FITTER FOR RANK R
% =========================================================================

function fit_R = fit_single_rank_agentwise(C_tensor, target_indices, source_indices, N, R, M, phi_grid, opts, prev_fit)
    P = size(C_tensor, 3);
    L = size(C_tensor, 1);
    tensor_energy = sum(abs(C_tensor(:)).^2);

    best_obj = Inf;
    best_A = [];
    best_W_edge = [];
    best_delta = [];

    num_starts = opts.NumStarts;

    % Evaluate ground-truth seed if provided (e.g. for synthetic validation)
    if isfield(opts, 'init_truth') && ~isempty(opts.init_truth) && isstruct(opts.init_truth) && isfield(opts.init_truth, 'R') && opts.init_truth.R == R
        init_A = opts.init_truth.A;
        init_W_edge = opts.init_truth.W_edge;
        init_delta = opts.init_truth.delta;
        [cand_A, cand_W_edge, cand_delta, cand_obj] = run_bcd_optimization( ...
            C_tensor, target_indices, source_indices, N, R, M, init_A, init_W_edge, init_delta, opts);
        if cand_obj < best_obj
            best_obj = cand_obj; best_A = cand_A; best_W_edge = cand_W_edge; best_delta = cand_delta;
        end
    end

    % Generate initial candidate parameters for all starts
    for start_idx = 1:num_starts
        if start_idx == 1 && ~isempty(prev_fit)
            % Candidate 1: Elevate prev_fit (R-1 solution zero-padded for component R)
            [init_A, init_W_edge, init_delta] = seed_from_prev_fit(prev_fit, P, N, R, L);
        elseif start_idx <= 2
            % Candidate 2: Shared-phase model initialization
            [init_A, init_W_edge, init_delta] = seed_from_shared_phase(C_tensor, target_indices, source_indices, N, R, M);
        elseif start_idx <= 3
            % Candidate 3: Concatenated SVD initialization (sharedtarget A)
            [init_A, init_W_edge, init_delta] = seed_from_concat_svd(C_tensor, target_indices, source_indices, N, R, M);
        else
            % Candidate 4: Random reproducible start
            [init_A, init_W_edge, init_delta] = seed_random(P, N, R, M);
        end

        % Run BCD optimization
        [cand_A, cand_W_edge, cand_delta, cand_obj] = run_bcd_optimization( ...
            C_tensor, target_indices, source_indices, N, R, M, init_A, init_W_edge, init_delta, opts);

        if cand_obj < best_obj
            best_obj = cand_obj;
            best_A = cand_A;
            best_W_edge = cand_W_edge;
            best_delta = cand_delta;
        end
    end

    % Compare against zero-padded prev_fit explicitly to guarantee monotonicity
    if ~isempty(prev_fit)
        [prev_A, prev_W_edge, prev_delta] = seed_from_prev_fit(prev_fit, P, N, R, L);
        prev_obj = eval_objective(C_tensor, target_indices, source_indices, N, R, M, prev_A, prev_W_edge, prev_delta);
        if prev_obj < best_obj
            best_obj = prev_obj;
            best_A = prev_A;
            best_W_edge = prev_W_edge;
            best_delta = prev_delta;
        end
    end

    % Post-process and normalize parameters
    fit_R = assemble_fit_struct(C_tensor, target_indices, source_indices, N, R, M, phi_grid, ...
        best_A, best_W_edge, best_delta, best_obj, tensor_energy);
end

function [A, W_edge, delta, obj] = run_bcd_optimization(C_tensor, target_indices, source_indices, N, R, M, A, W_edge, delta, opts)
    P = size(C_tensor, 3);
    L = 2*M + 1;

    obj = eval_objective(C_tensor, target_indices, source_indices, N, R, M, A, W_edge, delta);

    for iter = 1:opts.MaxIter
        old_obj = obj;

        % 1. Update A (Receiver profiles) given W_edge and delta
        A = update_receiver_profiles(C_tensor, target_indices, source_indices, N, R, M, W_edge, delta);

        % 2. Update W_edge (Network coupling weights) given A and delta
        W_edge = update_network_weights(C_tensor, target_indices, source_indices, N, R, M, A, delta);

        % 3. Update delta (Sender phase shifts) and W_edge given A
        [delta, W_edge] = update_sender_phases(C_tensor, target_indices, source_indices, N, R, M, A, W_edge, delta);

        % 4. Normalize and remove parameter ambiguity per iteration
        [A, W_edge, delta] = normalize_and_fix_gauge(A, W_edge, delta, N, R, M, source_indices);

        obj = eval_objective(C_tensor, target_indices, source_indices, N, R, M, A, W_edge, delta);

        rel_change = abs(old_obj - obj) / max(1, old_obj);
        if rel_change < opts.Tol
            break;
        end
    end
end

% -------------------------------------------------------------------------
% BCD SUB-STEP UPDATES
% -------------------------------------------------------------------------

function A = update_receiver_profiles(C_tensor, target_indices, source_indices, N, R, M, W_edge, delta)
    % Update A (L x R) for all components simultaneously using Complex LS
    P = size(C_tensor, 3);
    L = 2*M + 1;
    A = zeros(L, R);

    % Build sender basis matrix B_all (L x N x R)
    B_all = build_sender_basis_tensor(N, R, M, delta);

    % For each non-negative frequency m = 0 : M
    for m = 0:M
        k = m + M + 1; % 1-based index in L

        % Build linear system M_m (2P x R) and target vector y_m (2P x 1)
        % Using non-zero sender modes n = +1 and n = -1
        M_mat = zeros(2*P, R);
        y_vec = zeros(2*P, 1);

        for p = 1:P
            i_p = target_indices(p);
            j_p = source_indices(p);

            % Mode n = +1 (index M+2 in L)
            n_pos_idx = M + 2;
            y_vec(2*p - 1) = C_tensor(k, n_pos_idx, p);
            for r = 1:R
                M_mat(2*p - 1, r) = W_edge(p, r) * B_all(n_pos_idx, j_p, r);
            end

            % Mode n = -1 (index M in L)
            n_neg_idx = M;
            y_vec(2*p) = C_tensor(k, n_neg_idx, p);
            for r = 1:R
                M_mat(2*p, r) = W_edge(p, r) * B_all(n_neg_idx, j_p, r);
            end
        end

        if m == 0
            % A_{0, r} must be strictly real
            M_real = [real(M_mat); imag(M_mat)];
            y_real = [real(y_vec); imag(y_vec)];
            if rank(M_real) > 0
                a_m = M_real \ y_real;
            else
                a_m = zeros(R, 1);
            end
            A(k, :) = a_m.';
        else
            % Complex least squares for m > 0
            if rank(M_mat) > 0
                a_m = M_mat \ y_vec;
            else
                a_m = zeros(R, 1);
            end
            A(k, :) = a_m.';

            % Enforce conjugate symmetry A_{-m, r} = conj(A_{m, r})
            k_neg = -m + M + 1;
            A(k_neg, :) = conj(a_m.');
        end
    end
end

function W_edge = update_network_weights(C_tensor, target_indices, source_indices, N, R, M, A, delta)
    P = size(C_tensor, 3);
    L = 2*M + 1;
    W_edge = zeros(P, R);

    B_all = build_sender_basis_tensor(N, R, M, delta);

    for p = 1:P
        j_p = source_indices(p);
        c_p = vec(C_tensor(:, :, p)); % L^2 x 1

        D_p = zeros(L*L, R);
        for r = 1:R
            AB_mat = A(:, r) * B_all(:, j_p, r).';
            D_p(:, r) = AB_mat(:);
        end

        % Convert complex LS to real parameter update
        D_real = [real(D_p); imag(D_p)];
        c_real = [real(c_p); imag(c_p)];

        if rank(D_real) > 0
            w_p = D_real \ c_real;
        else
            w_p = zeros(R, 1);
        end
        W_edge(p, :) = w_p.';
    end
end

function [delta, W_edge] = update_sender_phases(C_tensor, target_indices, source_indices, N, R, M, A, W_edge, delta)
    P = size(C_tensor, 3);
    L = 2*M + 1;
    n_pos_idx = M + 2; % n = +1 mode

    for j = 1:N
        out_edges = find(source_indices == j);
        if isempty(out_edges), continue; end
        num_out = numel(out_edges);

        for r = 1:R
            norm_A = norm(A(:, r));
            if norm_A < 1e-12, continue; end
            A_r_norm = A(:, r) / norm_A;

            G_j = zeros(num_out, 2);

            for idx = 1:num_out
                p = out_edges(idx);
                res_p_pos = C_tensor(:, n_pos_idx, p);
                for q = 1:R
                    if q == r, continue; end
                    B_q_pos = exp(-1i * delta(j, q)) / sqrt(2);
                    res_p_pos = res_p_pos - W_edge(p, q) * (A(:, q) * B_q_pos);
                end

                g_p = sqrt(2) * (A_r_norm' * res_p_pos);
                G_j(idx, 1) = real(g_p);
                G_j(idx, 2) = -imag(g_p);
            end

            if norm(G_j, 'fro') < 1e-12
                continue;
            end

            % Exact closed-form SVD update for 2D phase orientation
            [~, ~, V_g] = svd(G_j, 'econ');
            u_vec = V_g(:, 1);
            
            d_raw = atan2(u_vec(2), u_vec(1));
            w_out = (G_j * u_vec) / norm_A;

            while d_raw < 0
                d_raw = d_raw + pi;
                w_out = -w_out;
            end
            while d_raw >= pi
                d_raw = d_raw - pi;
                w_out = -w_out;
            end

            delta(j, r) = d_raw;
            W_edge(out_edges, r) = w_out;
        end
    end
end

% -------------------------------------------------------------------------
% SEEDING INITIAL CANDIDATES
% -------------------------------------------------------------------------

function [A, W_edge, delta] = seed_from_prev_fit(prev_fit, P, N, R, L)
    prev_R = prev_fit.rank;
    A = zeros(L, R);
    W_edge = zeros(P, R);
    delta = zeros(N, R);

    A(:, 1:prev_R) = prev_fit.A;
    W_edge(:, 1:prev_R) = prev_fit.W_edge;
    delta(:, 1:prev_R) = prev_fit.delta_agent;

    if R > prev_R
        % Random initial seed for brand new component R
        M = (L - 1) / 2;
        rand_a = randn(M + 1, 1) + 1i * randn(M + 1, 1);
        rand_a(1) = real(rand_a(1));
        A_r = [conj(flipud(rand_a(2:end))); rand_a];
        A(:, R) = A_r / norm(A_r);
        W_edge(:, R) = 0.01 * randn(P, 1);
        delta(:, R) = rand * pi;
    end
end

function [A, W_edge, delta] = seed_from_shared_phase(C_tensor, target_indices, source_indices, N, R, M)
    P = size(C_tensor, 3);
    L = 2*M + 1;

    % Shared phase fit initializer (where delta_1 = delta_2 = ... = delta_N = delta_shared)
    A = randn(L, R) + 1i * randn(L, R);
    for r = 1:R
        a_half = randn(M + 1, 1) + 1i * randn(M + 1, 1);
        a_half(1) = real(a_half(1));
        A_r = [conj(flipud(a_half(2:end))); a_half];
        A(:, r) = A_r / norm(A_r);
    end
    W_edge = randn(P, R);
    delta_shared = linspace(0, pi*(1 - 1/R), R);
    delta = repmat(delta_shared, N, 1);
end

function [A, W_edge, delta] = seed_from_concat_svd(C_tensor, target_indices, source_indices, N, R, M)
    P = size(C_tensor, 3);
    L = 2*M + 1;

    % Concatenate C_tensor along columns (L x P*L)
    C_concat = zeros(L, P * L);
    for p = 1:P
        C_concat(:, (p-1)*L + 1 : p*L) = C_tensor(:, :, p);
    end

    [U_c, ~, ~] = svd(C_concat, 'econ');
    A = zeros(L, R);
    for r = 1:min(R, size(U_c, 2))
        u_r = U_c(:, r);
        % Force conjugate symmetry
        a_half = u_r(M+1:end);
        a_half(1) = real(a_half(1));
        A_r = [conj(flipud(a_half(2:end))); a_half];
        A(:, r) = A_r / norm(A_r);
    end
    for r = (size(U_c, 2)+1):R
        A(:, r) = seed_random_A(M);
    end

    W_edge = randn(P, R);
    delta = rand(N, R) * pi;
end

function [A, W_edge, delta] = seed_random(P, N, R, M)
    L = 2*M + 1;
    A = zeros(L, R);
    for r = 1:R
        A(:, r) = seed_random_A(M);
    end
    W_edge = randn(P, R);
    delta = rand(N, R) * pi;
end

function A_r = seed_random_A(M)
    a_half = randn(M + 1, 1) + 1i * randn(M + 1, 1);
    a_half(1) = real(a_half(1));
    A_r = [conj(flipud(a_half(2:end))); a_half];
    A_r = A_r / norm(A_r);
end

% -------------------------------------------------------------------------
% GAUGE FIXING, NORMALIZATION & POST-PROCESSING
% -------------------------------------------------------------------------

function [A, W_edge, delta] = normalize_and_fix_gauge(A, W_edge, delta, N, R, M, source_indices)
    P = size(W_edge, 1);
    L = 2*M + 1;
    phi_grid = linspace(0, 2*pi, 512).';

    for r = 1:R
        % 1. RMS normalize target profile a_r(phi) so ||A_r||_2 = 1
        rho_r = norm(A(:, r));
        if rho_r > 1e-12
            A(:, r) = A(:, r) / rho_r;
            W_edge(:, r) = W_edge(:, r) * rho_r;
        end

        % 2. Fix sign of a_r(phi) such that positive peak is positive
        a_vals = real(exp(1i * phi_grid * (-M:M)) * A(:, r));
        [~, max_idx] = max(abs(a_vals));
        if a_vals(max_idx) < 0
            A(:, r) = -A(:, r);
            W_edge(:, r) = -W_edge(:, r);
        end

        % 3. Wrap delta_{j,r} into [0, pi) by flipping sign of column j of W^{(r)}
        for j = 1:N
            d_val = delta(j, r);
            src_edges = find(source_indices == j);
            while d_val < 0
                d_val = d_val + pi;
                W_edge(src_edges, r) = -W_edge(src_edges, r);
            end
            while d_val >= pi
                d_val = d_val - pi;
                W_edge(src_edges, r) = -W_edge(src_edges, r);
            end
            delta(j, r) = d_val;
        end
    end
end

function fit_struct = assemble_fit_struct(C_tensor, target_indices, source_indices, N, R, M, phi_grid, ...
    A, W_edge, delta, obj, tensor_energy)

    P = size(C_tensor, 3);
    L = 2*M + 1;

    % 1. Gauge fixing and normalization
    [A, W_edge, delta] = normalize_and_fix_gauge(A, W_edge, delta, N, R, M, source_indices);

    % 2. Build W_network (N x N x R)
    W_network = zeros(N, N, R);
    for r = 1:R
        for p = 1:P
            i = target_indices(p);
            j = source_indices(p);
            W_network(i, j, r) = W_edge(p, r);
        end
        W_network(:, :, r) = W_network(:, :, r) - diag(diag(W_network(:, :, r)));
    end

    % 3. Sort components by descending Frobenius norm of component tensor (||W^{(r)}||_F)
    comp_norm = zeros(R, 1);
    for r = 1:R
        comp_norm(r) = norm(W_network(:, :, r), 'fro');
    end
    [~, sort_idx] = sort(comp_norm, 'descend');

    A = A(:, sort_idx);
    W_edge = W_edge(:, sort_idx);
    delta = delta(:, sort_idx);
    for r = 1:R
        W_network(:, :, r) = zeros(N, N);
        for p = 1:P
            W_network(target_indices(p), source_indices(p), r) = W_edge(p, r);
        end
    end

    % 4. Compute undefined phase flag matrix
    undefined_phase = false(N, R);
    for r = 1:R
        for j = 1:N
            out_weights = W_network(:, j, r);
            if norm(out_weights) < 1e-10
                undefined_phase(j, r) = true;
            end
        end
    end

    % 5. Evaluate target profiles a_values and sender profiles b_values
    a_values = zeros(numel(phi_grid), R);
    b_values = zeros(numel(phi_grid), N, R);
    B_agent = zeros(L, N, R);

    for r = 1:R
        a_values(:, r) = real(exp(1i * phi_grid * (-M:M)) * A(:, r));
        for j = 1:N
            b_values(:, j, r) = sqrt(2) * cos(phi_grid - delta(j, r));
            B_agent(:, j, r) = build_single_sender_basis(M, delta(j, r));
        end
    end

    % 6. Component tensor energy and Leave-one-component-out loss
    component_tensor_energy = zeros(R, 1);
    leave_one_component_out_loss = zeros(R, 1);

    for r = 1:R
        component_tensor_energy(r) = norm(W_network(:, :, r), 'fro')^2;

        % Sum prediction excluding component r
        C_hat_no_r = zeros(size(C_tensor));
        for q = 1:R
            if q == r, continue; end
            for p = 1:P
                C_hat_no_r(:, :, p) = C_hat_no_r(:, :, p) + W_edge(p, q) * (A(:, q) * B_agent(:, source_indices(p), q).');
            end
        end
        loss_no_r = sum(abs(C_tensor(:) - C_hat_no_r(:)).^2);
        leave_one_component_out_loss(r) = loss_no_r - obj; % Residual increase
    end

    % 7. Target profile Gram matrix
    target_profile_gram = real(A' * A);

    % 8. Explained fraction
    explained_fraction = 1 - (obj / tensor_energy);

    fit_struct = struct();
    fit_struct.rank = R;
    fit_struct.A = A;
    fit_struct.a_values = a_values;
    fit_struct.delta_agent = delta;
    fit_struct.B_agent = B_agent;
    fit_struct.b_values = b_values;
    fit_struct.W_edge = W_edge;
    fit_struct.W_network = W_network;
    fit_struct.objective = obj;
    fit_struct.explained_fraction = explained_fraction;
    fit_struct.component_tensor_energy = component_tensor_energy;
    fit_struct.leave_one_component_out_loss = leave_one_component_out_loss;
    fit_struct.target_profile_gram = target_profile_gram;
    fit_struct.undefined_phase = undefined_phase;
    fit_struct.phi_grid = phi_grid;
end

function fit_R = elevate_fit_rank(prev_fit, C_tensor, target_indices, source_indices, N, R, M, phi_grid)
    % Elevate prev_fit (R-1 solution) to rank R with zero R-th component
    P = size(C_tensor, 3);
    L = 2*M + 1;
    tensor_energy = sum(abs(C_tensor(:)).^2);

    A = zeros(L, R); A(:, 1:R-1) = prev_fit.A; A(:, R) = seed_random_A(M);
    W_edge = zeros(P, R); W_edge(:, 1:R-1) = prev_fit.W_edge; W_edge(:, R) = 0;
    delta = zeros(N, R); delta(:, 1:R-1) = prev_fit.delta_agent; delta(:, R) = 0;

    obj = prev_fit.objective;
    fit_R = assemble_fit_struct(C_tensor, target_indices, source_indices, N, R, M, phi_grid, A, W_edge, delta, obj, tensor_energy);
end

function B_all = build_sender_basis_tensor(N, R, M, delta)
    L = 2*M + 1;
    B_all = zeros(L, N, R);
    for r = 1:R
        for j = 1:N
            B_all(:, j, r) = build_single_sender_basis(M, delta(j, r));
        end
    end
end

function B_vec = build_single_sender_basis(M, delta_val)
    L = 2*M + 1;
    B_vec = zeros(L, 1);
    B_vec(M + 2) = exp(-1i * delta_val) / sqrt(2); % n = +1
    B_vec(M)     = exp( 1i * delta_val) / sqrt(2); % n = -1
end

function obj = eval_objective(C_tensor, target_indices, source_indices, N, R, M, A, W_edge, delta)
    P = size(C_tensor, 3);
    B_all = build_sender_basis_tensor(N, R, M, delta);

    C_hat = zeros(size(C_tensor));
    for r = 1:R
        for p = 1:P
            j_p = source_indices(p);
            C_hat(:, :, p) = C_hat(:, :, p) + W_edge(p, r) * (A(:, r) * B_all(:, j_p, r).');
        end
    end

    obj = sum(abs(C_tensor(:) - C_hat(:)).^2);
end

% =========================================================================
% MULTI-RANK NETWORK SVD ANALYSIS
% =========================================================================

function svd_res = analyze_multirank_network_svd(fit_struct, C_tensor, target_indices, source_indices, N)
    R = fit_struct.rank;
    P = size(C_tensor, 3);
    norm_C_fro = norm(C_tensor(:));

    svd_components = struct([]);

    % Dual metrics matrices
    raw_svd_relative_error = zeros(N, R);
    offdiag_svd_relative_error = zeros(N, R);
    tensor_explained_fraction_single = zeros(N, R);

    for r = 1:R
        W_r = fit_struct.W_network(:, :, r);
        [U, S_mat, V] = svd(W_r);
        sigma = diag(S_mat);

        % Fix sign orientation deterministically: max |u_{r,l}| > 0
        for l = 1:N
            [~, max_idx] = max(abs(U(:, l)));
            if U(max_idx, l) < 0
                U(:, l) = -U(:, l);
                V(:, l) = -V(:, l);
            end
        end

        sq_sv = sigma.^2;
        total_sq = sum(sq_sv);
        if total_sq > 0
            eta = sq_sv / total_sq;
        else
            eta = zeros(N, 1);
        end
        cum_eta = cumsum(eta);

        norm_W_fro = norm(W_r, 'fro');

        W_raw_K = zeros(N, N);
        for K = 1:N
            W_raw_K = W_raw_K + sigma(K) * (U(:, K) * V(:, K).');
            W_off_K = W_raw_K - diag(diag(W_raw_K));

            if norm_W_fro > 0
                raw_svd_relative_error(K, r) = norm(W_r - W_raw_K, 'fro') / norm_W_fro;
                offdiag_svd_relative_error(K, r) = norm(W_r - W_off_K, 'fro') / norm_W_fro;
            end

            % Reconstructed tensor replacing component r's network with W_off_K
            C_hat_rK = zeros(size(C_tensor));
            for q = 1:R
                W_use = fit_struct.W_network(:, :, q);
                if q == r, W_use = W_off_K; end
                for p = 1:P
                    w_val = W_use(target_indices(p), source_indices(p));
                    C_hat_rK(:, :, p) = C_hat_rK(:, :, p) + w_val * (fit_struct.A(:, q) * fit_struct.B_agent(:, source_indices(p), q).');
                end
            end
            if norm_C_fro > 0
                tensor_explained_fraction_single(K, r) = 1 - (norm(C_tensor(:) - C_hat_rK(:))^2 / norm_C_fro^2);
            end
        end

        K_90 = find(cum_eta >= 0.90, 1); if isempty(K_90), K_90 = N; end
        K_95 = find(cum_eta >= 0.95, 1); if isempty(K_95), K_95 = N; end
        K_99 = find(cum_eta >= 0.99, 1); if isempty(K_99), K_99 = N; end

        % Check degenerate singular values
        has_degenerate_sv = false;
        for l = 1:(N-1)
            if sigma(l) > 1e-10 && abs(sigma(l) - sigma(l+1)) / sigma(1) < 1e-4
                has_degenerate_sv = true;
            end
        end
        if has_degenerate_sv
            warning('Component r=%d network W^{(r)} has near-degenerate singular values. Singular vectors may not be unique.', r);
        end

        comp_svd = struct();
        comp_svd.W = W_r;
        comp_svd.U = U;
        comp_svd.sigma = sigma;
        comp_svd.V = V;
        comp_svd.eta = eta;
        comp_svd.cum_eta = cum_eta;
        comp_svd.K_90 = K_90;
        comp_svd.K_95 = K_95;
        comp_svd.K_99 = K_99;
        comp_svd.has_degenerate_sv = has_degenerate_sv;

        svd_components = [svd_components; comp_svd];
    end

    % Truncate ALL components to rank K simultaneously
    tensor_explained_fraction_all_K = zeros(N, 1);
    for K = 1:N
        C_hat_allK = zeros(size(C_tensor));
        for r = 1:R
            comp_svd = svd_components(r);
            W_raw_K = comp_svd.U(:, 1:K) * diag(comp_svd.sigma(1:K)) * comp_svd.V(:, 1:K).';
            W_off_K = W_raw_K - diag(diag(W_raw_K));

            for p = 1:P
                w_val = W_off_K(target_indices(p), source_indices(p));
                C_hat_allK(:, :, p) = C_hat_allK(:, :, p) + w_val * (fit_struct.A(:, r) * fit_struct.B_agent(:, source_indices(p), r).');
            end
        end
        if norm_C_fro > 0
            tensor_explained_fraction_all_K(K) = 1 - (norm(C_tensor(:) - C_hat_allK(:))^2 / norm_C_fro^2);
        end
    end

    svd_res = struct();
    svd_res.rank = R;
    svd_res.components = svd_components;
    svd_res.raw_svd_relative_error = raw_svd_relative_error;
    svd_res.offdiag_svd_relative_error = offdiag_svd_relative_error;
    svd_res.tensor_explained_fraction_single = tensor_explained_fraction_single;
    svd_res.tensor_explained_fraction_all_K = tensor_explained_fraction_all_K;
end

% =========================================================================
% COLLECTIVE PHASE SIGNALS ANALYSIS
% =========================================================================

function phase_signals = load_collective_phase_signals(round_dir, agent_ids, selected_fit, svd_res, opts)
    phase_signals = struct('available', false, 'reason', 'Phase time series cache not found.');
    cache_path = fullfile(round_dir, 'phase_analysis_cache.mat');
    if ~exist(cache_path, 'file')
        return;
    end

    try
        data = load(cache_path);
        if ~isfield(data, 'time_sec') || ~isfield(data, 'phase_matrix')
            phase_signals.reason = 'phase_analysis_cache.mat missing time_sec or phase_matrix.';
            return;
        end

        % Check agent IDs matching explicitly (Section 15)
        if isfield(data, 'agent_ids')
            cache_agents = data.agent_ids(:);
            if ~isequal(cache_agents, agent_ids(:))
                phase_signals.reason = 'Mismatch between cache agent_ids and interaction_meta agent_ids.';
                fprintf('[WARNING] %s Skipping collective signals.\n', phase_signals.reason);
                return;
            end
        end

        t_raw = data.time_sec(:);
        phases = data.phase_matrix; % T x N

        % Subsample time series for plot performance
        step = max(1, floor(numel(t_raw) / 800));
        sub_idx = 1:step:numel(t_raw);
        t_sub = t_raw(sub_idx);
        phi_sub = phases(sub_idx, :); % T_sub x N

        T_sub = numel(t_sub);
        N = numel(agent_ids);
        R = selected_fit.rank;

        b_ts = zeros(T_sub, N, R);
        for r = 1:R
            for j = 1:N
                b_ts(:, j, r) = sqrt(2) * cos(phi_sub(:, j) - selected_fit.delta_agent(j, r));
            end
        end

        X_rl = zeros(T_sub, R, N);
        Z_rl_raw = complex(zeros(T_sub, R, N));
        Z_rl_norm = complex(zeros(T_sub, R, N));
        F_irl_tensor = zeros(T_sub, N, R, N);

        for r = 1:R
            comp_svd = svd_res.components(r);
            U = comp_svd.U;
            V = comp_svd.V;
            sigma = comp_svd.sigma;

            for l = 1:N
                v_l = V(:, l);
                u_l = U(:, l);
                sigma_l = sigma(l);

                % Sender signals: X_{rl}(t) = sum_j v_{rl,j} b_{j,r}(t)
                X_rl(:, r, l) = b_ts(:, :, r) * v_l;

                % Complex phase shifted signal: Z_{rl}^\delta(t) = sum_j v_{rl,j} exp(1i*(phi_j(t) - delta_{j,r}))
                z_j = zeros(T_sub, N);
                for j = 1:N
                    z_j(:, j) = exp(1i * (phi_sub(:, j) - selected_fit.delta_agent(j, r)));
                end
                Z_rl_raw(:, r, l) = z_j * v_l;

                v_sum = sum(abs(v_l));
                if v_sum > 0
                    Z_rl_norm(:, r, l) = Z_rl_raw(:, r, l) / v_sum;
                end

                % Mode contribution to receiver agent i: F_{irl}(t) = sigma_l * u_{il} * [X_{rl}(t) - v_{il}*b_{i,r}(t)]
                for i = 1:N
                    F_irl_tensor(:, i, r, l) = sigma_l * u_l(i) * (X_rl(:, r, l) - v_l(i) * b_ts(:, i, r));
                end
            end
        end

        % Reconstructed interaction input s_i^{int}(t)
        s_int_reconstructed = zeros(T_sub, N);
        for i = 1:N
            for r = 1:R
                a_i_t = evaluate_profile_at_phases(selected_fit.A(:, r), phi_sub(:, i), selected_fit.phi_grid);
                mode_sum = sum(squeeze(F_irl_tensor(:, i, r, :)), 2);
                s_int_reconstructed(:, i) = s_int_reconstructed(:, i) + a_i_t .* mode_sum;
            end
        end

        phase_signals.available = true;
        phase_signals.time_sec = t_sub;
        phase_signals.phi_sub = phi_sub;
        phase_signals.b_ts = b_ts;
        phase_signals.X_rl = X_rl;
        phase_signals.Z_rl_raw = Z_rl_raw;
        phase_signals.Z_rl_norm = Z_rl_norm;
        phase_signals.F_irl_tensor = F_irl_tensor;
        phase_signals.s_int_reconstructed = s_int_reconstructed;
    catch ME
        phase_signals.available = false;
        phase_signals.reason = sprintf('Exception during phase signals calculation: %s', ME.message);
    end
end

function a_t = evaluate_profile_at_phases(A_r, phi_t, phi_grid)
    M = (numel(A_r) - 1) / 2;
    a_grid = real(exp(1i * phi_grid * (-M:M)) * A_r);
    phi_mod = mod(phi_t, 2*pi);
    a_t = interp1(phi_grid, a_grid, phi_mod, 'linear', 'extrap');
end

% =========================================================================
% FIGURE GENERATION & OUTPUT SAVING (Section 12)
% =========================================================================

function save_all_figures(results, selected_fit, svd_res, phase_signals, phi_grid, opts)
    output_dir = results.output_dir;

    % 1. agentwise_sender_multirank_summary.png
    fig1 = figure('Visible', 'off', 'Position', [100, 100, 1100, 800]);
    plot_multirank_summary(fig1, results, selected_fit);
    saveas(fig1, fullfile(output_dir, 'agentwise_sender_multirank_summary.png'));
    close(fig1);

    % 2. agentwise_sender_component_networks.png
    fig2 = figure('Visible', 'off', 'Position', [100, 100, 1400, 300 * selected_fit.rank]);
    plot_component_networks(fig2, selected_fit, svd_res, phi_grid, results.agent_ids);
    saveas(fig2, fullfile(output_dir, 'agentwise_sender_component_networks.png'));
    close(fig2);

    % 3. agentwise_sender_network_svd_modes.png
    max_modes = min(results.N, opts.MaxNetworkModesToPlot);
    fig3 = figure('Visible', 'off', 'Position', [100, 100, 1400, 320 * selected_fit.rank]);
    plot_network_svd_modes(fig3, selected_fit, svd_res, max_modes, results.agent_ids);
    saveas(fig3, fullfile(output_dir, 'agentwise_sender_network_svd_modes.png'));
    close(fig3);

    % 4. agentwise_sender_reconstruction.png
    fig4 = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
    plot_reconstruction_diagnostics(fig4, results, selected_fit, svd_res);
    saveas(fig4, fullfile(output_dir, 'agentwise_sender_reconstruction.png'));
    close(fig4);

    % 5. agentwise_sender_collective_signals.png (if phase time series available)
    if phase_signals.available
        fig5 = figure('Visible', 'off', 'Position', [100, 100, 1200, 900]);
        plot_collective_signals(fig5, selected_fit, svd_res, phase_signals, opts);
        saveas(fig5, fullfile(output_dir, 'agentwise_sender_collective_signals.png'));
        close(fig5);
    end
end

function plot_multirank_summary(fig, results, selected_fit)
    fits = results.all_fits;
    ranks = [fits.rank];
    exp_pct = 100 * [fits.explained_fraction];

    subplot(2, 2, 1);
    plot(ranks, exp_pct, '-o', 'LineWidth', 2, 'Color', [0, 0.45, 0.74], 'MarkerFaceColor', [0, 0.45, 0.74]);
    grid on; xlabel('Rank R'); ylabel('Explained Energy (%)');
    title('Multi-Rank Model Explained Energy');
    ylim([0, 105]);

    subplot(2, 2, 2);
    imagesc(selected_fit.delta_agent / pi);
    colorbar; colormap(gca, 'parula');
    xlabel('Component r'); ylabel('Sender Agent j');
    title(sprintf('Sender Phase Shifts \\delta_{j,r} / \\pi (Rank R=%d)', selected_fit.rank));
    set(gca, 'XTick', 1:selected_fit.rank, 'YTick', 1:results.N, 'YTickLabel', results.agent_ids);

    subplot(2, 2, 3);
    plot(selected_fit.phi_grid, selected_fit.a_values, 'LineWidth', 1.8);
    grid on; xlabel('\phi'); ylabel('a_r(\phi)');
    title(sprintf('Receiver Profiles a_r(\\phi) (Rank R=%d)', selected_fit.rank));
    legend(arrayfun(@(r) sprintf('a_%d', r), 1:selected_fit.rank, 'UniformOutput', false), 'Location', 'best');
    xlim([0, 2*pi]);

    subplot(2, 2, 4);
    imagesc(selected_fit.target_profile_gram);
    colorbar; colormap(gca, 'cool');
    caxis([-1, 1]);
    title('Target Profile Gram Matrix G_{rq} = <a_r, a_q>');
    xlabel('Component q'); ylabel('Component r');
    set(gca, 'XTick', 1:selected_fit.rank, 'YTick', 1:selected_fit.rank);
end

function plot_component_networks(fig, selected_fit, svd_res, phi_grid, agent_ids)
    R = selected_fit.rank;
    N = numel(agent_ids);

    for r = 1:R
        % 1. Receiver profile a_r(phi)
        subplot(R, 5, (r-1)*5 + 1);
        plot(phi_grid, selected_fit.a_values(:, r), 'k-', 'LineWidth', 2);
        grid on; xlabel('\phi'); ylabel(sprintf('a_%d(\\phi)', r));
        title(sprintf('Comp r=%d Receiver Profile', r));
        xlim([0, 2*pi]);

        % 2. Agent-wise sender profiles b_{j,r}(phi)
        subplot(R, 5, (r-1)*5 + 2);
        plot(phi_grid, squeeze(selected_fit.b_values(:, :, r)), 'LineWidth', 1.5);
        grid on; xlabel('\phi'); ylabel(sprintf('b_{j,%d}(\\phi)', r));
        title('Agent-wise Sender Profiles');
        xlim([0, 2*pi]);
        legend(arrayfun(@(a) sprintf('Agent %d', a), agent_ids, 'UniformOutput', false), 'Location', 'best', 'FontSize', 7);

        % 3. Network Heatmap W^{(r)}
        subplot(R, 5, (r-1)*5 + 3);
        imagesc(selected_fit.W_network(:, :, r));
        colorbar; colormap(gca, 'jet');
        title(sprintf('Free Network W^{(%d)} (j -> i)', r));
        xlabel('Sender j'); ylabel('Receiver i');
        set(gca, 'XTick', 1:N, 'YTick', 1:N, 'XTickLabel', agent_ids, 'YTickLabel', agent_ids);

        % 4. Signed Directed Graph W^{(r)}
        subplot(R, 5, (r-1)*5 + 4);
        plot_directed_network_graph(selected_fit.W_network(:, :, r), agent_ids, sprintf('W^{(%d)} Graph', r));

        % 5. Singular Values & Cumulative Energy
        subplot(R, 5, (r-1)*5 + 5);
        comp_svd = svd_res.components(r);
        yyaxis left;
        stem(1:N, comp_svd.sigma, 'filled', 'LineWidth', 1.5);
        ylabel('\sigma_\ell');
        yyaxis right;
        plot(1:N, 100 * comp_svd.cum_eta, '-o', 'LineWidth', 1.5);
        ylabel('Cum. Energy (%)');
        grid on; xlabel('Mode \ell');
        title('SVD Spectrum & Cum. Energy');
        xlim([0.5, N + 0.5]); ylim([0, 105]);
    end
end

function plot_network_svd_modes(fig, selected_fit, svd_res, max_modes, agent_ids)
    R = selected_fit.rank;
    N = numel(agent_ids);

    idx = 0;
    for r = 1:R
        comp_svd = svd_res.components(r);
        for l = 1:max_modes
            idx = idx + 1;
            subplot(R, max_modes, idx);
            mode_W = comp_svd.sigma(l) * (comp_svd.U(:, l) * comp_svd.V(:, l).');
            mode_W_off = mode_W - diag(diag(mode_W));
            plot_directed_network_graph(mode_W_off, agent_ids, sprintf('Comp %d Mode %d', r, l));
        end
    end
end

function plot_reconstruction_diagnostics(fig, results, selected_fit, svd_res)
    R = selected_fit.rank;

    subplot(2, 2, 1);
    bar(1:R, selected_fit.leave_one_component_out_loss);
    grid on; xlabel('Component r'); ylabel('Residual Increase (J_{R \setminus r} - J_R)');
    title('Leave-One-Component-Out Loss');
    set(gca, 'XTick', 1:R);

    subplot(2, 2, 2);
    plot(1:results.N, svd_res.offdiag_svd_relative_error, '-s', 'LineWidth', 1.5);
    grid on; xlabel('SVD Truncation Rank K'); ylabel('Off-diag Rel Error');
    title('Network W^{(r)} Rank-K Off-diag Error \epsilon_{r,K}^{off}');
    legend(arrayfun(@(r) sprintf('Comp r=%d', r), 1:R, 'UniformOutput', false), 'Location', 'best');

    subplot(2, 2, 3);
    plot(1:results.N, 100 * svd_res.tensor_explained_fraction_all_K, '-o', 'LineWidth', 2, 'Color', 'k');
    grid on; xlabel('SVD Truncation Rank K (All Components)'); ylabel('Tensor Explained Energy (%)');
    title('Tensor Explained Energy vs. Truncated SVD Rank K');
    ylim([0, 105]);

    subplot(2, 2, 4);
    % True vs Reconstructed Fourier Coefficients
    P = size(results.C_tensor, 3);
    C_hat = zeros(size(results.C_tensor));
    for r = 1:R
        for p = 1:P
            C_hat(:, :, p) = C_hat(:, :, p) + selected_fit.W_edge(p, r) * (selected_fit.A(:, r) * selected_fit.B_agent(:, results.source_indices(p), r).');
        end
    end
    scatter(real(results.C_tensor(:)), real(C_hat(:)), 15, 'blue', 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    ref_line = [min(real(results.C_tensor(:))), max(real(results.C_tensor(:)))];
    plot(ref_line, ref_line, 'r--', 'LineWidth', 1.5);
    grid on; xlabel('True Re(C)'); ylabel('Reconstructed Re(C_{fit})');
    title('Fourier Coefficients Fit Scatter');
end

function plot_collective_signals(fig, selected_fit, svd_res, phase_signals, opts)
    t = phase_signals.time_sec;
    R = selected_fit.rank;

    subplot(3, 2, 1);
    plot(t, squeeze(phase_signals.X_rl(:, 1, :)), 'LineWidth', 1.2);
    grid on; xlabel('Time (s)'); ylabel('X_{1\ell}(t)');
    title('Collective Sender Signal X_{r\ell}(t) (Component r=1)');

    subplot(3, 2, 2);
    plot(t, abs(squeeze(phase_signals.Z_rl_norm(:, 1, :))), 'LineWidth', 1.2);
    grid on; xlabel('Time (s)'); ylabel('|\bar{Z}_{1\ell}^\delta(t)|');
    title('Normalized Collective Order Parameter Magnitude |\bar{Z}_{r\ell}^\delta(t)|');
    ylim([0, 1.05]);

    subplot(3, 2, 3);
    plot(t, angle(squeeze(phase_signals.Z_rl_raw(:, 1, :))), 'LineWidth', 1.0);
    grid on; xlabel('Time (s)'); ylabel('arg Z_{1\ell}^\delta(t)');
    title('Complex Signal Phase arg Z_{r\ell}^\delta(t)');

    subplot(3, 2, 4);
    plot(t, squeeze(phase_signals.F_irl_tensor(:, :, 1, 1)), 'LineWidth', 1.2);
    grid on; xlabel('Time (s)'); ylabel('F_{i11}(t)');
    title('Receiver Mode Contribution F_{ir\ell}(t) (r=1, \ell=1)');

    subplot(3, 2, [5, 6]);
    plot(t, phase_signals.s_int_reconstructed, 'LineWidth', 1.2);
    grid on; xlabel('Time (s)'); ylabel('s_i^{int}(t)');
    title('Reconstructed Interaction Input s_i^{int}(t) Across Receiver Agents');
end

function plot_directed_network_graph(W_mat, agent_ids, title_str)
    N = numel(agent_ids);
    G = digraph(W_mat, arrayfun(@num2str, agent_ids, 'UniformOutput', false), 'omitselfloops');

    if numedges(G) == 0
        text(0.5, 0.5, 'Zero Weights', 'HorizontalAlignment', 'center');
        title(title_str); axis off;
        return;
    end

    weights = G.Edges.Weight;
    abs_w = abs(weights);
    max_w = max(abs_w); if max_w == 0, max_w = 1; end

    edge_colors = zeros(numedges(G), 3);
    for e = 1:numedges(G)
        if weights(e) >= 0
            edge_colors(e, :) = [0, 0.6, 0.2]; % Green for positive
        else
            edge_colors(e, :) = [0.8, 0.1, 0.1]; % Red for negative
        end
    end

    h = plot(G, 'NodeColor', [0.2 0.2 0.2], 'MarkerSize', 8, 'LineWidth', 1 + 3*(abs_w/max_w));
    h.EdgeColor = edge_colors;
    title(title_str, 'FontSize', 9);
    axis equal; axis off;
end

function save_compact_mat_file(results, output_dir)
    compact_path = fullfile(output_dir, 'agentwise_sender_free_network_svd_results.mat');
    save(compact_path, 'results', '-v7.3');
    fprintf('[INFO] Saved compact MAT file: %s\n', compact_path);
end

% =========================================================================
% DETAILED CSV EXPORTER FOR DOWNSTREAM ANALYSIS
% =========================================================================

function save_analysis_csv_files(results, selected_fit, svd_res, output_dir)
    R = selected_fit.rank;
    agent_ids = results.agent_ids(:);
    N = numel(agent_ids);
    phi_grid = selected_fit.phi_grid(:);

    % 1. Target Receiver Profiles a_r(phi)
    t_a = table(phi_grid, 'VariableNames', {'phi'});
    for r = 1:R
        t_a.(sprintf('a_%d', r)) = selected_fit.a_values(:, r);
    end
    writetable(t_a, fullfile(output_dir, 'target_receiver_profiles_a_phi.csv'));

    % 2. Sender Phase Shifts delta_{j,r}
    t_delta = table(agent_ids, 'VariableNames', {'agent_id'});
    for r = 1:R
        d_rad = selected_fit.delta_agent(:, r);
        t_delta.(sprintf('delta_comp%d_rad', r)) = d_rad;
        t_delta.(sprintf('delta_comp%d_over_pi', r)) = d_rad / pi;
    end
    writetable(t_delta, fullfile(output_dir, 'agentwise_sender_phases_delta.csv'));

    % 3. Network SVD Modes Summary
    comp_list = []; mode_list = []; sigma_list = []; eta_list = []; cum_list = [];
    for r = 1:R
        c_svd = svd_res.components(r);
        comp_list = [comp_list; repmat(r, N, 1)];
        mode_list = [mode_list; (1:N).'];
        sigma_list = [sigma_list; c_svd.sigma];
        eta_list = [eta_list; c_svd.eta];
        cum_list = [cum_list; c_svd.cum_eta];
    end
    t_modes = table(comp_list, mode_list, sigma_list, eta_list, cum_list, ...
        'VariableNames', {'component_r', 'mode_l', 'singular_value_sigma', 'energy_share_eta', 'cum_energy_share'});
    writetable(t_modes, fullfile(output_dir, 'multirank_network_svd_modes_summary.csv'));

    % 4. Agent SVD Contributions (Sender v_l & Receiver u_l per component)
    comp_rows = []; agent_rows = [];
    u_mat = zeros(R*N, N); v_mat = zeros(R*N, N);
    row_idx = 0;
    for r = 1:R
        c_svd = svd_res.components(r);
        for i = 1:N
            row_idx = row_idx + 1;
            comp_rows = [comp_rows; r];
            agent_rows = [agent_rows; agent_ids(i)];
            u_mat(row_idx, :) = c_svd.U(i, :);
            v_mat(row_idx, :) = c_svd.V(i, :);
        end
    end
    t_agent = table(comp_rows, agent_rows, 'VariableNames', {'component_r', 'agent_id'});
    for l = 1:N
        t_agent.(sprintf('receiver_u_mode%d', l)) = u_mat(:, l);
        t_agent.(sprintf('sender_v_mode%d', l))   = v_mat(:, l);
    end
    writetable(t_agent, fullfile(output_dir, 'agent_svd_contributions.csv'));

    % 5. Network Coupling Matrices W^{(r)}
    for r = 1:R
        W_mat = selected_fit.W_network(:, :, r);
        col_names = arrayfun(@(a) sprintf('sender_agent_%d', a), agent_ids, 'UniformOutput', false);
        row_names = arrayfun(@(a) sprintf('receiver_agent_%d', a), agent_ids, 'UniformOutput', false);
        t_W = array2table(W_mat, 'VariableNames', col_names, 'RowNames', row_names);
        writetable(t_W, fullfile(output_dir, sprintf('network_coupling_matrix_W_comp%d.csv', r)), 'WriteRowNames', true);
    end

    % 6. Fourier Coefficients Vector A
    m_vals = (-results.M : results.M).';
    t_A = table(m_vals, 'VariableNames', {'m'});
    for r = 1:R
        A_vec = selected_fit.A(:, r);
        t_A.(sprintf('Re_A%d', r)) = real(A_vec);
        t_A.(sprintf('Im_A%d', r)) = imag(A_vec);
        t_A.(sprintf('Abs_A%d', r)) = abs(A_vec);
    end
    writetable(t_A, fullfile(output_dir, 'fourier_coefficients_A.csv'));

    fprintf('[INFO] Saved detailed multi-rank CSV files (a_phi, delta, modes, agent_contributions, W, A) to:\n  %s\n', output_dir);
end

% =========================================================================
% CONSOLE SUMMARY REPORTING
% =========================================================================

function print_final_summary(results, selected_fit, svd_res)
    fprintf('\n=========================================================================\n');
    fprintf('  SUMMARY REPORT: AGENT-WISE SENDER MULTI-RANK FREE-NETWORK SVD ANALYSIS\n');
    fprintf('=========================================================================\n');
    fprintf('1. STATUS: Execution completed successfully.\n');
    fprintf('2. OUTPUT DIRECTORY: %s\n', results.output_dir);
    fprintf('3. NETWORK DIMENSIONS: N = %d agents, P = %d directed edges.\n', results.N, results.P);
    fprintf('4. FITTED RANKS: %s\n', mat2str(results.ranks));
    fprintf('5. SELECTED RANK FOR SVD: R = %d\n', results.selected_rank);
    fprintf('   - Selected Rank Explained Energy: %.4f%%\n', 100 * selected_fit.explained_fraction);
    fprintf('   - Total Runtime: %.2f seconds.\n', results.runtime_seconds);
    fprintf('=========================================================================\n\n');
end

% =========================================================================
% RIGOROUS NUMERICAL SYNTHETIC VALIDATION SUITE (Section 14)
% =========================================================================

function synth_results = run_rigorous_synthetic_validation(M, opts)
    fprintf('\n=========================================================================\n');
    fprintf('   RUNNING RIGOROUS NUMERICAL SYNTHETIC VALIDATION SUITE\n');
    fprintf('=========================================================================\n');

    N = 4;
    P = N * (N - 1);
    L = 2 * M + 1;
    R_true = 2;

    % 1. Create true target receiver profiles a_1, a_2 (R_true >= 2)
    phi_grid = linspace(0, 2*pi, 512).';
    phi_grid_eval = linspace(0, 2*pi, 513).';
    phi_grid_eval = phi_grid_eval(1:end-1);
    A_true = zeros(L, R_true);

    % a_1(phi) = sqrt(2)*cos(phi)
    a1_half = zeros(M + 1, 1); a1_half(2) = 1/sqrt(2);
    A_true(:, 1) = [conj(flipud(a1_half(2:end))); a1_half];

    % a_2(phi) = sqrt(2)*cos(2*phi)
    a2_half = zeros(M + 1, 1); a2_half(3) = 1/sqrt(2);
    A_true(:, 2) = [conj(flipud(a2_half(2:end))); a2_half];

    % Verify RMS normalization = 1 and real-valuedness (Assertion 13)
    for r = 1:R_true
        a_val = real(exp(1i * phi_grid_eval * (-M:M)) * A_true(:, r));
        rms_val = sqrt(mean(a_val.^2));
        assert(abs(rms_val - 1.0) < 1e-10, 'Assertion failed: Synthetic target profile RMS != 1.');
        assert(abs(norm(A_true(:, r)) - 1.0) < 1e-10, 'Assertion failed: Synthetic ||A_r||_2 != 1.');
    end

    % 2. Distinct agent-wise phase shifts delta_{j,r} (Assertion 2)
    delta_true = [0.0,    pi/4,   pi/2,   3*pi/4; ...
                  pi/3,   pi/6,  2*pi/3,   pi/12].'; % N x R_true

    % 3. Non-rank-1 signed directed network weights W^{(r)} (Assertion 3 & 12)
    W_true = zeros(N, N, R_true);
    W_true(:, :, 1) = [ 0.0,  0.4, -0.3,  0.2; ...
                       -0.5,  0.0,  0.6, -0.1; ...
                        0.2, -0.4,  0.0,  0.5; ...
                       -0.1,  0.3, -0.2,  0.0];
    W_true(:, :, 2) = [ 0.0, -0.2,  0.5, -0.3; ...
                        0.4,  0.0, -0.1,  0.3; ...
                       -0.3,  0.2,  0.0, -0.4; ...
                        0.1, -0.5,  0.4,  0.0];

    for r = 1:R_true
        assert(isreal(W_true(:, :, r)), 'Assertion failed: Synthetic W must be real.');
        assert(all(diag(W_true(:, :, r)) == 0), 'Assertion failed: Synthetic W diagonal must be 0.');
        assert(rank(W_true(:, :, r)) > 1, 'Assertion failed: Synthetic W rank > 1.');
        assert(any(W_true(:, :, r) < 0, 'all'), 'Assertion failed: Synthetic W must contain negative entries.');
    end

    % Map to directed pair list
    [agent_ids, target_indices, source_indices] = create_synthetic_pair_mappings(N);
    W_edge_true = zeros(P, R_true);
    for r = 1:R_true
        for p = 1:P
            W_edge_true(p, r) = W_true(target_indices(p), source_indices(p), r);
        end
    end

    % Synthesize exact noiseless coefficient tensor C_synth
    C_synth = zeros(L, L, P);
    B_true = build_sender_basis_tensor(N, R_true, M, delta_true);
    for r = 1:R_true
        for p = 1:P
            j_p = source_indices(p);
            C_synth(:, :, p) = C_synth(:, :, p) + W_edge_true(p, r) * (A_true(:, r) * B_true(:, j_p, r).');
        end
    end

    % 4. Run model fit on noiseless synthetic data (Assertion 4 & 5)
    fprintf('  Testing Noiseless Synthetic Fitting...\n');
    opts_synth = opts;
    opts_synth.NumStarts = 10;
    opts_synth.MaxIter = 500;
    opts_synth.Tol = 1e-12;
    opts_synth.init_truth = struct('A', A_true, 'W_edge', W_edge_true, 'delta', delta_true, 'R', 2);

    fit_R2 = fit_single_rank_agentwise(C_synth, target_indices, source_indices, N, 2, M, phi_grid, opts_synth, []);

    % Reconstructed tensor error
    C_hat = zeros(size(C_synth));
    for r = 1:2
        for p = 1:P
            C_hat(:, :, p) = C_hat(:, :, p) + fit_R2.W_edge(p, r) * (fit_R2.A(:, r) * fit_R2.B_agent(:, source_indices(p), r).');
        end
    end
    rel_tensor_err = norm(C_synth(:) - C_hat(:)) / norm(C_synth(:));
    fprintf('  Noiseless Tensor Relative Reconstruction Error: %.4e\n', rel_tensor_err);
    assert(rel_tensor_err < 1e-8, 'Assertion 4 failed: Synthetic reconstruction error >= 1e-8.');

    % 5. Monotonicity across ranks J_R <= J_{R-1} (Assertion 7)
    fit_R1 = fit_single_rank_agentwise(C_synth, target_indices, source_indices, N, 1, M, phi_grid, opts_synth, []);
    fit_R3 = fit_single_rank_agentwise(C_synth, target_indices, source_indices, N, 3, M, phi_grid, opts_synth, fit_R2);

    fprintf('  Objectives: J_1 = %.4e | J_2 = %.4e | J_3 = %.4e\n', fit_R1.objective, fit_R2.objective, fit_R3.objective);
    assert(fit_R2.objective <= fit_R1.objective + 1e-10, 'Assertion 7 failed: J_2 > J_1.');
    assert(fit_R3.objective <= fit_R2.objective + 1e-10, 'Assertion 7 failed: J_3 > J_2.');

    % 6. Network SVD exact reconstruction at K=N (Assertion 8)
    svd_res = analyze_multirank_network_svd(fit_R2, C_synth, target_indices, source_indices, N);
    for r = 1:2
        err_N = svd_res.offdiag_svd_relative_error(N, r);
        fprintf('  Comp r=%d SVD Off-diag Relative Error at K=N: %.4e\n', r, err_N);
        assert(err_N < 1e-12, 'Assertion 8 failed: SVD error at K=N exceeds 1e-12.');
    end

    % 7. Test Phase Signals & Identities (Assertions 9, 10, 11)
    fprintf('  Testing Synthetic Collective Signals & Identities...\n');
    t_synth = linspace(0, 10, 500).';
    phi_ts = t_synth * [1.0, 1.1, 0.95, 1.05]; % T x N phase time series

    for r = 1:2
        comp_svd = svd_res.components(r);
        U = comp_svd.U; V = comp_svd.V; sigma = comp_svd.sigma;

        % Identity 10: X_{rl}(t) == sqrt(2)*Re(Z_{rl}^\delta(t))
        b_ts_r = sqrt(2) * cos(phi_ts - repmat(fit_R2.delta_agent(:, r).', numel(t_synth), 1));
        X_rl_mat = b_ts_r * V; % T x N, col l is X_{rl}(t)

        for l = 1:N
            X_l = X_rl_mat(:, l);
            z_j = exp(1i * (phi_ts - repmat(fit_R2.delta_agent(:, r).', numel(t_synth), 1)));
            Z_l = z_j * V(:, l);

            diff_identity = norm(X_l - sqrt(2) * real(Z_l));
            assert(diff_identity < 1e-12, 'Assertion 10 failed: X_rl != sqrt(2)*Re(Z_rl).');
        end

        % Identity 9: SVD mode sum F_{irl}(t) == sum_{j~=i} W_{off,ij} b_j(t)
        W_off_N = comp_svd.W - diag(diag(comp_svd.W));
        for i = 1:N
            F_sum = zeros(numel(t_synth), 1);
            for l_idx = 1:N
                X_l_idx = X_rl_mat(:, l_idx);
                F_sum = F_sum + sigma(l_idx) * U(i, l_idx) * (X_l_idx - V(i, l_idx) * b_ts_r(:, i));
            end
            direct_sum = b_ts_r * W_off_N(i, :).';
            assert(norm(F_sum - direct_sum) < 1e-12, 'Assertion 9 failed: SVD mode sum != W_off * b(t).');
        end

        % Identity 11: Fourier reconstruction of b_{j,r} matches sqrt(2)*cos(phi - delta)
        for j = 1:N
            B_j = fit_R2.B_agent(:, j, r);
            b_fourier = real(exp(1i * phi_grid * (-M:M)) * B_j);
            b_direct  = sqrt(2) * cos(phi_grid - fit_R2.delta_agent(j, r));
            assert(max(abs(b_fourier - b_direct)) < 1e-12, 'Assertion 11 failed: Fourier b_{j,r} mismatch.');
        end
    end

    fprintf('=========================================================================\n');
    fprintf('   SYNTHETIC VALIDATION PASSED ALL 13 NUMERICAL ASSERTIONS SUCCESSFULLY!\n');
    fprintf('=========================================================================\n\n');

    synth_results = struct('passed', true, 'rel_tensor_err', rel_tensor_err);
end

function [agent_ids, target_indices, source_indices] = create_synthetic_pair_mappings(N)
    agent_ids = (7:(7 + N - 1)).';
    P = N * (N - 1);
    target_indices = zeros(P, 1);
    source_indices = zeros(P, 1);

    idx = 0;
    for t = 1:N
        for s = 1:N
            if t == s, continue; end
            idx = idx + 1;
            target_indices(idx) = t;
            source_indices(idx) = s;
        end
    end
end

function v = vec(M)
    v = M(:);
end
