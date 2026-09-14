function [results, res_max, res_min] = simulate_u_alpha_optimal_phase_dynamics(source, varargin)
% SIMULATE_U_ALPHA_OPTIMAL_PHASE_DYNAMICS
% Calculates, optimizes, and simulates the network phase dynamics under the
% phase-shifted potential:
%
%   U_j(theta) = Re[ exp(1i * alpha) * conj(x_j) * (W * x)_j ]
%              = sum_k W_{jk} * cos(theta_k - theta_j + alpha)
%
% where:
%   x_j = exp(1i * theta_j) is the complex phasor of agent j,
%   alpha in R is the sender phase shift (loaded automatically from
%         sender_phase_shift_delta.csv or specified by the user),
%   W in R^{N x N} is the coupling weight matrix.
%
% Furthermore, using the Singular Value Decomposition (SVD) of W:
%   W = sum_ell s_ell * u_ell * v_ell^T
% this function performs exact modal decomposition of the potential:
%
%   U_j(theta) = sum_ell s_ell * u_{j, ell} * Re[ exp(1i * alpha) * conj(x_j) * Z_ell ]
%
% where Z_ell = sum_k v_{k, ell} * x_k is the weighted order parameter
% of sender mode ell.
%
% Features:
%   1. Multi-start global optimization of the total network potential:
%        U_total(theta) = sum_j U_j(theta)
%      and identification of all stationary equilibria / extrema (grad U = 0).
%   2. Gradient flow relaxation simulation:
%        d(theta_j)/dt = +/- grad_gain * (dU_j / dtheta_j)
%                      = +/- grad_gain * sum_k W_{jk} * sin(theta_k - theta_j + alpha)
%   3. Exact SVD mode contribution analysis:
%        U_j = sum_ell U_j^{(ell)}
%      at optimal states and along relaxation trajectories.
%   4. High-resolution LaTeX-rendered visualizations:
%      - Gradient flow relaxation trajectories & order parameters |Z_ell(t)|
%      - Optimal phasors on the complex unit circle with mode order parameter vectors
%      - SVD mode contribution stacked bar charts & heatmaps
%      - Pairwise interaction energy matrices E_{jk} = W_{jk} * cos(theta_k - theta_j + alpha)
%
% Usage:
%   res = simulate_u_alpha_optimal_phase_dynamics();
%   res = simulate_u_alpha_optimal_phase_dynamics('SStickFlat');
%   res = simulate_u_alpha_optimal_phase_dynamics('SStickFlat', 'AlphaDeg', 26.74);
%   res = simulate_u_alpha_optimal_phase_dynamics([], 'SaveOutputs', true);

    % =========================================================================
    % USER CONFIGURATION PRESETS (For F5 / Run without arguments)
    % =========================================================================
    DEFAULT_DATASET_OR_W = fullfile('EstimateL', 'SStickFlat', 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');

    % Phase shift alpha [rad] - EASILY CONFIGURE HERE:
    % Default: -0.2 * pi (~ -0.6283 rad = -36.0 deg)
    DEFAULT_ALPHA_RAD     = -0.2 * pi;

    % Optimization settings
    DEFAULT_NUM_STARTS    = 100;   % Multi-start initial points
    DEFAULT_RANDOM_SEED   = [];    % Random seed (empty [] for fresh random state on each run, integer for fixed seed)

    % Dynamic simulation settings
    DEFAULT_INITIAL_THETA = [];      % Initial phase vector [rad] (empty [] uses random initial state)
    DEFAULT_FLOW_TYPE     = 'local'; % 'local' (oscillator dynamics) or 'total' (energy ascent)
    DEFAULT_SIM_TIME_SEC  = 30.0;    % Duration (s)
    DEFAULT_SIM_DT        = 0.01;    % Time step dt (s)
    DEFAULT_GRAD_GAIN     = 10.0;    % Rate / gain factor

    % Reference agent for relative phase calculation (empty [] uses 1st agent)
    DEFAULT_REF_AGENT_ID  = [];

    % Visualization & Export settings
    DEFAULT_PLOT_TRAJECTORIES  = true;
    DEFAULT_PLOT_PHASORS       = true;
    DEFAULT_PLOT_MODES         = true;
    DEFAULT_PLOT_ENERGY_MATRIX = true;
    DEFAULT_AGENT_OFFSET       = 0;     % Display ID offset
    DEFAULT_SAVE_OUTPUTS       = false; % Save CSV, MAT, and PNG figures
    DEFAULT_OUTPUT_DIR         = '';
    % =========================================================================

    % Flexible argument handling:
    % Allow calls like:
    %   simulate_u_alpha_optimal_phase_dynamics()                     -> uses DEFAULT_ALPHA_RAD (-0.2*pi)
    %   simulate_u_alpha_optimal_phase_dynamics(-0.2*pi)              -> alpha = -0.2*pi
    %   simulate_u_alpha_optimal_phase_dynamics('SStickFlat', -0.2*pi)-> alpha = -0.2*pi
    %   simulate_u_alpha_optimal_phase_dynamics(..., 'AlphaDeg', -36)
    raw_args = varargin;
    if nargin < 1 || isempty(source)
        source = DEFAULT_DATASET_OR_W;
    elseif isnumeric(source) && isscalar(source)
        % First argument is numeric: user passed alpha directly!
        alpha_val = source;
        source = DEFAULT_DATASET_OR_W;
        raw_args = [{'AlphaRad', alpha_val}, raw_args];
    elseif ischar(source) || isstring(source)
        source_str = char(source);
        if ~isfile(source_str) && ~isfolder(source_str) && ~endsWith(lower(source_str), '.csv')
            s_cand1 = fullfile('EstimateL', source_str, 'low_rank_analysis', 'M10', ...
                'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
            s_cand2 = fullfile(source_str, 'low_rank_analysis', 'M10', ...
                'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
            if isfile(s_cand1) || isfile(s_cand2)
                % Valid dataset name
            else
                raw_args = [{source}, raw_args];
                source = DEFAULT_DATASET_OR_W;
            end
        end
    end

    % Check if second argument is numeric alpha: e.g. ('SStickFlat', -0.2*pi)
    if ~isempty(raw_args) && isnumeric(raw_args{1}) && isscalar(raw_args{1})
        raw_args = [{'AlphaRad', raw_args{1}}, raw_args(2:end)];
    end

    opts = parse_u_alpha_options(DEFAULT_ALPHA_RAD, DEFAULT_NUM_STARTS, DEFAULT_RANDOM_SEED, ...
        DEFAULT_INITIAL_THETA, DEFAULT_FLOW_TYPE, DEFAULT_SIM_TIME_SEC, DEFAULT_SIM_DT, DEFAULT_GRAD_GAIN, ...
        DEFAULT_REF_AGENT_ID, DEFAULT_PLOT_TRAJECTORIES, DEFAULT_PLOT_PHASORS, ...
        DEFAULT_PLOT_MODES, DEFAULT_PLOT_ENERGY_MATRIX, DEFAULT_AGENT_OFFSET, ...
        DEFAULT_SAVE_OUTPUTS, DEFAULT_OUTPUT_DIR, raw_args{:});

    if ~isempty(opts.RandomSeed)
        rng(opts.RandomSeed, 'twister');
    else
        rng('shuffle');
    end

    % 1. Load coupling matrix W, agent IDs, SVD modes, and alpha
    [W, agent_ids, w_source_path, svd_data, alpha_auto] = load_network_and_svd_data(source);
    N = numel(agent_ids);
    if N < 2
        error('Weight matrix W must contain at least 2 agents (got %d).', N);
    end

    % Determine alpha (user override > preset > file auto-load)
    if ~isempty(opts.AlphaRad)
        alpha = opts.AlphaRad;
    elseif ~isempty(opts.AlphaDeg)
        alpha = deg2rad(opts.AlphaDeg);
    elseif ~isempty(DEFAULT_ALPHA_RAD)
        alpha = DEFAULT_ALPHA_RAD;
    elseif ~isempty(alpha_auto)
        alpha = alpha_auto;
    else
        alpha = -0.2 * pi;
    end

    % Determine reference agent
    if isempty(opts.ReferenceAgentId)
        ref_agent_id = agent_ids(1);
        ref_idx = 1;
    else
        ref_idx = find(agent_ids == opts.ReferenceAgentId, 1);
        if isempty(ref_idx)
            warning('[WARN] ReferenceAgentId %d not in agent_ids. Falling back to agent %d.', ...
                opts.ReferenceAgentId, agent_ids(1));
            ref_agent_id = agent_ids(1);
            ref_idx = 1;
        else
            ref_agent_id = opts.ReferenceAgentId;
        end
    end

    % Print banner
    fprintf('=========================================================================\n');
    fprintf('   PHASE-SHIFTED POTENTIAL U_j(theta, alpha) OPTIMIZATION & MODAL DYNAMICS\n');
    fprintf('=========================================================================\n');
    fprintf('  Source W: %s\n', w_source_path);
    fprintf('  Agents (N=%d): %s | Reference Agent: ID %d\n', N, mat2str(agent_ids), ref_agent_id);
    fprintf('  Phase shift alpha: %+.4f rad (%+.2f deg)\n', alpha, rad2deg(alpha));
    fprintf('  SVD Modes (Rank=%d): %d modes loaded / computed\n', numel(svd_data.S), numel(svd_data.S));
    fprintf('  Flow type: %s | Starts: %d | Sim duration: %.1fs | dt: %.3fs\n', ...
        opts.FlowType, opts.NumStarts, opts.SimTimeSec, opts.SimDt);

    % 2. Multi-start Global Optimization for Maximum and Minimum of U_total
    fprintf('[INFO] Searching for Global Maximum and Minimum of U_total(theta)...\n');
    opt_results = run_u_alpha_global_optimization(W, alpha, svd_data, opts.NumStarts, ref_idx);

    % 3. Gradient Flow Dynamic Simulation from Initial State
    if ~isempty(opts.InitialTheta)
        theta_init_sim = opts.InitialTheta(:);
        if numel(theta_init_sim) ~= N
            error('InitialTheta length (%d) must match number of agents N=%d.', numel(theta_init_sim), N);
        end
        fprintf('[INFO] Using user-specified initial phase vector for gradient flow:\n');
        for k = 1:N
            fprintf('         Agent %2d: %+7.2f deg (%+7.4f rad)\n', ...
                agent_ids(k) + opts.AgentOffset, rad2deg(theta_init_sim(k)), theta_init_sim(k));
        end
    else
        % Generate fresh, independent uniform random initial phase in [-pi, pi]
        theta_init_sim = 2 * pi * rand(N, 1) - pi;
        fprintf('[INFO] Generated fresh random initial phase vector for gradient flow:\n');
        for k = 1:N
            fprintf('         Agent %2d: %+7.2f deg (%+7.4f rad)\n', ...
                agent_ids(k) + opts.AgentOffset, rad2deg(theta_init_sim(k)), theta_init_sim(k));
        end
    end
    fprintf('[INFO] Simulating %s gradient flow dynamics (maximization & minimization)...\n', opts.FlowType);
    sim_results = run_u_alpha_gradient_flow_simulations(W, alpha, svd_data, theta_init_sim, ...
        opts.SimTimeSec, opts.SimDt, opts.GradGain, opts.FlowType, ref_idx);

    % 4. Assemble complete results struct
    results = struct();
    results.source_path = w_source_path;
    results.agent_ids = agent_ids;
    results.N = N;
    results.alpha = alpha;
    results.alpha_deg = rad2deg(alpha);
    results.reference_agent_id = ref_agent_id;
    results.reference_idx = ref_idx;
    results.W = W;
    results.svd = svd_data;
    results.maximum = opt_results.maximum;
    results.minimum = opt_results.minimum;
    results.extrema = opt_results.extrema;
    results.all_starts = opt_results.all_starts;
    results.simulation = sim_results;
    results.options = opts;

    % Secondary outputs
    res_max = results.maximum;
    res_min = results.minimum;

    % 5. Print comprehensive numerical summary
    print_u_alpha_results_summary(results, opts.AgentOffset);

    % 6. Plotting
    figures = struct();
    if opts.PlotTrajectories
        figures.trajectories = plot_u_alpha_trajectories(sim_results, agent_ids, ...
            ref_agent_id, opts.AgentOffset, alpha, opt_results.maximum.U_mean, opt_results.minimum.U_mean);
    else
        figures.trajectories = [];
    end

    if opts.PlotPhasors
        figures.phasors = plot_u_alpha_phasor_diagrams(results, opts.AgentOffset);
    else
        figures.phasors = [];
    end

    if opts.PlotModes
        figures.modes = plot_u_alpha_mode_contributions(results, opts.AgentOffset);
    else
        figures.modes = [];
    end

    if opts.PlotEnergyMatrix
        figures.energy_matrix = plot_u_alpha_pairwise_energy(results, opts.AgentOffset);
    else
        figures.energy_matrix = [];
    end
    results.figures = figures;

    % 7. Save outputs if requested
    if opts.SaveOutputs
        results.export = save_u_alpha_analysis_outputs(results, opts);
    end

    fprintf('[INFO] Analysis completed successfully.\n');
end

% =========================================================================
% MATHEMATICAL FORMULATION: POTENTIAL, GRADIENTS, SVD MODES
% =========================================================================

function [U_j, U_total, U_mean, grad_local, grad_total, U_j_modes, Z_modes] = ...
    evaluate_u_alpha(theta, W, alpha, S_vals, U_mat, V_mat)
% Evaluates:
%   U_j = Re[ exp(i*alpha) * conj(x_j) * (Wx)_j ] = sum_k W_{jk} cos(theta_k - theta_j + alpha)
%   grad_local_j = dU_j / dtheta_j = sum_k W_{jk} sin(theta_k - theta_j + alpha)
%   grad_total_j = dU_total / dtheta_j
%   U_j^{(ell)} = s_ell * u_{j, ell} * Re[ exp(i*alpha) * conj(x_j) * Z_ell ]
%   Z_ell = sum_k v_{k, ell} * x_k

    theta = theta(:);
    N = numel(theta);
    x = exp(1i * theta);

    Wx = W * x;
    rot_x_conj_Wx = exp(1i * alpha) * (conj(x) .* Wx);

    % Local potential U_j and total
    U_j = real(rot_x_conj_Wx);
    U_total = sum(U_j);
    U_mean = U_total / N;

    % Local gradient dU_j / dtheta_j
    grad_local = imag(rot_x_conj_Wx);

    % Total gradient dU_total / dtheta_j
    W_sym = 0.5 * (W + W.');
    W_skew = 0.5 * (W - W.');
    delta_mat = theta.' - theta; % delta(j, k) = theta_k - theta_j
    sin_mat = sin(delta_mat);
    cos_mat = cos(delta_mat);

    grad_total = 2 * sum(W_sym .* sin_mat, 2) * cos(alpha) + ...
                 2 * sum(W_skew .* cos_mat, 2) * sin(alpha);

    % Exact SVD Mode Decomposition
    R_modes = numel(S_vals);
    U_j_modes = zeros(N, R_modes);
    Z_modes = zeros(R_modes, 1);

    for ell = 1:R_modes
        Z_ell = sum(V_mat(:, ell) .* x);
        Z_modes(ell) = Z_ell;
        zeta_ell = S_vals(ell) * U_mat(:, ell) .* (exp(1i * alpha) * conj(x) * Z_ell);
        U_j_modes(:, ell) = real(zeta_ell);
    end
end

function [f, g] = u_alpha_objective_for_optim(theta, W, alpha, S_vals, U_mat, V_mat, sign_factor)
    [~, U_total, ~, ~, grad_total] = evaluate_u_alpha(theta, W, alpha, S_vals, U_mat, V_mat);
    f = sign_factor * U_total;
    g = sign_factor * grad_total;
end

% =========================================================================
% GLOBAL OPTIMIZATION & EXTREMA CLASSIFICATION
% =========================================================================

function opt_res = run_u_alpha_global_optimization(W, alpha, svd_data, num_starts, ref_idx)
    N = size(W, 1);
    S_vals = svd_data.S;
    U_mat = svd_data.U;
    V_mat = svd_data.V;

    use_optim_toolbox = (exist('fminunc', 'file') == 2);

    best_max_U = -Inf;
    best_max_theta = [];
    best_min_U = Inf;
    best_min_theta = [];

    start_U_max = zeros(num_starts, 1);
    start_U_min = zeros(num_starts, 1);

    % Diverse initial candidates
    initial_candidates = cell(num_starts, 1);
    % Spectral / SVD candidates
    initial_candidates{1} = angle(U_mat(:, 1));
    initial_candidates{2} = angle(V_mat(:, 1));
    initial_candidates{3} = zeros(N, 1);                    % In-phase
    initial_candidates{4} = pi * (mod(1:N, 2).');           % Alternating
    initial_candidates{5} = (2 * pi / N) * (0:N-1).';       % Splay / traveling wave
    initial_candidates{6} = (pi / 2) * (mod(0:N-1, 4).');   % Quadrature

    for k = 7:num_starts
        initial_candidates{k} = 2 * pi * rand(N, 1) - pi;
    end

    theta_init_sim = initial_candidates{min(7, num_starts)};

    candidates_extrema = cell(2 * num_starts, 1);
    cand_count = 0;

    for s = 1:num_starts
        th0 = initial_candidates{s};

        if use_optim_toolbox
            opts_opt = optimoptions('fminunc', 'Display', 'off', ...
                'Algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, ...
                'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12, 'MaxIterations', 500);

            % Maximization
            obj_max = @(th) u_alpha_objective_for_optim(th, W, alpha, S_vals, U_mat, V_mat, -1);
            [th_max_opt, fval_max] = fminunc(obj_max, th0, opts_opt);
            u_max_cand = -fval_max;

            % Minimization
            obj_min = @(th) u_alpha_objective_for_optim(th, W, alpha, S_vals, U_mat, V_mat, +1);
            [th_min_opt, fval_min] = fminunc(obj_min, th0, opts_opt);
            u_min_cand = fval_min;
        else
            [th_max_opt, u_max_cand] = optimize_u_alpha_local(th0, W, alpha, S_vals, U_mat, V_mat, +1);
            [th_min_opt, u_min_cand] = optimize_u_alpha_local(th0, W, alpha, S_vals, U_mat, V_mat, -1);
        end

        start_U_max(s) = u_max_cand;
        start_U_min(s) = u_min_cand;

        cand_count = cand_count + 1;
        candidates_extrema{cand_count} = th_max_opt;
        cand_count = cand_count + 1;
        candidates_extrema{cand_count} = th_min_opt;

        if u_max_cand > best_max_U
            best_max_U = u_max_cand;
            best_max_theta = th_max_opt;
        end

        if u_min_cand < best_min_U
            best_min_U = u_min_cand;
            best_min_theta = th_min_opt;
        end
    end

    % Cluster and classify all unique stationary extrema & equilibria
    extrema = cluster_and_classify_u_alpha_extrema(candidates_extrema(1:cand_count), ...
        W, alpha, svd_data, ref_idx);

    max_theta_norm = normalize_phase_vector(best_max_theta, ref_idx);
    min_theta_norm = normalize_phase_vector(best_min_theta, ref_idx);

    opt_res = struct();
    opt_res.theta_init_sim = theta_init_sim;
    opt_res.extrema = extrema;
    opt_res.maximum = assemble_u_alpha_state(max_theta_norm, W, alpha, svd_data, 'Global Maximum (Max Potential)', ref_idx);
    opt_res.minimum = assemble_u_alpha_state(min_theta_norm, W, alpha, svd_data, 'Global Minimum (Min Potential)', ref_idx);
    opt_res.all_starts = struct('max_values', start_U_max, 'min_values', start_U_min, 'num_starts', num_starts);
end

function [th_opt, u_opt] = optimize_u_alpha_local(th0, W, alpha, S_vals, U_mat, V_mat, direction_sign)
    th = th0;
    lr = 0.2;
    for iter = 1:600
        [~, u_tot, ~, ~, grad_tot] = evaluate_u_alpha(th, W, alpha, S_vals, U_mat, V_mat);
        step = direction_sign * grad_tot;
        if norm(step) < 1e-9
            break;
        end
        for bt = 1:20
            th_new = th + lr * step;
            [~, u_new] = evaluate_u_alpha(th_new, W, alpha, S_vals, U_mat, V_mat);
            if direction_sign * (u_new - u_tot) >= 0.1 * lr * (grad_tot' * step)
                th = th_new;
                break;
            end
            lr = lr * 0.5;
        end
        lr = min(lr * 1.1, 1.0);
    end
    th_opt = th;
    [~, u_opt] = evaluate_u_alpha(th_opt, W, alpha, S_vals, U_mat, V_mat);
end

function state = assemble_u_alpha_state(theta_norm, W, alpha, svd_data, state_name, ~)
    [U_j, U_total, U_mean, grad_local, grad_total, U_j_modes, Z_modes] = ...
        evaluate_u_alpha(theta_norm, W, alpha, svd_data.S, svd_data.U, svd_data.V);

    x = exp(1i * theta_norm);
    delta_theta_mat = wrap_to_pi(theta_norm.' - theta_norm); % delta(j, k) = th_k - th_j
    energy_matrix = W .* cos(delta_theta_mat + alpha);

    state = struct();
    state.name = state_name;
    state.theta_rad = theta_norm;
    state.theta_deg = rad2deg(theta_norm);
    state.phasor_x = x;
    state.U_j = U_j;
    state.U_total = U_total;
    state.U_mean = U_mean;
    state.grad_local = grad_local;
    state.grad_total = grad_total;
    state.grad_local_norm = norm(grad_local);
    state.grad_total_norm = norm(grad_total);
    state.U_j_modes = U_j_modes;
    state.U_total_modes = sum(U_j_modes, 1).';
    state.Z_modes = Z_modes;
    state.Z_modes_amp = abs(Z_modes);
    state.energy_matrix = energy_matrix;
    state.order_param = abs(mean(x));
end

function extrema = cluster_and_classify_u_alpha_extrema(candidates, W, alpha, svd_data, ref_idx)
    N = size(W, 1);
    tol_phase = 0.08;
    tol_u = 1e-4;

    raw_extrema = struct('theta_norm', {}, 'theta_deg', {}, 'theta_rad', {}, ...
        'U_total', {}, 'U_mean', {}, 'U_j', {}, 'grad_total_norm', {}, ...
        'grad_local_norm', {}, 'type', {}, 'type_code', {}, ...
        'order_param', {}, 'count', {}, 'is_global_max', {}, 'is_global_min', {}, ...
        'U_j_modes', {}, 'Z_modes', {});

    Q = null(ones(1, N)); % Subspace orthogonal to shift vector

    for c = 1:numel(candidates)
        th = candidates{c};
        if isempty(th), continue; end
        th_norm = normalize_phase_vector(th, ref_idx);
        [U_j, U_tot, U_m, grad_loc, grad_tot, U_modes, Z_m] = ...
            evaluate_u_alpha(th_norm, W, alpha, svd_data.S, svd_data.U, svd_data.V);

        if norm(grad_tot) > 5e-3
            continue;
        end

        match_idx = 0;
        for e = 1:numel(raw_extrema)
            diff_th = wrap_to_pi(th_norm - raw_extrema(e).theta_norm);
            d_phase = norm(diff_th);
            d_u = abs(U_tot - raw_extrema(e).U_total);
            if d_phase < tol_phase || (d_phase < 0.20 && d_u < tol_u)
                match_idx = e;
                break;
            end
        end

        if match_idx > 0
            raw_extrema(match_idx).count = raw_extrema(match_idx).count + 1;
        else
            % Compute numerical Hessian of U_total for stability analysis
            H = compute_numerical_hessian(th_norm, W, alpha, svd_data);
            H_proj = Q' * H * Q;
            lambda_proj = sort(real(eig(H_proj)), 'ascend');

            tol_zero = 1e-4;
            n_pos = sum(lambda_proj > tol_zero);
            n_neg = sum(lambda_proj < -tol_zero);

            if n_pos == 0
                type_str = 'Stable Maximum (Attractor)';
                type_code = 'STABLE_MAX';
            elseif n_neg == 0
                type_str = 'Unstable Minimum (Repeller)';
                type_code = 'UNSTABLE_MIN';
            else
                type_str = sprintf('Saddle Point (Index %d)', n_pos);
                type_code = sprintf('SADDLE_%d', n_pos);
            end

            x = exp(1i * th_norm);
            order_p = abs(mean(x));

            entry = struct();
            entry.theta_norm = th_norm;
            entry.theta_deg = rad2deg(th_norm);
            entry.theta_rad = th_norm;
            entry.U_total = U_tot;
            entry.U_mean = U_m;
            entry.U_j = U_j;
            entry.grad_total_norm = norm(grad_tot);
            entry.grad_local_norm = norm(grad_loc);
            entry.type = type_str;
            entry.type_code = type_code;
            entry.order_param = order_p;
            entry.count = 1;
            entry.is_global_max = false;
            entry.is_global_min = false;
            entry.U_j_modes = U_modes;
            entry.Z_modes = Z_m;

            raw_extrema(end+1) = entry; %#ok<AGROW>
        end
    end

    if ~isempty(raw_extrema)
        [~, sort_idx] = sort([raw_extrema.U_total], 'descend');
        extrema = raw_extrema(sort_idx);
        extrema(1).is_global_max = true;
        extrema(end).is_global_min = true;
    else
        extrema = raw_extrema;
    end
end

function H = compute_numerical_hessian(theta, W, alpha, svd_data)
    N = numel(theta);
    H = zeros(N, N);
    h_step = 1e-5;
    for k = 1:N
        th_plus = theta; th_plus(k) = th_plus(k) + h_step;
        th_minus = theta; th_minus(k) = th_minus(k) - h_step;
        [~, ~, ~, ~, g_plus] = evaluate_u_alpha(th_plus, W, alpha, svd_data.S, svd_data.U, svd_data.V);
        [~, ~, ~, ~, g_minus] = evaluate_u_alpha(th_minus, W, alpha, svd_data.S, svd_data.U, svd_data.V);
        H(:, k) = (g_plus - g_minus) / (2 * h_step);
    end
    H = 0.5 * (H + H.');
end

% =========================================================================
% GRADIENT FLOW TIME-SERIES DYNAMIC SIMULATION
% =========================================================================

function sim_res = run_u_alpha_gradient_flow_simulations(W, alpha, svd_data, theta0, duration_sec, dt, grad_gain, flow_type, ref_idx)
    time = (0:dt:duration_sec).';
    n_steps = numel(time);
    N = size(W, 1);
    R_modes = numel(svd_data.S);

    use_local_flow = strcmpi(flow_type, 'local');

    % Maximization flow
    theta_max_traj = zeros(n_steps, N);
    U_total_max_traj = zeros(n_steps, 1);
    U_mean_max_traj = zeros(n_steps, 1);
    U_j_max_traj = zeros(n_steps, N);
    Z_amp_max_traj = zeros(n_steps, R_modes);

    theta_curr = theta0(:);
    for t = 1:n_steps
        theta_max_traj(t, :) = theta_curr.';
        [U_j, U_tot, U_m, g_loc, g_tot, ~, Z_m] = evaluate_u_alpha(theta_curr, W, alpha, ...
            svd_data.S, svd_data.U, svd_data.V);
        U_total_max_traj(t) = U_tot;
        U_mean_max_traj(t) = U_m;
        U_j_max_traj(t, :) = U_j.';
        Z_amp_max_traj(t, :) = abs(Z_m).';

        if t < n_steps
            if use_local_flow
                g_step = g_loc;
            else
                g_step = g_tot;
            end
            theta_curr = theta_curr + dt * (grad_gain * g_step);
        end
    end

    % Minimization flow
    theta_min_traj = zeros(n_steps, N);
    U_total_min_traj = zeros(n_steps, 1);
    U_mean_min_traj = zeros(n_steps, 1);
    U_j_min_traj = zeros(n_steps, N);
    Z_amp_min_traj = zeros(n_steps, R_modes);

    theta_curr = theta0(:);
    for t = 1:n_steps
        theta_min_traj(t, :) = theta_curr.';
        [U_j, U_tot, U_m, g_loc, g_tot, ~, Z_m] = evaluate_u_alpha(theta_curr, W, alpha, ...
            svd_data.S, svd_data.U, svd_data.V);
        U_total_min_traj(t) = U_tot;
        U_mean_min_traj(t) = U_m;
        U_j_min_traj(t, :) = U_j.';
        Z_amp_min_traj(t, :) = abs(Z_m).';

        if t < n_steps
            if use_local_flow
                g_step = g_loc;
            else
                g_step = g_tot;
            end
            theta_curr = theta_curr - dt * (grad_gain * g_step);
        end
    end

    rel_theta_max = wrap_to_pi(theta_max_traj - theta_max_traj(:, ref_idx));
    rel_theta_min = wrap_to_pi(theta_min_traj - theta_min_traj(:, ref_idx));

    sim_res = struct();
    sim_res.time = time;
    sim_res.dt = dt;
    sim_res.grad_gain = grad_gain;
    sim_res.flow_type = flow_type;
    sim_res.theta_init = theta0;
    sim_res.max_traj = struct('theta', theta_max_traj, 'relative_phase', rel_theta_max, ...
        'U_total', U_total_max_traj, 'U_mean', U_mean_max_traj, 'U_j', U_j_max_traj, ...
        'Z_amp', Z_amp_max_traj);
    sim_res.min_traj = struct('theta', theta_min_traj, 'relative_phase', rel_theta_min, ...
        'U_total', U_total_min_traj, 'U_mean', U_mean_min_traj, 'U_j', U_j_min_traj, ...
        'Z_amp', Z_amp_min_traj);
end

% =========================================================================
% DATA LOADING
% =========================================================================

function [W, agent_ids, source_path, svd_data, alpha_auto] = load_network_and_svd_data(source)
    if isempty(source)
        source = fullfile('EstimateL', 'SStickFlat', 'low_rank_analysis', 'M10', ...
            'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
    end

    if isnumeric(source) && ismatrix(source) && size(source, 1) == size(source, 2)
        W = source;
        N = size(W, 1);
        agent_ids = 1:N;
        source_path = 'Direct numeric matrix';
        svd_data = compute_svd_struct(W, agent_ids);
        alpha_auto = 0.0;
        return;
    end

    if isstruct(source) && isfield(source, 'W')
        W = source.W;
        if isfield(source, 'agent_ids')
            agent_ids = source.agent_ids(:).';
        else
            agent_ids = 1:size(W, 1);
        end
        source_path = 'Struct field W';
        svd_data = compute_svd_struct(W, agent_ids);
        alpha_auto = 0.0;
        return;
    end

    source_str = char(source);
    candidate_csv = fullfile(source_str, 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
    estimate_candidate = fullfile('EstimateL', source_str, 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');

    if exist(source_str, 'file') && ~isfolder(source_str)
        target_file = source_str;
    elseif exist(candidate_csv, 'file')
        target_file = candidate_csv;
    elseif exist(estimate_candidate, 'file')
        target_file = estimate_candidate;
    else
        base_dir = fileparts(fileparts(mfilename('fullpath')));
        local_cand = fullfile(base_dir, source_str);
        if exist(local_cand, 'file') && ~isfolder(local_cand)
            target_file = local_cand;
        else
            error('Coupling matrix W CSV not found at: %s', source_str);
        end
    end

    source_path = target_file;
    T = readtable(target_file);

    var_names = T.Properties.VariableNames;
    sender_cols = var_names(startsWith(var_names, 'sender_agent_'));

    if ~isempty(sender_cols)
        agent_ids = zeros(1, numel(sender_cols));
        for k = 1:numel(sender_cols)
            agent_ids(k) = str2double(regexprep(sender_cols{k}, '^sender_agent_', ''));
        end
        W = table2array(T(:, sender_cols));
    else
        agent_ids = 1:height(T);
        W = table2array(T(:, 2:end));
    end

    N = size(W, 1);
    W(1:N+1:end) = 0;

    % Load SVD data and alpha from folder if present
    analysis_dir = fileparts(target_file);
    svd_contrib_csv = fullfile(analysis_dir, 'agent_svd_contributions.csv');
    svd_summary_csv = fullfile(analysis_dir, 'network_svd_modes_summary.csv');
    delta_csv = fullfile(analysis_dir, 'sender_phase_shift_delta.csv');

    alpha_auto = [];
    if exist(delta_csv, 'file')
        T_delta = readtable(delta_csv);
        if ismember('delta_rad', T_delta.Properties.VariableNames)
            alpha_auto = T_delta.delta_rad(1);
        end
    end

    if exist(svd_contrib_csv, 'file') && exist(svd_summary_csv, 'file')
        svd_data = load_existing_svd_csvs(svd_contrib_csv, svd_summary_csv, W, agent_ids);
    else
        svd_data = compute_svd_struct(W, agent_ids);
    end
end

function svd_data = load_existing_svd_csvs(contrib_csv, summary_csv, W, agent_ids)
    T_sum = readtable(summary_csv);
    S_vals = T_sum.singular_value_sigma(:);
    R = numel(S_vals);

    T_contrib = readtable(contrib_csv);
    N = numel(agent_ids);
    U_mat = zeros(N, R);
    V_mat = zeros(N, R);

    for m = 1:R
        u_col = sprintf('receiver_u_mode%d', m);
        v_col = sprintf('sender_v_mode%d', m);
        if ismember(u_col, T_contrib.Properties.VariableNames)
            U_mat(:, m) = T_contrib.(u_col);
        end
        if ismember(v_col, T_contrib.Properties.VariableNames)
            V_mat(:, m) = T_contrib.(v_col);
        end
    end

    svd_data = struct('S', S_vals, 'U', U_mat, 'V', V_mat, 'R', R);
end

function svd_data = compute_svd_struct(W, agent_ids)
    [U_raw, S_raw, V_raw] = svd(W);
    S_vals = diag(S_raw);
    R = numel(S_vals);
    svd_data = struct('S', S_vals, 'U', U_raw, 'V', V_raw, 'R', R);
end

% =========================================================================
% PLOTTING FUNCTIONS (LaTeX Enabled, Professional Academic Style)
% =========================================================================

function fig = plot_u_alpha_trajectories(sim_res, agent_ids, ref_agent_id, offset, alpha, max_u, min_u)
    fig = figure('Color', 'w', 'Position', [100, 100, 1080, 680], ...
        'Name', 'Gradient Flow Relaxation Dynamics of U_j(theta, alpha)');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    time = sim_res.time;
    N = numel(agent_ids);
    colors = lines(N);

    % --- Subplot 1: Maximization Relative Phase Trajectories ---
    ax1 = nexttile(1);
    hold(ax1, 'on');
    for k = 1:N
        y_val = sim_res.max_traj.relative_phase(:, k);
        for j = 2:numel(y_val)
            if abs(y_val(j) - y_val(j-1)) > pi
                y_val(j) = NaN;
            end
        end
        plot(ax1, time, y_val, 'LineWidth', 1.6, 'Color', colors(k, :), ...
            'DisplayName', sprintf('$\\mathrm{Agent\\ %d}$', agent_ids(k) + offset));
    end
    format_angle_axis(ax1, time);
    ylabel(ax1, sprintf('$\\theta_j - \\theta_{%d}$ (rad)', ref_agent_id + offset), 'Interpreter', 'latex', 'FontSize', 12);
    title(ax1, sprintf('Maximization Flow: Relative Phase Evolution ($\\alpha = %+.2f^\\circ$)', rad2deg(alpha)), ...
        'Interpreter', 'latex', 'FontSize', 12);
    legend(ax1, 'Location', 'eastoutside', 'Interpreter', 'latex');

    % --- Subplot 2: Minimization Relative Phase Trajectories ---
    ax2 = nexttile(2);
    hold(ax2, 'on');
    for k = 1:N
        y_val = sim_res.min_traj.relative_phase(:, k);
        for j = 2:numel(y_val)
            if abs(y_val(j) - y_val(j-1)) > pi
                y_val(j) = NaN;
            end
        end
        plot(ax2, time, y_val, 'LineWidth', 1.6, 'Color', colors(k, :), ...
            'DisplayName', sprintf('$\\mathrm{Agent\\ %d}$', agent_ids(k) + offset));
    end
    format_angle_axis(ax2, time);
    ylabel(ax2, sprintf('$\\theta_j - \\theta_{%d}$ (rad)', ref_agent_id + offset), 'Interpreter', 'latex', 'FontSize', 12);
    title(ax2, sprintf('Minimization Flow: Relative Phase Evolution ($\\alpha = %+.2f^\\circ$)', rad2deg(alpha)), ...
        'Interpreter', 'latex', 'FontSize', 12);
    legend(ax2, 'Location', 'eastoutside', 'Interpreter', 'latex');

    % --- Subplot 3: Maximization Mean Potential U_mean(t) & Mode Amplitudes ---
    ax3 = nexttile(3);
    hold(ax3, 'on');
    yyaxis(ax3, 'left');
    plot(ax3, time, sim_res.max_traj.U_mean, 'b-', 'LineWidth', 2.0, 'DisplayName', '$U_{\mathrm{mean}}(t)$');
    yline(ax3, max_u, 'b--', 'LineWidth', 1.2, 'DisplayName', sprintf('$U_{\\mathrm{mean}}^{\\max} = %+0.5f$', max_u));
    ylabel(ax3, '$U_{\mathrm{mean}}(\theta)$', 'Interpreter', 'latex', 'FontSize', 12);
    grid(ax3, 'on'); box(ax3, 'on');
    xlim(ax3, [time(1), time(end)]);
    set(ax3, 'TickLabelInterpreter', 'latex');

    yyaxis(ax3, 'right');
    R_modes = size(sim_res.max_traj.Z_amp, 2);
    mode_styles = {'-', '--', ':', '-.'};
    for m = 1:min(R_modes, 3)
        plot(ax3, time, sim_res.max_traj.Z_amp(:, m), mode_styles{m}, 'LineWidth', 1.3, ...
            'DisplayName', sprintf('$|Z_{%d}(t)|$', m));
    end
    ylabel(ax3, 'Order Parameters $|Z_\ell(t)|$', 'Interpreter', 'latex', 'FontSize', 12);
    xlabel(ax3, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 11);
    title(ax3, 'Ascent Dynamics: Potential $U_{\mathrm{mean}}(t)$ \& Modes $|Z_\ell(t)|$', 'Interpreter', 'latex', 'FontSize', 12);
    legend(ax3, 'Location', 'southeast', 'Interpreter', 'latex');

    % --- Subplot 4: Minimization Mean Potential U_mean(t) & Mode Amplitudes ---
    ax4 = nexttile(4);
    hold(ax4, 'on');
    yyaxis(ax4, 'left');
    plot(ax4, time, sim_res.min_traj.U_mean, 'r-', 'LineWidth', 2.0, 'DisplayName', '$U_{\mathrm{mean}}(t)$');
    yline(ax4, min_u, 'r--', 'LineWidth', 1.2, 'DisplayName', sprintf('$U_{\\mathrm{mean}}^{\\min} = %+0.5f$', min_u));
    ylabel(ax4, '$U_{\mathrm{mean}}(\theta)$', 'Interpreter', 'latex', 'FontSize', 12);
    grid(ax4, 'on'); box(ax4, 'on');
    xlim(ax4, [time(1), time(end)]);
    set(ax4, 'TickLabelInterpreter', 'latex');

    yyaxis(ax4, 'right');
    for m = 1:min(R_modes, 3)
        plot(ax4, time, sim_res.min_traj.Z_amp(:, m), mode_styles{m}, 'LineWidth', 1.3, ...
            'DisplayName', sprintf('$|Z_{%d}(t)|$', m));
    end
    ylabel(ax4, 'Order Parameters $|Z_\ell(t)|$', 'Interpreter', 'latex', 'FontSize', 12);
    xlabel(ax4, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 11);
    title(ax4, 'Descent Dynamics: Potential $U_{\mathrm{mean}}(t)$ \& Modes $|Z_\ell(t)|$', 'Interpreter', 'latex', 'FontSize', 12);
    legend(ax4, 'Location', 'northeast', 'Interpreter', 'latex');

    apply_figure_styling(fig);
end

function fig = plot_u_alpha_phasor_diagrams(results, offset)
    fig = figure('Color', 'w', 'Position', [150, 150, 1000, 480], ...
        'Name', 'Optimal Phasors & SVD Order Vectors on Complex Unit Circle');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Subplot 1: Maximum
    ax1 = nexttile(1);
    title_max = sprintf('Global Maximum: $U_{\\mathrm{mean}} = %+0.5f$ ($\\alpha = %+.1f^\\circ$)', ...
        results.maximum.U_mean, results.alpha_deg);
    render_single_u_alpha_phasor(ax1, results.maximum, results.agent_ids, results.W, results.alpha, ...
        results.svd, offset, title_max);

    % Subplot 2: Minimum
    ax2 = nexttile(2);
    title_min = sprintf('Global Minimum: $U_{\\mathrm{mean}} = %+0.5f$ ($\\alpha = %+.1f^\\circ$)', ...
        results.minimum.U_mean, results.alpha_deg);
    render_single_u_alpha_phasor(ax2, results.minimum, results.agent_ids, results.W, results.alpha, ...
        results.svd, offset, title_min);

    apply_figure_styling(fig);
end

function render_single_u_alpha_phasor(ax, state, agent_ids, W, alpha, svd_data, offset, title_str)
    hold(ax, 'on'); axis(ax, 'equal');
    xlim(ax, [-1.65, 1.65]); ylim(ax, [-1.65, 1.65]);
    grid(ax, 'on'); box(ax, 'on');
    set(ax, 'TickLabelInterpreter', 'latex');

    circle_th = linspace(0, 2*pi, 300);
    plot(ax, cos(circle_th), sin(circle_th), 'Color', [0.75, 0.75, 0.75], 'LineWidth', 1.2);
    plot(ax, [-1.35, 1.35], [0, 0], 'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.8);
    plot(ax, [0, 0], [-1.35, 1.35], 'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.8);

    N = numel(agent_ids);
    th_vals = state.theta_rad;
    x_coords = cos(th_vals);
    y_coords = sin(th_vals);

    % Draw interaction coupling edges with alpha phase angle
    max_w = max(abs(W(:))) + eps;
    for j = 1:N
        for k = (j+1):N
            w_val = 0.5 * (W(j, k) + W(k, j));
            if abs(w_val) < 1e-6, continue; end
            lw = 0.5 + 3.0 * (abs(w_val) / max_w);
            if w_val > 0
                c_edge = [0.15, 0.45, 0.85, 0.6];
            else
                c_edge = [0.85, 0.20, 0.15, 0.6];
            end
            plot(ax, [x_coords(j), x_coords(k)], [y_coords(j), y_coords(k)], ...
                'Color', c_edge, 'LineWidth', lw);
        end
    end

    % Draw SVD mode order parameter vectors Z_ell inside the circle
    mode_colors = [0.1, 0.7, 0.2; 0.9, 0.5, 0.0; 0.6, 0.2, 0.8];
    for ell = 1:min(numel(svd_data.S), 2)
        Z_val = state.Z_modes(ell);
        if abs(Z_val) > 1e-4
            quiver(ax, 0, 0, real(Z_val), imag(Z_val), 0, 'Color', mode_colors(ell, :), ...
                'LineWidth', 2.0, 'MaxHeadSize', 0.4, ...
                'DisplayName', sprintf('Mode %d: $Z_{%d}$', ell, ell));
        end
    end

    % Draw agent node markers
    colors = lines(N);
    visited = false(N, 1);
    for i = 1:N
        if visited(i), continue; end
        cluster = i;
        for j = i+1:N
            if abs(wrap_to_pi(th_vals(i) - th_vals(j))) < 0.15
                cluster = [cluster, j]; %#ok<AGROW>
            end
        end
        visited(cluster) = true;

        for m = 1:numel(cluster)
            idx = cluster(m);
            plot(ax, x_coords(idx), y_coords(idx), 'o', 'MarkerSize', 11, ...
                'MarkerFaceColor', colors(idx, :), 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
        end

        mean_th = angle(mean(exp(1i * th_vals(cluster))));
        cx = cos(mean_th);
        cy = sin(mean_th);

        id_strs = arrayfun(@(id) sprintf('%d', id + offset), agent_ids(cluster), 'UniformOutput', false);
        id_combined = strjoin(id_strs, ',');
        deg_val = rad2deg(mean_th);

        tx = 1.25 * cx;
        ty = 1.25 * cy;
        text(ax, tx, ty, sprintf('$\\mathrm{ID\\ %s}$\n$(%+0.0f^\\circ)$', id_combined, deg_val), ...
            'Interpreter', 'latex', 'FontSize', 10, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'BackgroundColor', [1, 1, 1, 0.85], 'Margin', 1);
    end

    title(ax, title_str, 'Interpreter', 'latex', 'FontSize', 12);
    xlabel(ax, '$\mathrm{Re}(x_j)$', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax, '$\mathrm{Im}(x_j)$', 'Interpreter', 'latex', 'FontSize', 11);
end

function fig = plot_u_alpha_mode_contributions(results, offset)
    fig = figure('Color', 'w', 'Position', [150, 150, 1050, 420], ...
        'Name', 'SVD Mode Decomposition of Potentials U_j');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    agent_labels = arrayfun(@(id) sprintf('$\\mathrm{ID\\ %d}$', id + offset), results.agent_ids, 'UniformOutput', false);
    R_modes = numel(results.svd.S);
    mode_labels = arrayfun(@(m) sprintf('Mode %d ($s_%d=%.3f$)', m, m, results.svd.S(m)), 1:R_modes, 'UniformOutput', false);

    % Subplot 1: Maximum state modal contributions
    ax1 = nexttile(1);
    b1 = bar(ax1, 1:results.N, results.maximum.U_j_modes, 'stacked');
    grid(ax1, 'on'); box(ax1, 'on');
    set(ax1, 'XTick', 1:results.N, 'XTickLabel', agent_labels, 'TickLabelInterpreter', 'latex');
    ylabel(ax1, '$U_j^{(\ell)} = s_\ell u_{j\ell} \mathrm{Re}[e^{i\alpha}\overline{x_j} Z_\ell]$', 'Interpreter', 'latex', 'FontSize', 11);
    xlabel(ax1, 'Agent $j$', 'Interpreter', 'latex', 'FontSize', 11);
    title(ax1, sprintf('Global Maximum: Mode Decomposition ($U_{\\mathrm{total}} = %+0.5f$)', results.maximum.U_total), ...
        'Interpreter', 'latex', 'FontSize', 12);

    % Subplot 2: Minimum state modal contributions
    ax2 = nexttile(2);
    bar(ax2, 1:results.N, results.minimum.U_j_modes, 'stacked');
    grid(ax2, 'on'); box(ax2, 'on');
    set(ax2, 'XTick', 1:results.N, 'XTickLabel', agent_labels, 'TickLabelInterpreter', 'latex');
    ylabel(ax2, '$U_j^{(\ell)} = s_\ell u_{j\ell} \mathrm{Re}[e^{i\alpha}\overline{x_j} Z_\ell]$', 'Interpreter', 'latex', 'FontSize', 11);
    xlabel(ax2, 'Agent $j$', 'Interpreter', 'latex', 'FontSize', 11);
    title(ax2, sprintf('Global Minimum: Mode Decomposition ($U_{\\mathrm{total}} = %+0.5f$)', results.minimum.U_total), ...
        'Interpreter', 'latex', 'FontSize', 12);

    lgd = legend(ax1, b1, mode_labels, 'Orientation', 'horizontal', 'Interpreter', 'latex', 'FontSize', 10);
    lgd.Layout.Tile = 'north';

    apply_figure_styling(fig);
end

function fig = plot_u_alpha_pairwise_energy(results, offset)
    fig = figure('Color', 'w', 'Position', [150, 150, 950, 420], ...
        'Name', 'Pairwise Phase-Shifted Interaction Energy Matrix');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    agent_labels = arrayfun(@(id) sprintf('$\\mathrm{ID\\ %d}$', id + offset), results.agent_ids, 'UniformOutput', false);
    max_val = max([abs(results.maximum.energy_matrix(:)); abs(results.minimum.energy_matrix(:)); eps]);

    % Subplot 1: Maximum
    ax1 = nexttile(1);
    imagesc(ax1, results.maximum.energy_matrix, [-max_val, max_val]);
    colormap(ax1, balance_colormap());
    colorbar(ax1); axis(ax1, 'square');
    set(ax1, 'XTick', 1:results.N, 'XTickLabel', agent_labels, 'YTick', 1:results.N, 'YTickLabel', agent_labels, ...
        'TickLabelInterpreter', 'latex');
    title(ax1, sprintf('Max Pairwise $E_{jk} = W_{jk}\\cos(\\theta_k - \\theta_j + \\alpha)$\n$U_{\\mathrm{total}} = %+0.5f$', ...
        results.maximum.U_total), 'Interpreter', 'latex', 'FontSize', 11);
    xlabel(ax1, 'Sender Agent $k$', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax1, 'Receiver Agent $j$', 'Interpreter', 'latex', 'FontSize', 11);

    % Subplot 2: Minimum
    ax2 = nexttile(2);
    imagesc(ax2, results.minimum.energy_matrix, [-max_val, max_val]);
    colormap(ax2, balance_colormap());
    colorbar(ax2); axis(ax2, 'square');
    set(ax2, 'XTick', 1:results.N, 'XTickLabel', agent_labels, 'YTick', 1:results.N, 'YTickLabel', agent_labels, ...
        'TickLabelInterpreter', 'latex');
    title(ax2, sprintf('Min Pairwise $E_{jk} = W_{jk}\\cos(\\theta_k - \\theta_j + \\alpha)$\n$U_{\\mathrm{total}} = %+0.5f$', ...
        results.minimum.U_total), 'Interpreter', 'latex', 'FontSize', 11);
    xlabel(ax2, 'Sender Agent $k$', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax2, 'Receiver Agent $j$', 'Interpreter', 'latex', 'FontSize', 11);

    apply_figure_styling(fig);
end

function cmap = balance_colormap()
    N_c = 256;
    r = [linspace(0.1, 0.95, N_c/2), linspace(0.95, 0.8, N_c/2)].';
    g = [linspace(0.2, 0.95, N_c/2), linspace(0.95, 0.2, N_c/2)].';
    b = [linspace(0.8, 0.95, N_c/2), linspace(0.95, 0.1, N_c/2)].';
    cmap = [r, g, b];
end

function format_angle_axis(ax, time)
    grid(ax, 'on'); box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    ylim(ax, [-pi, pi]);
    yticks(ax, [-pi, -pi/2, 0, pi/2, pi]);
    yticklabels(ax, {'$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'});
    set(ax, 'TickLabelInterpreter', 'latex');
    xlabel(ax, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 11);
end

function apply_figure_styling(fig)
    set(fig, 'Color', 'w');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    all_axes = findobj(fig, 'Type', 'axes');
    for a = 1:numel(all_axes)
        ax = all_axes(a);
        set(ax, 'FontName', 'Times', 'Box', 'on', 'TickLabelInterpreter', 'latex');
        cur_fs = get(ax, 'FontSize');
        if isempty(cur_fs) || cur_fs > 13 || cur_fs < 10
            set(ax, 'FontSize', 11);
        end

        if ~isempty(ax.Title) && ~isempty(ax.Title.String)
            set(ax.Title, 'Interpreter', 'latex');
        end
        if ~isempty(ax.XLabel) && ~isempty(ax.XLabel.String)
            set(ax.XLabel, 'Interpreter', 'latex');
        end
        if ~isempty(ax.YLabel) && ~isempty(ax.YLabel.String)
            set(ax.YLabel, 'Interpreter', 'latex');
        end
    end

    all_legends = findobj(fig, 'Type', 'legend');
    for l = 1:numel(all_legends)
        set(all_legends(l), 'Interpreter', 'latex', 'FontSize', 10);
    end

    all_texts = findobj(fig, 'Type', 'text');
    for t = 1:numel(all_texts)
        set(all_texts(t), 'Interpreter', 'latex');
    end
end

function export_figure_to_png(fig, file_path)
    apply_figure_styling(fig);
    has_exportgraphics = ~isempty(which('exportgraphics')) || (exist('exportgraphics', 'builtin') == 5);
    if has_exportgraphics
        exportgraphics(fig, file_path, 'Resolution', 300);
    else
        saveas(fig, file_path);
    end
end

% =========================================================================
% PRINT & EXPORT HELPERS
% =========================================================================

function print_u_alpha_results_summary(res, offset)
    fprintf('-------------------------------------------------------------------------\n');
    fprintf('  OPTIMAL RESULTS SUMMARY (alpha = %+.4f rad = %+.2f deg)\n', res.alpha, res.alpha_deg);
    fprintf('-------------------------------------------------------------------------\n');

    % Print Maximum
    fprintf('  [GLOBAL MAXIMUM]\n');
    fprintf('    Total Potential U_total = %+8.6f  (Mean U = %+8.6f)\n', ...
        res.maximum.U_total, res.maximum.U_mean);
    fprintf('    Order Parameter |R| = %.4f | Local Gradient Norm = %.2e\n', ...
        res.maximum.order_param, res.maximum.grad_local_norm);
    fprintf('    Relative Phases (deg):\n');
    for i = 1:res.N
        fprintf('      Agent ID %2d (idx %d):  %+8.2f deg  (%+7.4f rad) | U_j = %+8.6f\n', ...
            res.agent_ids(i) + offset, i, res.maximum.theta_deg(i), res.maximum.theta_rad(i), res.maximum.U_j(i));
    end
    fprintf('    SVD Mode Contributions to U_total:\n');
    for m = 1:numel(res.svd.S)
        fprintf('      Mode %d (s_%d = %.4f): U_total^{(%d)} = %+8.6f (Share: %5.1f%%) | |Z_%d| = %.4f\n', ...
            m, m, res.svd.S(m), m, res.maximum.U_total_modes(m), ...
            100 * res.maximum.U_total_modes(m) / (res.maximum.U_total + eps), ...
            m, res.maximum.Z_modes_amp(m));
    end
    fprintf('\n');

    % Print Minimum
    fprintf('  [GLOBAL MINIMUM]\n');
    fprintf('    Total Potential U_total = %+8.6f  (Mean U = %+8.6f)\n', ...
        res.minimum.U_total, res.minimum.U_mean);
    fprintf('    Order Parameter |R| = %.4f | Local Gradient Norm = %.2e\n', ...
        res.minimum.order_param, res.minimum.grad_local_norm);
    fprintf('    Relative Phases (deg):\n');
    for i = 1:res.N
        fprintf('      Agent ID %2d (idx %d):  %+8.2f deg  (%+7.4f rad) | U_j = %+8.6f\n', ...
            res.agent_ids(i) + offset, i, res.minimum.theta_deg(i), res.minimum.theta_rad(i), res.minimum.U_j(i));
    end
    fprintf('    SVD Mode Contributions to U_total:\n');
    for m = 1:numel(res.svd.S)
        fprintf('      Mode %d (s_%d = %.4f): U_total^{(%d)} = %+8.6f (Share: %5.1f%%) | |Z_%d| = %.4f\n', ...
            m, m, res.svd.S(m), m, res.minimum.U_total_modes(m), ...
            100 * res.minimum.U_total_modes(m) / (res.minimum.U_total + eps), ...
            m, res.minimum.Z_modes_amp(m));
    end
    fprintf('\n');

    % Print Discovered Stationary Extrema
    if isfield(res, 'extrema') && ~isempty(res.extrema)
        fprintf('-------------------------------------------------------------------------\n');
        fprintf('  ALL IDENTIFIED STATIONARY EXTREMA & EQUILIBRIA (N_extrema = %d)\n', numel(res.extrema));
        fprintf('-------------------------------------------------------------------------\n');
        fprintf('  #   Type                       U_total    U_mean    |R|    Hits  Phases (deg) [Agent %s]\n', ...
            mat2str(res.agent_ids + offset));
        for e = 1:numel(res.extrema)
            ex = res.extrema(e);
            marker = ' ';
            if isfield(ex, 'is_global_max') && ex.is_global_max
                marker = '*';
            elseif isfield(ex, 'is_global_min') && ex.is_global_min
                marker = '-';
            end
            deg_str = sprintf('%+6.1f ', ex.theta_deg);
            fprintf('  %2d%c %-26s %+9.6f %+9.6f %5.3f  %4d  [%s]\n', ...
                e, marker, ex.type, ex.U_total, ex.U_mean, ex.order_param, ex.count, strtrim(deg_str));
        end
        fprintf('  (*: Global Maximum / Stable Attractor, -: Global Minimum / Repeller)\n');
    end
    fprintf('-------------------------------------------------------------------------\n\n');
end

function export = save_u_alpha_analysis_outputs(results, opts)
    out_dir = opts.OutputDir;
    if isempty(out_dir)
        source_dir = fileparts(results.source_path);
        out_dir = fullfile(source_dir, 'u_alpha_optimal_phase_analysis');
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % 1. Save optimal phase summary CSV (global max & min)
    summary_csv = fullfile(out_dir, 'u_alpha_optimal_phases_summary.csv');
    var_names = {'agent_id', 'max_phase_deg', 'max_phase_rad', 'max_U_j', ...
                 'min_phase_deg', 'min_phase_rad', 'min_U_j'};
    T_summary = table(results.agent_ids(:), results.maximum.theta_deg(:), results.maximum.theta_rad(:), ...
                      results.maximum.U_j(:), results.minimum.theta_deg(:), results.minimum.theta_rad(:), ...
                      results.minimum.U_j(:), 'VariableNames', var_names);
    writetable(T_summary, summary_csv);

    % 2. Save all identified stationary extrema & equilibria CSV
    extrema_csv = fullfile(out_dir, 'u_alpha_extrema_phase_relationships.csv');
    if isfield(results, 'extrema') && ~isempty(results.extrema)
        M_ext = numel(results.extrema);
        ext_ids = (1:M_ext).';
        ext_types = {results.extrema.type}.';
        ext_type_codes = {results.extrema.type_code}.';
        is_gmax = [results.extrema.is_global_max].';
        is_gmin = [results.extrema.is_global_min].';
        u_totals = [results.extrema.U_total].';
        u_means = [results.extrema.U_mean].';
        order_params = [results.extrema.order_param].';
        grad_norms = [results.extrema.grad_total_norm].';
        basin_hits = [results.extrema.count].';

        T_ext = table(ext_ids, ext_types, ext_type_codes, is_gmax, is_gmin, ...
                      u_totals, u_means, order_params, grad_norms, basin_hits, ...
                      'VariableNames', {'extrema_id', 'classification', 'type_code', ...
                      'is_global_max', 'is_global_min', 'U_total', 'U_mean', ...
                      'order_parameter', 'gradient_norm', 'basin_hits'});

        for a = 1:results.N
            col_deg = sprintf('phi_deg_agent_%d', results.agent_ids(a));
            col_rad = sprintf('phi_rad_agent_%d', results.agent_ids(a));
            col_uj = sprintf('U_agent_%d', results.agent_ids(a));
            vals_deg = zeros(M_ext, 1);
            vals_rad = zeros(M_ext, 1);
            vals_uj = zeros(M_ext, 1);
            for e = 1:M_ext
                vals_deg(e) = results.extrema(e).theta_deg(a);
                vals_rad(e) = results.extrema(e).theta_rad(a);
                vals_uj(e) = results.extrema(e).U_j(a);
            end
            T_ext.(col_deg) = vals_deg;
            T_ext.(col_rad) = vals_rad;
            T_ext.(col_uj) = vals_uj;
        end
        writetable(T_ext, extrema_csv);
    else
        extrema_csv = '';
    end

    % 3. Save SVD mode contributions detailed summary CSV
    mode_contrib_csv = fullfile(out_dir, 'u_alpha_mode_contributions_summary.csv');
    R_modes = numel(results.svd.S);
    N = results.N;
    agent_id_col = results.agent_ids(:);

    T_modes = table(agent_id_col, 'VariableNames', {'agent_id'});
    for m = 1:R_modes
        T_modes.(sprintf('max_mode_%d_U_j', m)) = results.maximum.U_j_modes(:, m);
        T_modes.(sprintf('min_mode_%d_U_j', m)) = results.minimum.U_j_modes(:, m);
    end
    writetable(T_modes, mode_contrib_csv);

    % 4. Save complete MAT results (exclude figure handles)
    mat_path = fullfile(out_dir, 'u_alpha_optimal_phase_results.mat');
    res_save = results;
    if isfield(res_save, 'figures')
        res_save = rmfield(res_save, 'figures');
    end
    save(mat_path, 'res_save', '-v7.3');

    % 5. Save PNG figures (300 DPI)
    saved_figs = {};
    if isfield(results, 'figures')
        if ~isempty(results.figures.trajectories) && isvalid(results.figures.trajectories)
            p_traj = fullfile(out_dir, 'u_alpha_gradient_flow_trajectories.png');
            export_figure_to_png(results.figures.trajectories, p_traj);
            saved_figs{end+1} = p_traj; %#ok<AGROW>
        end
        if ~isempty(results.figures.phasors) && isvalid(results.figures.phasors)
            p_phasor = fullfile(out_dir, 'u_alpha_optimal_phasor_diagrams.png');
            export_figure_to_png(results.figures.phasors, p_phasor);
            saved_figs{end+1} = p_phasor; %#ok<AGROW>
        end
        if ~isempty(results.figures.modes) && isvalid(results.figures.modes)
            p_modes = fullfile(out_dir, 'u_alpha_mode_contributions.png');
            export_figure_to_png(results.figures.modes, p_modes);
            saved_figs{end+1} = p_modes; %#ok<AGROW>
        end
        if ~isempty(results.figures.energy_matrix) && isvalid(results.figures.energy_matrix)
            p_energy = fullfile(out_dir, 'u_alpha_pairwise_interaction_matrix.png');
            export_figure_to_png(results.figures.energy_matrix, p_energy);
            saved_figs{end+1} = p_energy; %#ok<AGROW>
        end
    end

    export = struct('output_dir', out_dir, 'summary_csv', summary_csv, ...
                    'extrema_csv', extrema_csv, 'mode_contrib_csv', mode_contrib_csv, ...
                    'mat_path', mat_path, 'saved_figures', {saved_figs});
    fprintf('[INFO] Saved analysis outputs to: %s\n', out_dir);
    if ~isempty(extrema_csv)
        fprintf('[INFO] Extrema phase relationships table: %s\n', extrema_csv);
    end
    if ~isempty(mode_contrib_csv)
        fprintf('[INFO] SVD modal contributions table: %s\n', mode_contrib_csv);
    end
end

function theta_norm = normalize_phase_vector(theta, ref_idx)
    theta = theta(:);
    shifted = theta - theta(ref_idx);
    theta_norm = wrap_to_pi(shifted);
    theta_norm(ref_idx) = 0;
end

function phase_wrapped = wrap_to_pi(phase)
    phase_wrapped = atan2(sin(phase), cos(phase));
end

function opts = parse_u_alpha_options(def_alpha, def_num_starts, def_rand_seed, def_init_theta, def_flow, ...
    def_sim_time, def_dt, def_gain, def_ref_id, def_plot_traj, def_plot_phasors, ...
    def_plot_modes, def_plot_energy, def_offset, def_save, def_outdir, varargin)

    p = inputParser;
    p.CaseSensitive = false;
    p.KeepUnmatched = true;

    addParameter(p, 'Alpha', def_alpha, @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'AlphaRad', def_alpha, @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'AlphaDeg', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'NumStarts', def_num_starts, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'RandomSeed', def_rand_seed, @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'InitialTheta', def_init_theta, @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'FlowType', def_flow, @(x) ischar(x) || isstring(x));
    addParameter(p, 'SimTimeSec', def_sim_time, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'SimDt', def_dt, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'GradGain', def_gain, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'ReferenceAgentId', def_ref_id, @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'PlotTrajectories', def_plot_traj, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotPhasors', def_plot_phasors, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotModes', def_plot_modes, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotEnergyMatrix', def_plot_energy, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'AgentOffset', def_offset, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'SaveOutputs', def_save, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'OutputDir', def_outdir, @(x) ischar(x) || isstring(x));

    % Aliases
    addParameter(p, 'Theta0', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'ThetaInit', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'SavePlots', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'SavePlot', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotTraj', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotMode', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotEnergy', [], @(x) islogical(x) || isnumeric(x));

    parse(p, varargin{:});
    opts = p.Results;

    if ~isempty(opts.Alpha)
        opts.AlphaRad = opts.Alpha;
    end
    if ~isempty(opts.Theta0)
        opts.InitialTheta = opts.Theta0;
    end
    if ~isempty(opts.ThetaInit)
        opts.InitialTheta = opts.ThetaInit;
    end
    if ~isempty(opts.SavePlots)
        opts.SaveOutputs = opts.SavePlots;
    end
    if ~isempty(opts.SavePlot)
        opts.SaveOutputs = opts.SavePlot;
    end
    if ~isempty(opts.PlotTraj)
        opts.PlotTrajectories = opts.PlotTraj;
    end
    if ~isempty(opts.PlotMode)
        opts.PlotModes = opts.PlotMode;
    end
    if ~isempty(opts.PlotEnergy)
        opts.PlotEnergyMatrix = opts.PlotEnergy;
    end

    opts.FlowType = char(opts.FlowType);
    opts.OutputDir = char(opts.OutputDir);
    opts.PlotTrajectories = logical(opts.PlotTrajectories);
    opts.PlotPhasors = logical(opts.PlotPhasors);
    opts.PlotModes = logical(opts.PlotModes);
    opts.PlotEnergyMatrix = logical(opts.PlotEnergyMatrix);
    opts.SaveOutputs = logical(opts.SaveOutputs);
end
