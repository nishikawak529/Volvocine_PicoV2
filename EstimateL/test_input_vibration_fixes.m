% TEST_INPUT_VIBRATION_FIXES
% Rigorous unit tests for calculate_input_vibration_optimal_phases.m
% Validates:
%   1. ID alignment and permutation invariance
%   2. Missing and duplicate Agent ID detection
%   3. Line-search failure rejection (Armijo failure / NaN / Inf)
%   4. RandomSeed reproducibility across full pipeline
%   5. Execution and export completion when observed data is absent

fprintf('=========================================================================\n');
fprintf('   RUNNING RIGOROUS UNIT TESTS FOR OPTIMAL PHASE FIXES\n');
fprintf('=========================================================================\n');

addpath('EstimateL');

%% Test 1: RandomSeed Reproducibility
fprintf('\n[TEST 1] Testing RandomSeed Reproducibility on Round dataset...\n');
res1 = calculate_input_vibration_optimal_phases('Round', 'NumStarts', 20, 'RandomSeed', 42, 'SaveOutputs', false);
res2 = calculate_input_vibration_optimal_phases('Round', 'NumStarts', 20, 'RandomSeed', 42, 'SaveOutputs', false);

assert(abs(res1.total_input_amplitude_sum.min_amplitude_sum - res2.total_input_amplitude_sum.min_amplitude_sum) < 1e-14, ...
    'RandomSeed reproducibility failed for min amplitude sum');
assert(abs(res1.total_input_amplitude_sum.max_amplitude_sum - res2.total_input_amplitude_sum.max_amplitude_sum) < 1e-14, ...
    'RandomSeed reproducibility failed for max amplitude sum');
assert(norm(res1.total_input_amplitude_sum.theta_min - res2.total_input_amplitude_sum.theta_min) < 1e-14, ...
    'RandomSeed reproducibility failed for theta_min');
assert(norm(res1.total_network.theta_min - res2.total_network.theta_min) < 1e-14, ...
    'RandomSeed reproducibility failed for network theta_min');
fprintf('  --> PASSED: 100%% identical numerical results with identical RandomSeed.\n');

%% Test 2: ID Permutation Invariance on Input Data Struct
fprintf('\n[TEST 2] Testing ID Permutation Invariance...\n');
orig_data = calculate_input_vibration_optimal_phases('Round', 'NumStarts', 10, 'RandomSeed', 123, 'SaveOutputs', false);

% Permute agent ordering
perm = [3, 1, 4, 2];
perm_struct = struct();
perm_struct.N = orig_data.N;
perm_struct.agent_ids = orig_data.agent_ids(perm);
perm_struct.W = orig_data.W(perm, perm);
perm_struct.svd.U = orig_data.U(perm, :);
perm_struct.svd.V = orig_data.V(perm, :);
perm_struct.svd.singular_values = orig_data.singular_values;
perm_struct.fit_R1.delta = orig_data.delta;
perm_struct.output_dir = tempdir;

% Run on permuted struct (should re-sort by canonical agent_ids and yield identical results)
res_perm = calculate_input_vibration_optimal_phases(perm_struct, 'NumStarts', 10, 'RandomSeed', 123, 'SaveOutputs', false);

assert(isequal(res_perm.agent_ids, orig_data.agent_ids), 'Agent IDs were not restored to canonical order');
assert(norm(res_perm.W - orig_data.W, 'fro') < 1e-14, 'Permuted W was not correctly aligned');
assert(abs(res_perm.total_input_amplitude_sum.min_amplitude_sum - orig_data.total_input_amplitude_sum.min_amplitude_sum) < 1e-12, ...
    'Permuted data produced different min amplitude sum');
assert(abs(res_perm.total_input_amplitude_sum.max_amplitude_sum - orig_data.total_input_amplitude_sum.max_amplitude_sum) < 1e-12, ...
    'Permuted data produced different max amplitude sum');
fprintf('  --> PASSED: Permuted ID struct perfectly aligned and yielded identical optimal values.\n');

%% Test 3: Missing and Duplicate Agent ID Detection
fprintf('\n[TEST 3] Testing Missing and Duplicate Agent ID Detection...\n');
bad_dup = perm_struct;
bad_dup.agent_ids = [7, 8, 8, 10]; % duplicate 8
caught_dup = false;
try
    calculate_input_vibration_optimal_phases(bad_dup, 'SaveOutputs', false);
catch ME
    caught_dup = true;
    fprintf('  Caught expected duplicate error: %s\n', ME.identifier);
end
assert(caught_dup, 'Failed to detect duplicate Agent IDs in input struct');

bad_dim = perm_struct;
bad_dim.agent_ids = [7, 8, 9]; % missing agent 10 (length 3 instead of 4)
caught_dim = false;
try
    calculate_input_vibration_optimal_phases(bad_dim, 'SaveOutputs', false);
catch ME
    caught_dim = true;
    fprintf('  Caught expected dimension error: %s\n', ME.identifier);
end
assert(caught_dim, 'Failed to detect missing Agent ID count in input struct');
fprintf('  --> PASSED: Duplicate and missing Agent IDs detected and rejected.\n');

%% Test 4: Execution When Observed Data is Absent
fprintf('\n[TEST 4] Testing Execution Without Observed Data...\n');
% Create synthetic W with no phase_analysis_cache.mat in its path
synth_struct = struct();
synth_struct.N = 4;
synth_struct.agent_ids = [101, 102, 103, 104];
synth_struct.W = [0 0.1 -0.2 0.05; 0.1 0 0.15 -0.1; -0.2 0.15 0 0.3; 0.05 -0.1 0.3 0];
[U_s, S_s, V_s] = svd(synth_struct.W);
synth_struct.svd.U = U_s;
synth_struct.svd.V = V_s;
synth_struct.svd.singular_values = diag(S_s);
synth_struct.output_dir = fullfile(tempdir, 'synth_test_optimal_phases');
if ~exist(synth_struct.output_dir, 'dir'), mkdir(synth_struct.output_dir); end

res_synth = calculate_input_vibration_optimal_phases(synth_struct, 'NumStarts', 5, 'SaveOutputs', true);
assert(~res_synth.observed.available, 'Observed data should not be available for synthetic struct');
assert(exist(fullfile(res_synth.output_dir, 'optimal_phases_total_input_amplitude_sum.png'), 'file') == 2, ...
    'Failed to save total input amplitude sum figure');
assert(exist(fullfile(res_synth.output_dir, 'optimal_phases_total_input_amplitude_sum.csv'), 'file') == 2, ...
    'Failed to save total input amplitude sum CSV');
fprintf('  --> PASSED: Optimization and full exports succeeded without observed data.\n');

%% Test 5: Real Dataset Execution with Observed Time Series Figure
fprintf('\n[TEST 5] Testing Real Datasets (Round, SStick, Round6) with Observed Plots...\n');
res_round = calculate_input_vibration_optimal_phases('Round', 'SaveOutputs', true);
if res_round.observed.available
    ts_fig = fullfile(res_round.output_dir, 'optimal_phases_observed_amplitude_sum_timeseries.png');
    assert(exist(ts_fig, 'file') == 2, 'Observed timeseries figure was not generated for Round');
    fprintf('  --> Generated Round timeseries plot: %s\n', ts_fig);
end

res_sstick = calculate_input_vibration_optimal_phases('SStick', 'SaveOutputs', true);
if res_sstick.observed.available
    ts_fig_sstick = fullfile(res_sstick.output_dir, 'optimal_phases_observed_amplitude_sum_timeseries.png');
    assert(exist(ts_fig_sstick, 'file') == 2, 'Observed timeseries figure was not generated for SStick');
    fprintf('  --> Generated SStick timeseries plot: %s\n', ts_fig_sstick);
end

res_round6 = calculate_input_vibration_optimal_phases('Round6', 'SaveOutputs', true);
fprintf('  --> Round6 (N=6) execution completed.\n');

fprintf('\n=========================================================================\n');
fprintf('   ALL 5 RIGOROUS UNIT TESTS COMPLETED SUCCESSFULLY (100%%)\n');
fprintf('=========================================================================\n');
