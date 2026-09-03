function [psi_grid, gamma_values, meta] = reconstruct_exported_gamma(gamma_source, psi_grid, component)
% RECONSTRUCT_EXPORTED_GAMMA Re-evaluate exported PRC / Gamma data.
%
% Usage:
%   % Run with no arguments (uses default snippet in gamma_exports):
%   [psi, z, meta] = reconstruct_exported_gamma();
%
%   % From PRC snippet txt file:
%   [psi, z, meta] = reconstruct_exported_gamma('gamma_exports/prc_snippet_target_optimal_z_psid_0.00pi.txt');
%   [psi, z, meta] = reconstruct_exported_gamma('gamma_exports/prc_snippet_ref_w1.txt');
%
%   % From MAT export:
%   S = load('gamma_exports/gamma_export_latest.mat');
%   [psi, gamma_true] = reconstruct_exported_gamma(S.gamma_export.true_gamma);
%   [psi, gamma_full] = reconstruct_exported_gamma(S.gamma_export.agents(1).derived_gamma, [], 'full');
%
%   % From CSV export:
%   [psi, gamma] = reconstruct_exported_gamma('gamma_exports/gamma_curve_agent2_derived.csv');
%
% Inputs:
%   gamma_source : Optional. Exported gamma struct, or file path string/char (.txt, .mat, .csv).
%                  If omitted or empty, searches for a default snippet file in gamma_exports.
%   psi_grid     : Optional. Column vector of psi values. If omitted, the
%                  stored grid or default linspace(-pi, pi, 512) is used.
%   component    : Optional. Component selector.
%                  For resonant gamma: 'selected', 'full', 'symmetric', 'antisymmetric'.
%                  For true gamma: 'true', 'agent1', 'agent2'.
%
% Outputs:
%   psi_grid     : Evaluation grid vector.
%   gamma_values : Evaluated PRC / Gamma values on psi_grid.
%   meta         : Metadata struct describing the resolved source.

    if nargin < 1 || isempty(gamma_source)
        gamma_source = resolve_default_gamma_source();
    end
    if nargin < 2
        psi_grid = [];
    end
    if nargin < 3
        component = '';
    end

    if nargin == 2 && (ischar(psi_grid) || (isstring(psi_grid) && isscalar(psi_grid)))
        component = psi_grid;
        psi_grid = [];
    end

    if isstring(component)
        component = char(component);
    end
    if isstring(psi_grid) && isscalar(psi_grid)
        psi_grid = char(psi_grid);
    end

    % Check if input is a file path
    if ischar(gamma_source) || (isstring(gamma_source) && isscalar(gamma_source))
        file_path = char(gamma_source);
        if ~isfile(file_path)
            resolved = resolve_file_path(file_path);
            if isempty(resolved)
                error('reconstruct_exported_gamma:FileNotFound', 'File not found: %s', file_path);
            end
            file_path = resolved;
        end

        [~, ~, ext] = fileparts(char(file_path));
        switch lower(ext)
            case '.txt'
                [psi_grid, gamma_values, meta] = reconstruct_txt_snippet(file_path, psi_grid);
                return;
            case '.mat'
                mat_data = load(file_path);
                if isfield(mat_data, 'gamma_export')
                    gamma_source = mat_data.gamma_export;
                else
                    gamma_source = mat_data;
                end
            case '.csv'
                [psi_grid, gamma_values, meta] = reconstruct_csv_file(file_path, psi_grid, component);
                return;
            otherwise
                error('reconstruct_exported_gamma:UnsupportedExt', 'Unsupported file extension: %s', ext);
        end
    end

    gamma_definition = unwrap_exported_gamma_source(gamma_source);
    if ~isstruct(gamma_definition)
        error('reconstruct_exported_gamma:InvalidSource', 'gamma_source must be a struct or file path.');
    end

    if isfield(gamma_definition, 'prc_a') || isfield(gamma_definition, 'prc_b')
        [psi_grid, gamma_values, meta] = reconstruct_snippet_struct(gamma_definition, psi_grid);
        return;
    end

    if isfield(gamma_definition, 'gamma_true') && isfield(gamma_definition, 'psi_grid')
        [psi_grid, gamma_values, meta] = reconstruct_sampled_true_gamma(gamma_definition, psi_grid, component);
        return;
    end

    if isfield(gamma_definition, 'harmonic_index') && isfield(gamma_definition, 'gamma_cos') && isfield(gamma_definition, 'gamma_sin')
        [psi_grid, gamma_values, meta] = reconstruct_harmonic_gamma(gamma_definition, psi_grid, component);
        return;
    end

    if isfield(gamma_definition, 'psi_grid_centered') || isfield(gamma_definition, 'psi_grid')
        [psi_grid, gamma_values, meta] = reconstruct_sampled_gamma(gamma_definition, psi_grid, component);
        return;
    end

    error('reconstruct_exported_gamma:UnsupportedStruct', 'Unsupported gamma_source structure.');
end

% =========================================================================
% Helpers
% =========================================================================

function default_source = resolve_default_gamma_source()
    candidates = {
        fullfile('gamma_exports', 'prc_snippet_target_optimal_z_psid_0.00pi.txt'), ...
        fullfile('gamma_exports', 'prc_snippet_target_optimal_z.txt'), ...
        fullfile('gamma_exports', 'prc_snippet_ref_w1.txt'), ...
        fullfile('gamma_exports', 'gamma_export_latest.mat')
    };

    for i = 1:numel(candidates)
        if isfile(candidates{i})
            default_source = candidates{i};
            return;
        end
    end

    script_dir = fileparts(char(mfilename('fullpath')));
    for i = 1:numel(candidates)
        cand = fullfile(script_dir, candidates{i});
        if isfile(cand)
            default_source = cand;
            return;
        end
        cand = fullfile(script_dir, '..', candidates{i});
        if isfile(cand)
            default_source = cand;
            return;
        end
    end

    exp_dir = fullfile(pwd, 'gamma_exports');
    if ~isfolder(exp_dir)
        exp_dir = fullfile(script_dir, 'gamma_exports');
    end
    if isfolder(exp_dir)
        files = dir(fullfile(exp_dir, 'prc_snippet_*.txt'));
        if ~isempty(files)
            default_source = fullfile(files(1).folder, files(1).name);
            return;
        end
    end

    error('reconstruct_exported_gamma:NoDefaultSource', ...
        'No gamma_source provided and no default exported files found in gamma_exports.');
end

function resolved_path = resolve_file_path(file_path)
    file_path = char(file_path);
    resolved_path = '';
    if isfile(file_path)
        resolved_path = file_path;
        return;
    end

    cand = fullfile(pwd, file_path);
    if isfile(cand)
        resolved_path = cand;
        return;
    end

    script_dir = fileparts(char(mfilename('fullpath')));
    cand = fullfile(script_dir, file_path);
    if isfile(cand)
        resolved_path = cand;
        return;
    end

    cand = fullfile(script_dir, '..', file_path);
    if isfile(cand)
        resolved_path = cand;
        return;
    end

    cand = fullfile(script_dir, 'gamma_exports', file_path);
    if isfile(cand)
        resolved_path = cand;
        return;
    end
    cand = fullfile(script_dir, '..', 'gamma_exports', file_path);
    if isfile(cand)
        resolved_path = cand;
        return;
    end
end

function [psi_grid, gamma_values, meta] = reconstruct_txt_snippet(file_path, psi_grid)
    file_path = char(file_path);
    snippet = parse_prc_snippet_file(file_path);
    if isempty(psi_grid)
        psi_grid = linspace(-pi, pi, 512).';
    else
        validateattributes(psi_grid, {'numeric'}, {'vector', 'finite'}, mfilename, 'psi_grid');
        psi_grid = psi_grid(:);
    end

    prc_a = snippet.prc_a(:);
    prc_b = snippet.prc_b(:);
    n_h = snippet.prc_harmonics;

    gamma_values = prc_a(1) * ones(size(psi_grid));
    for n = 1:n_h
        if n + 1 <= numel(prc_a)
            gamma_values = gamma_values + prc_a(n + 1) * cos(n * psi_grid);
        end
        if n + 1 <= numel(prc_b)
            gamma_values = gamma_values + prc_b(n + 1) * sin(n * psi_grid);
        end
    end

    meta = struct();
    meta.source_type = 'prc_snippet_txt';
    meta.file_path = file_path;
    meta.label = snippet.label_text;
    meta.prc_harmonics = n_h;
    meta.prc_a = prc_a;
    meta.prc_b = prc_b;
    meta.target_psi_d_rad = snippet.target_psi_d_rad;
    meta.target_psi_d_over_pi = snippet.target_psi_d_over_pi;
    meta.target_psi_phys_d_rad = snippet.target_psi_phys_d_rad;
    meta.target_psi_phys_d_over_pi = snippet.target_psi_phys_d_over_pi;
    meta.available = true;
end

function snippet = parse_prc_snippet_file(file_path)
    file_path = char(file_path);
    raw_text = fileread(file_path);
    lines_txt = splitlines(string(raw_text));

    prc_harmonics = [];
    label_text = '';
    target_psi_d_rad = [];
    target_psi_d_over_pi = [];
    target_psi_phys_d_rad = [];
    target_psi_phys_d_over_pi = [];

    for i = 1:numel(lines_txt)
        line = char(lines_txt(i));

        tok_label = regexp(line, '^\s*#\s*Auto-generated from\s*(.*)\s*$', 'tokens', 'once');
        if ~isempty(tok_label)
            label_text = strtrim(tok_label{1});
        end

        tok_h = regexp(line, '^\s*prc_harmonics\s*=\s*(\d+)\s*$', 'tokens', 'once');
        if ~isempty(tok_h)
            prc_harmonics = str2double(tok_h{1});
        end

        tok_psid = regexp(line, '^\s*target_psi_d_rad\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$', 'tokens', 'once');
        if ~isempty(tok_psid)
            target_psi_d_rad = str2double(tok_psid{1});
        end

        tok_psid_pi = regexp(line, '^\s*target_psi_d_over_pi\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$', 'tokens', 'once');
        if ~isempty(tok_psid_pi)
            target_psi_d_over_pi = str2double(tok_psid_pi{1});
        end

        tok_phys = regexp(line, '^\s*target_psi_phys_d_rad\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$', 'tokens', 'once');
        if ~isempty(tok_phys)
            target_psi_phys_d_rad = str2double(tok_phys{1});
        end

        tok_phys_pi = regexp(line, '^\s*target_psi_phys_d_over_pi\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$', 'tokens', 'once');
        if ~isempty(tok_phys_pi)
            target_psi_phys_d_over_pi = str2double(tok_phys_pi{1});
        end
    end

    if isempty(label_text)
        [~, name, ext] = fileparts(char(file_path));
        label_text = [name ext];
    end

    if isempty(prc_harmonics) || ~isfinite(prc_harmonics) || prc_harmonics < 0
        prc_harmonics = 10;
    else
        prc_harmonics = floor(prc_harmonics);
    end

    prc_a = zeros(prc_harmonics + 1, 1);
    prc_b = zeros(prc_harmonics + 1, 1);

    num_pattern = '([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)';
    pat_a = ['^\s*prc_a\[(\d+)\]\s*=\s*' num_pattern '\s*$'];
    pat_b = ['^\s*prc_b\[(\d+)\]\s*=\s*' num_pattern '\s*$'];

    for i = 1:numel(lines_txt)
        line = char(lines_txt(i));

        tok_a = regexp(line, pat_a, 'tokens', 'once');
        if ~isempty(tok_a)
            idx = str2double(tok_a{1});
            val = str2double(tok_a{2});
            if isfinite(idx) && isfinite(val) && idx >= 0 && idx <= prc_harmonics
                prc_a(idx + 1) = val;
            end
            continue;
        end

        tok_b = regexp(line, pat_b, 'tokens', 'once');
        if ~isempty(tok_b)
            idx = str2double(tok_b{1});
            val = str2double(tok_b{2});
            if isfinite(idx) && isfinite(val) && idx >= 0 && idx <= prc_harmonics
                prc_b(idx + 1) = val;
            end
        end
    end

    snippet = struct();
    snippet.prc_harmonics = prc_harmonics;
    snippet.prc_a = prc_a;
    snippet.prc_b = prc_b;
    snippet.label_text = label_text;
    snippet.target_psi_d_rad = target_psi_d_rad;
    snippet.target_psi_d_over_pi = target_psi_d_over_pi;
    snippet.target_psi_phys_d_rad = target_psi_phys_d_rad;
    snippet.target_psi_phys_d_over_pi = target_psi_phys_d_over_pi;
end

function [psi_grid, gamma_values, meta] = reconstruct_snippet_struct(snippet, psi_grid)
    if isempty(psi_grid)
        psi_grid = linspace(-pi, pi, 512).';
    else
        validateattributes(psi_grid, {'numeric'}, {'vector', 'finite'}, mfilename, 'psi_grid');
        psi_grid = psi_grid(:);
    end

    prc_a = get_optional_field(snippet, 'prc_a', []);
    prc_b = get_optional_field(snippet, 'prc_b', []);
    n_h = get_optional_field(snippet, 'prc_harmonics', max(numel(prc_a), numel(prc_b)) - 1);

    if isempty(prc_a)
        prc_a = zeros(n_h + 1, 1);
    end
    if isempty(prc_b)
        prc_b = zeros(n_h + 1, 1);
    end

    gamma_values = prc_a(1) * ones(size(psi_grid));
    for n = 1:n_h
        if n + 1 <= numel(prc_a)
            gamma_values = gamma_values + prc_a(n + 1) * cos(n * psi_grid);
        end
        if n + 1 <= numel(prc_b)
            gamma_values = gamma_values + prc_b(n + 1) * sin(n * psi_grid);
        end
    end

    meta = struct();
    meta.source_type = 'prc_snippet_struct';
    meta.label = get_optional_field(snippet, 'label_text', 'Snippet Struct');
    meta.prc_harmonics = n_h;
    meta.prc_a = prc_a;
    meta.prc_b = prc_b;
    meta.target_psi_d_rad = get_optional_field(snippet, 'target_psi_d_rad', []);
    meta.target_psi_d_over_pi = get_optional_field(snippet, 'target_psi_d_over_pi', []);
    meta.target_psi_phys_d_rad = get_optional_field(snippet, 'target_psi_phys_d_rad', []);
    meta.target_psi_phys_d_over_pi = get_optional_field(snippet, 'target_psi_phys_d_over_pi', []);
    meta.available = true;
end

function [psi_grid, gamma_values, meta] = reconstruct_csv_file(file_path, psi_grid, component)
    file_path = char(file_path);
    opts = detectImportOptions(file_path);
    t = readtable(file_path, opts);
    col_names = lower(t.Properties.VariableNames);

    if ismember('psi', col_names)
        csv_psi = t.psi;
    else
        error('reconstruct_exported_gamma:CsvNoPsi', 'CSV file must contain a "psi" column.');
    end

    if isempty(component)
        component = 'selected';
    else
        component = lower(strtrim(char(component)));
    end

    target_col = '';
    switch component
        case {'selected', 'full'}
            if ismember('gamma_selected', col_names)
                target_col = 'gamma_selected';
            elseif ismember('gamma_full', col_names)
                target_col = 'gamma_full';
            elseif ismember('gamma', col_names)
                target_col = 'gamma';
            elseif ismember('gamma_true', col_names)
                target_col = 'gamma_true';
            end
        case 'symmetric'
            if ismember('gamma_symmetric', col_names)
                target_col = 'gamma_symmetric';
            end
        case 'antisymmetric'
            if ismember('gamma_antisymmetric', col_names)
                target_col = 'gamma_antisymmetric';
            end
    end

    if isempty(target_col)
        target_col = t.Properties.VariableNames{2};
    end

    csv_vals = t.(target_col);

    if isempty(psi_grid)
        psi_grid = csv_psi(:);
        gamma_values = csv_vals(:);
    else
        psi_grid = psi_grid(:);
        gamma_values = interp1(csv_psi(:), csv_vals(:), psi_grid, 'linear', 'extrap');
    end

    [~, name, ext] = fileparts(char(file_path));
    meta = struct();
    meta.source_type = 'csv';
    meta.file_path = file_path;
    meta.label = [name ext];
    meta.component = target_col;
    meta.available = true;
end

function gamma_definition = unwrap_exported_gamma_source(gamma_source)
    gamma_definition = gamma_source;
    if isstruct(gamma_source) && isfield(gamma_source, 'gamma_resonance')
        gamma_definition = gamma_source.gamma_resonance;
    end
end

function [psi_grid, gamma_values, meta] = reconstruct_harmonic_gamma(gamma_definition, psi_grid, component)
    harmonic_index = gamma_definition.harmonic_index(:);
    gamma_cos = gamma_definition.gamma_cos(:);
    gamma_sin = gamma_definition.gamma_sin(:);
    selected_component = get_selected_component_name(gamma_definition);
    resolved_component = normalize_resonant_component(component, selected_component);

    if isempty(psi_grid)
        psi_grid = get_default_psi_grid(gamma_definition, 512);
    else
        validateattributes(psi_grid, {'numeric'}, {'vector', 'finite'}, mfilename, 'psi_grid');
        psi_grid = psi_grid(:);
    end

    gamma_symmetric = zeros(size(psi_grid));
    gamma_antisymmetric = zeros(size(psi_grid));
    for idx = 1:numel(harmonic_index)
        k = harmonic_index(idx);
        gamma_symmetric = gamma_symmetric + gamma_cos(idx) * cos(k * psi_grid);
        gamma_antisymmetric = gamma_antisymmetric + gamma_sin(idx) * sin(k * psi_grid);
    end
    gamma_full = gamma_symmetric + gamma_antisymmetric;

    switch resolved_component
        case 'full'
            gamma_values = gamma_full;
        case 'symmetric'
            gamma_values = gamma_symmetric;
        case 'antisymmetric'
            gamma_values = gamma_antisymmetric;
        otherwise
            error('Unsupported resonant component: %s', resolved_component);
    end

    meta = struct();
    meta.source_type = 'harmonic';
    meta.component = resolved_component;
    meta.selected_component = selected_component;
    meta.psi_label = get_optional_field(gamma_definition, 'psi_label', '');
    meta.harmonic_index = harmonic_index;
    meta.gamma_cos = gamma_cos;
    meta.gamma_sin = gamma_sin;
    meta.available = get_optional_field(gamma_definition, 'enabled', true);
end

function [psi_grid, gamma_values, meta] = reconstruct_sampled_gamma(gamma_definition, psi_grid, component)
    selected_component = get_selected_component_name(gamma_definition);
    resolved_component = normalize_resonant_component(component, selected_component);
    sample_psi = get_default_psi_grid(gamma_definition, 512);
    sample_values = get_sampled_resonant_values(gamma_definition, resolved_component);

    if isempty(sample_psi) || isempty(sample_values)
        error('The exported gamma structure does not contain sampled values for component %s.', resolved_component);
    end

    if isempty(psi_grid)
        psi_grid = sample_psi(:);
        gamma_values = sample_values(:);
    else
        validateattributes(psi_grid, {'numeric'}, {'vector', 'finite'}, mfilename, 'psi_grid');
        psi_grid = psi_grid(:);
        gamma_values = interp1(sample_psi(:), sample_values(:), psi_grid, 'linear', 'extrap');
    end

    meta = struct();
    meta.source_type = 'sampled-resonant';
    meta.component = resolved_component;
    meta.selected_component = selected_component;
    meta.psi_label = get_optional_field(gamma_definition, 'psi_label', '');
    meta.available = get_optional_field(gamma_definition, 'enabled', true);
end

function [psi_grid, gamma_values, meta] = reconstruct_sampled_true_gamma(gamma_definition, psi_grid, component)
    if isempty(component)
        resolved_component = 'true';
    else
        resolved_component = lower(strtrim(char(component)));
    end

    sample_psi = gamma_definition.psi_grid(:);
    if isempty(sample_psi)
        error('The exported true gamma structure does not contain psi_grid.');
    end

    switch resolved_component
        case {'true', 'selected', 'full'}
            sample_values = gamma_definition.gamma_true(:);
            resolved_component = 'true';
        case 'agent1'
            sample_values = gamma_definition.gamma_agent_1(:);
        case 'agent2'
            sample_values = gamma_definition.gamma_agent_2(:);
        otherwise
            error('Unsupported true gamma component: %s', resolved_component);
    end

    if isempty(psi_grid)
        psi_grid = sample_psi;
        gamma_values = sample_values;
    else
        validateattributes(psi_grid, {'numeric'}, {'vector', 'finite'}, mfilename, 'psi_grid');
        psi_grid = psi_grid(:);
        gamma_values = interp1(sample_psi, sample_values, psi_grid, 'linear', 'extrap');
    end

    meta = struct();
    meta.source_type = 'sampled-true';
    meta.component = resolved_component;
    meta.selected_component = 'true';
    meta.psi_label = '\psi';
    meta.available = get_optional_field(gamma_definition, 'available', true);
    meta.agent_id_1 = get_optional_field(gamma_definition, 'agent_id_1', []);
    meta.agent_id_2 = get_optional_field(gamma_definition, 'agent_id_2', []);
end

function psi_grid = get_default_psi_grid(gamma_definition, default_count)
    if isfield(gamma_definition, 'psi_grid_centered') && ~isempty(gamma_definition.psi_grid_centered)
        psi_grid = gamma_definition.psi_grid_centered(:);
        return;
    end
    if isfield(gamma_definition, 'psi_grid') && ~isempty(gamma_definition.psi_grid)
        psi_grid = gamma_definition.psi_grid(:);
        return;
    end
    psi_grid = linspace(-pi, pi, default_count).';
end

function component_name = get_selected_component_name(gamma_definition)
    component_name = 'full';
    if isfield(gamma_definition, 'component') && ~isempty(gamma_definition.component)
        component_name = lower(strtrim(char(gamma_definition.component)));
    end
end

function resolved_component = normalize_resonant_component(component, selected_component)
    if isempty(component)
        resolved_component = selected_component;
        return;
    end

    resolved_component = lower(strtrim(char(component)));
    if strcmp(resolved_component, 'selected')
        resolved_component = selected_component;
    end
end

function sample_values = get_sampled_resonant_values(gamma_definition, component)
    switch component
        case 'full'
            sample_values = get_optional_field(gamma_definition, 'gamma_values_full_centered', []);
            if isempty(sample_values)
                sample_values = get_optional_field(gamma_definition, 'gamma_values_full', []);
            end
        case 'symmetric'
            sample_values = get_optional_field(gamma_definition, 'gamma_values_symmetric_centered', []);
            if isempty(sample_values)
                sample_values = get_optional_field(gamma_definition, 'gamma_values_symmetric', []);
            end
        case 'antisymmetric'
            sample_values = get_optional_field(gamma_definition, 'gamma_values_antisymmetric_centered', []);
            if isempty(sample_values)
                sample_values = get_optional_field(gamma_definition, 'gamma_values_antisymmetric', []);
            end
        otherwise
            error('Unsupported resonant component: %s', component);
    end

    if isempty(sample_values)
        sample_values = get_optional_field(gamma_definition, 'gamma_values_centered', []);
        if isempty(sample_values)
            sample_values = get_optional_field(gamma_definition, 'gamma_values', []);
        end
    end
end

function value = get_optional_field(source_struct, field_name, default_value)
    value = default_value;
    if isstruct(source_struct) && isfield(source_struct, field_name)
        value = source_struct.(field_name);
    end
end
