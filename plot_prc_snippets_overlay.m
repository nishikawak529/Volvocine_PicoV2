function out = plot_prc_snippets_overlay(input_path, n_samples)
% PLOT_PRC_SNIPPETS_OVERLAY Reconstruct and plot PRCs from snippet txt files.
%
% Usage:
%   out = plot_prc_snippets_overlay()
%   out = plot_prc_snippets_overlay('gamma_exports')
%   out = plot_prc_snippets_overlay('gamma_exports/prc_snippet_target_optimal_z_psid_0.00pi.txt')
%   out = plot_prc_snippets_overlay({'prc_snippet_target_optimal_z_psid_0.00pi.txt', 'prc_snippet_ref_w1.txt'})
%
% Inputs:
%   input_path : Optional. Folder path, file path, or cell array of paths.
%                If omitted or empty, searches for gamma_exports directory.
%   n_samples  : Optional. Number of grid points for psi in [-pi, pi]. Default is 1024.
%
% Output:
%   out : Struct containing evaluation grid, reconstructed PRC matrix,
%         file metadata, labels, and figure handle.

    ensure_local_function_folder_on_path();

    if nargin < 1 || isempty(input_path)
        input_path = resolve_default_snippet_dir();
    end
    if nargin < 2 || isempty(n_samples)
        n_samples = 1024;
    end

    txt_files = collect_snippet_files(input_path);
    if isempty(txt_files)
        error('plot_prc_snippets_overlay:NoFilesFound', ...
            'No snippet txt files found for input_path');
    end

    psi = linspace(-pi, pi, n_samples).';
    n_files = numel(txt_files);

    labels = cell(n_files, 1);
    prc_harmonics = zeros(n_files, 1);
    z_matrix = zeros(n_samples, n_files);
    meta_list = cell(n_files, 1);

    for k = 1:n_files
        file_path = txt_files{k};
        [~, z_val, meta] = reconstruct_exported_gamma(file_path, psi);
        
        labels{k} = meta.label;
        prc_harmonics(k) = meta.prc_harmonics;
        z_matrix(:, k) = z_val;
        meta_list{k} = meta;
    end

    fig = figure('Color', 'w', 'Name', 'Reconstructed PRC snippets');
    ax = axes('Parent', fig);
    hold(ax, 'on');

    plot(ax, psi, zeros(size(psi)), ':', 'LineWidth', 1.0, 'Color', [0.6, 0.6, 0.6], 'DisplayName', '0 line');

    cmap = lines(max(n_files, 3));
    for k = 1:n_files
        plot(ax, psi, z_matrix(:, k), 'LineWidth', 1.8, 'Color', cmap(k, :), 'DisplayName', labels{k});
    end

    % Plot target equilibrium markers if available
    for k = 1:n_files
        meta = meta_list{k};
        if isfield(meta, 'target_psi_d_rad') && ~isempty(meta.target_psi_d_rad) && isfinite(meta.target_psi_d_rad)
            psid = meta.target_psi_d_rad;
            [~, idx_eq] = min(abs(psi - psid));
            z_eq = z_matrix(idx_eq, k);
            plot(ax, psid, z_eq, 'o', 'MarkerSize', 7, 'MarkerFaceColor', cmap(k, :), ...
                'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
            xline(ax, psid, ':', 'Color', cmap(k, :), 'LineWidth', 0.8, 'HandleVisibility', 'off');
        end
    end

    xlabel(ax, '$$\psi$$', 'Interpreter', 'latex');
    ylabel(ax, '$$z(\psi)$$', 'Interpreter', 'latex');
    title(ax, 'Reconstructed PRCs from snippet txt files', 'Interpreter', 'none');

    xlim(ax, [-pi, pi]);
    xticks(ax, [-pi, -pi/2, 0, pi/2, pi]);
    xticklabels(ax, {'$$-\pi$$', '$$-\pi/2$$', '0', '$$\pi/2$$', '$$\pi$$'});
    grid(ax, 'on');
    box(ax, 'on');

    ax.XLabel.Interpreter = 'latex';
    ax.YLabel.Interpreter = 'latex';
    ax.TickLabelInterpreter = 'latex';

    legend(ax, 'Location', 'best', 'Interpreter', 'none');

    out = struct();
    out.files = txt_files;
    out.labels = labels;
    out.prc_harmonics = prc_harmonics;
    out.psi = psi;
    out.z_matrix = z_matrix;
    out.meta = meta_list;
    out.figure = fig;

    fprintf('[INFO] Reconstructed and plotted %d PRC snippet file(s).\n', n_files);
end

% =========================================================================
% Helpers
% =========================================================================

function txt_files = collect_snippet_files(input_path)
    txt_files = {};

    if iscell(input_path) || (isstring(input_path) && numel(input_path) > 1)
        for i = 1:numel(input_path)
            if iscell(input_path)
                item = input_path{i};
            else
                item = char(input_path(i));
            end
            sub_files = collect_snippet_files(item);
            txt_files = [txt_files; sub_files(:)]; %#ok<AGROW>
        end
        return;
    end

    p = char(input_path);
    resolved_p = resolve_input_path(p);
    if isempty(resolved_p)
        return;
    end

    if isfolder(resolved_p)
        files = dir(fullfile(resolved_p, 'prc_snippet_*.txt'));
        if isempty(files)
            files = dir(fullfile(resolved_p, '*.txt'));
        end
        if isempty(files)
            return;
        end

        [~, order] = sort({files.name});
        files = files(order);

        txt_files = cell(numel(files), 1);
        for i = 1:numel(files)
            txt_files{i} = fullfile(files(i).folder, files(i).name);
        end
        return;
    end

    if isfile(resolved_p)
        txt_files = {resolved_p};
    end
end

function resolved_path = resolve_default_snippet_dir()
    candidate_pwd = fullfile(pwd, 'gamma_exports');
    if isfolder(candidate_pwd)
        resolved_path = candidate_pwd;
        return;
    end

    script_dir = fileparts(char(mfilename('fullpath')));
    candidate_script = fullfile(script_dir, 'gamma_exports');
    if isfolder(candidate_script)
        resolved_path = candidate_script;
        return;
    end

    candidate_parent = fullfile(script_dir, '..', 'gamma_exports');
    if isfolder(candidate_parent)
        resolved_path = candidate_parent;
        return;
    end

    resolved_path = pwd;
end

function resolved_path = resolve_input_path(input_path)
    input_path = char(input_path);
    resolved_path = '';
    if isfolder(input_path) || isfile(input_path)
        resolved_path = input_path;
        return;
    end

    cand_pwd = fullfile(pwd, input_path);
    if isfolder(cand_pwd) || isfile(cand_pwd)
        resolved_path = cand_pwd;
        return;
    end

    script_dir = fileparts(char(mfilename('fullpath')));
    cand_script = fullfile(script_dir, input_path);
    if isfolder(cand_script) || isfile(cand_script)
        resolved_path = cand_script;
        return;
    end

    cand_parent = fullfile(script_dir, '..', input_path);
    if isfolder(cand_parent) || isfile(cand_parent)
        resolved_path = cand_parent;
        return;
    end

    cand_exports = fullfile(script_dir, '..', 'gamma_exports', input_path);
    if isfolder(cand_exports) || isfile(cand_exports)
        resolved_path = cand_exports;
        return;
    end
end

function ensure_local_function_folder_on_path()
    local_dir = fileparts(char(mfilename('fullpath')));
    if ~contains(path, local_dir)
        addpath(local_dir);
    end
end
