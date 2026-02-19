% =========================================================================
% REFINE_RESOLUTION_SINC
% =========================================================================
% Post-processing script that refines the resolution measurements from the
% parameter sweep by sinc-interpolating the 1D PSF cuts from the original
% 5 cm grid spacing down to 1 cm, yielding more precise -3 dB and -6 dB
% width estimates without re-running the expensive backprojection.
%
% Method:
%   For each completed trial, this script:
%     1. Loads the complex 3D image volume (Img) and coordinate grids
%     2. Extracts 1D cuts through the peak along x, y, and z
%     3. Sinc-interpolates each cut to a 1 cm fine grid using the
%        Whittaker-Shannon reconstruction formula:
%           f(x') = sum_k f(x_k) * sinc((x' - x_k) / dx)
%     4. Recomputes the -3 dB and -6 dB widths on the refined cuts
%     5. Stores both original (5 cm) and refined (1 cm) resolution values
%
% Outputs:
%   sweep_results/sinc_refined/resolution_comparison.csv  - Full table
%   sweep_results/sinc_refined/resolution_comparison.mat  - Same as .mat
%   sweep_results/sinc_refined/heatmap_*.png              - n2 vs n3 heatmaps
%   Per-trial sinc cut overlay plots in each trial's plots/ subfolder
%
% The heatmap figures match the style of the existing sweep heatmaps,
% enabling direct visual comparison between original and refined metrics.
% =========================================================================

clc; clear; close all;

%% ======================== PARAMETER GRID =================================
% Must match the values used in run_parameter_sweep.m
n2_values     = linspace(3.2, 4.0, 9);
n3_values     = linspace(3.2, 4.0, 9);
depth1_values = 0.20:0.10:1.60;

baseOutputDir = fullfile(pwd, 'sweep_results');
refinedDir    = fullfile(baseOutputDir, 'sinc_refined');
if ~exist(refinedDir, 'dir'), mkdir(refinedDir); end

%% ======================== BUILD TRIAL LIST ===============================
[N2_grid, N3_grid, D1_grid] = ndgrid(n2_values, n3_values, depth1_values);
n2_list     = N2_grid(:);
n3_list     = N3_grid(:);
depth1_list = D1_grid(:);
nTotal      = numel(n2_list);

% Thresholds (same as analyze_bp_quality_fn.m)
thresh_3dB = 10^(-3/20);   % ~0.7079
thresh_6dB = 10^(-6/20);   % ~0.5012

% Target fine-grid spacing (1 cm)
fine_step = 0.01;

%% ======================== PRE-ALLOCATE ===================================
% Original resolution (from the 5 cm grid)
orig_res3_x = NaN(nTotal, 1);   orig_res3_y = NaN(nTotal, 1);   orig_res3_z = NaN(nTotal, 1);
orig_res6_x = NaN(nTotal, 1);   orig_res6_y = NaN(nTotal, 1);   orig_res6_z = NaN(nTotal, 1);

% Refined resolution (sinc-interpolated to 1 cm)
ref_res3_x = NaN(nTotal, 1);    ref_res3_y = NaN(nTotal, 1);    ref_res3_z = NaN(nTotal, 1);
ref_res6_x = NaN(nTotal, 1);    ref_res6_y = NaN(nTotal, 1);    ref_res6_z = NaN(nTotal, 1);

% Refined peak location (from sinc-interpolated cuts)
ref_peakX = NaN(nTotal, 1);     ref_peakY = NaN(nTotal, 1);     ref_peakZ = NaN(nTotal, 1);

% Refined resolution volume and equivalent diameter
ref_res3_vol  = NaN(nTotal, 1);
ref_res3_diam = NaN(nTotal, 1);

status_list = repmat({'not_found'}, nTotal, 1);

%% ======================== LOAD PREVIOUS RESULTS (incremental) ============
% If a previous run already produced results, load them so we only process
% new trials. This makes re-runs fast: only newly completed sweep trials
% are sinc-interpolated; everything else is carried forward.
prevMatFile = fullfile(refinedDir, 'resolution_comparison.mat');
if exist(prevMatFile, 'file')
    fprintf('Loading previous results from %s ...\n', prevMatFile);
    prev = load(prevMatFile, 'T_ok');
    T_prev = prev.T_ok;
    for p = 1:height(T_prev)
        idx = find(abs(n2_list - T_prev.n2(p)) < 1e-10 & ...
                   abs(n3_list - T_prev.n3(p)) < 1e-10 & ...
                   abs(depth1_list - T_prev.depth1_m(p)) < 1e-10, 1);
        if ~isempty(idx)
            orig_res3_x(idx) = T_prev.Orig_Res3dB_X(p);
            orig_res3_y(idx) = T_prev.Orig_Res3dB_Y(p);
            orig_res3_z(idx) = T_prev.Orig_Res3dB_Z(p);
            orig_res6_x(idx) = T_prev.Orig_Res6dB_X(p);
            orig_res6_y(idx) = T_prev.Orig_Res6dB_Y(p);
            orig_res6_z(idx) = T_prev.Orig_Res6dB_Z(p);
            ref_res3_x(idx)  = T_prev.Sinc_Res3dB_X(p);
            ref_res3_y(idx)  = T_prev.Sinc_Res3dB_Y(p);
            ref_res3_z(idx)  = T_prev.Sinc_Res3dB_Z(p);
            ref_res6_x(idx)  = T_prev.Sinc_Res6dB_X(p);
            ref_res6_y(idx)  = T_prev.Sinc_Res6dB_Y(p);
            ref_res6_z(idx)  = T_prev.Sinc_Res6dB_Z(p);
            ref_peakX(idx)   = T_prev.Sinc_PeakX(p);
            ref_peakY(idx)   = T_prev.Sinc_PeakY(p);
            ref_peakZ(idx)   = T_prev.Sinc_PeakZ(p);
            ref_res3_vol(idx)  = T_prev.Sinc_Res3dB_Vol(p);
            ref_res3_diam(idx) = T_prev.Sinc_Res3dB_EquivDiam(p);
            status_list{idx} = char(T_prev.Status(p));
        end
    end
    nPrev = sum(~strcmp(status_list, 'not_found'));
    fprintf('  Loaded %d previously processed trials.\n', nPrev);
else
    nPrev = 0;
end

%% ======================== MAIN LOOP ======================================
fprintf('============================================================\n');
fprintf('   SINC-INTERPOLATED RESOLUTION REFINEMENT (5 cm -> 1 cm)\n');
fprintf('============================================================\n');
fprintf('Trials to process: %d  (previously done: %d)\n\n', nTotal, nPrev);

tic;
nNewlyProcessed = 0;

for k = 1:nTotal
    n2_val     = n2_list(k);
    n3_val     = n3_list(k);
    depth1_val = depth1_list(k);

    % --- Skip if already processed from a previous run ---
    if strcmp(status_list{k}, 'success') || strcmp(status_list{k}, 'copied')
        continue;
    end

    % --- n2 == n3 deduplication: use the reference folder (first depth1) ---
    if abs(n2_val - n3_val) < 1e-10 && abs(depth1_val - depth1_values(1)) > 1e-10
        refFolder = sprintf('n2_%.2f_n3_%.2f_d1_%.2f', n2_val, n3_val, depth1_values(1));
        refIdx = find(abs(n2_list - n2_val) < 1e-10 & ...
                      abs(n3_list - n3_val) < 1e-10 & ...
                      abs(depth1_list - depth1_values(1)) < 1e-10, 1);

        if ~isempty(refIdx) && strcmp(status_list{refIdx}, 'success')
            % Copy results from reference trial
            orig_res3_x(k) = orig_res3_x(refIdx);  orig_res3_y(k) = orig_res3_y(refIdx);  orig_res3_z(k) = orig_res3_z(refIdx);
            orig_res6_x(k) = orig_res6_x(refIdx);  orig_res6_y(k) = orig_res6_y(refIdx);  orig_res6_z(k) = orig_res6_z(refIdx);
            ref_res3_x(k)  = ref_res3_x(refIdx);   ref_res3_y(k)  = ref_res3_y(refIdx);   ref_res3_z(k)  = ref_res3_z(refIdx);
            ref_res6_x(k)  = ref_res6_x(refIdx);   ref_res6_y(k)  = ref_res6_y(refIdx);   ref_res6_z(k)  = ref_res6_z(refIdx);
            ref_peakX(k)   = ref_peakX(refIdx);     ref_peakY(k)   = ref_peakY(refIdx);    ref_peakZ(k)   = ref_peakZ(refIdx);
            ref_res3_vol(k)  = ref_res3_vol(refIdx);
            ref_res3_diam(k) = ref_res3_diam(refIdx);
            status_list{k} = 'copied';
            nNewlyProcessed = nNewlyProcessed + 1;
            continue;
        end
    end

    % --- Locate trial folder and result file ---
    folderName = sprintf('n2_%.2f_n3_%.2f_d1_%.2f', n2_val, n3_val, depth1_val);
    matFile    = fullfile(baseOutputDir, folderName, 'mat', 'bp_hybrid_results.mat');

    if ~exist(matFile, 'file')
        continue;   % Trial not yet computed
    end

    try
        % ---- Load data ----
        data  = load(matFile, 'Img', 'xGrid', 'yGrid', 'zGrid');
        Img   = data.Img;
        xGrid = double(data.xGrid(:).');
        yGrid = double(data.yGrid(:).');
        zGrid = double(data.zGrid(:).');

        M = abs(Img);
        [Nx, Ny, Nz] = size(M);
        peakVal = max(M(:));
        M_norm  = M / peakVal;

        % Original grid spacing
        dx = xGrid(2) - xGrid(1);
        dy = yGrid(2) - yGrid(1);
        dz = zGrid(2) - zGrid(1);

        % ---- Find peak ----
        [~, peakIdx] = max(M(:));
        [ix_pk, iy_pk, iz_pk] = ind2sub([Nx, Ny, Nz], peakIdx);

        % ---- Extract 1D cuts through the peak ----
        cut_x = double(squeeze(M_norm(:, iy_pk, iz_pk)));
        cut_y = double(squeeze(M_norm(ix_pk, :, iz_pk)));
        cut_z = double(squeeze(M_norm(ix_pk, iy_pk, :)));
        cut_y = cut_y(:);
        cut_z = cut_z(:);

        % ---- Original resolution (5 cm grid, linear interp) ----
        orig_res3_x(k) = computeResolution_local(xGrid, cut_x, thresh_3dB);
        orig_res3_y(k) = computeResolution_local(yGrid, cut_y, thresh_3dB);
        orig_res3_z(k) = computeResolution_local(zGrid, cut_z, thresh_3dB);
        orig_res6_x(k) = computeResolution_local(xGrid, cut_x, thresh_6dB);
        orig_res6_y(k) = computeResolution_local(yGrid, cut_y, thresh_6dB);
        orig_res6_z(k) = computeResolution_local(zGrid, cut_z, thresh_6dB);

        % ---- Build fine grids (1 cm spacing) ----
        xFine = (xGrid(1):fine_step:xGrid(end)).';
        yFine = (yGrid(1):fine_step:yGrid(end)).';
        zFine = (zGrid(1):fine_step:zGrid(end)).';

        % ---- Sinc interpolation ----
        cut_x_fine = sinc_interp(xGrid, cut_x, xFine, dx);
        cut_y_fine = sinc_interp(yGrid, cut_y, yFine, dy);
        cut_z_fine = sinc_interp(zGrid, cut_z, zFine, dz);

        % ---- Refined resolution (1 cm grid) ----
        ref_res3_x(k) = computeResolution_local(xFine, cut_x_fine, thresh_3dB);
        ref_res3_y(k) = computeResolution_local(yFine, cut_y_fine, thresh_3dB);
        ref_res3_z(k) = computeResolution_local(zFine, cut_z_fine, thresh_3dB);
        ref_res6_x(k) = computeResolution_local(xFine, cut_x_fine, thresh_6dB);
        ref_res6_y(k) = computeResolution_local(yFine, cut_y_fine, thresh_6dB);
        ref_res6_z(k) = computeResolution_local(zFine, cut_z_fine, thresh_6dB);

        % ---- Refined peak location (from sinc maxima) ----
        [~, ix_fine] = max(cut_x_fine);
        [~, iy_fine] = max(cut_y_fine);
        [~, iz_fine] = max(cut_z_fine);
        ref_peakX(k) = xFine(ix_fine);
        ref_peakY(k) = yFine(iy_fine);
        ref_peakZ(k) = zFine(iz_fine);

        % ---- Refined volume and equivalent diameter ----
        ref_res3_vol(k)  = ref_res3_x(k) * ref_res3_y(k) * ref_res3_z(k);
        ref_res3_diam(k) = 2 * (3 * ref_res3_vol(k) / (4 * pi))^(1/3);

        status_list{k} = 'success';
        nNewlyProcessed = nNewlyProcessed + 1;

        % ---- Per-trial overlay plot: original vs sinc-interpolated ----
        plotDir = fullfile(baseOutputDir, folderName, 'plots');
        if ~exist(plotDir, 'dir'), mkdir(plotDir); end

        fig = figure('Color', 'w', 'Position', [50 50 1600 500], 'Visible', 'off');

        % X cut
        subplot(1, 3, 1);
        plot(xGrid, cut_x, 'bo-', 'MarkerSize', 4, 'LineWidth', 1, 'DisplayName', 'Original (5 cm)');
        hold on;
        plot(xFine, cut_x_fine, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Sinc interp (1 cm)');
        yline(thresh_3dB, 'k--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
        yline(thresh_6dB, 'k:', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
        % Show resolution markers
        xline(ref_peakX(k) - ref_res3_x(k)/2, 'g-', 'LineWidth', 1.5);
        xline(ref_peakX(k) + ref_res3_x(k)/2, 'g-', 'LineWidth', 1.5);
        hold off;
        xlim([xGrid(1) xGrid(end)]); ylim([0 1.05]);
        xlabel('x [m]'); ylabel('Normalized Amplitude');
        title(sprintf('X Cut  |  Orig: %.4f m  |  Refined: %.4f m', orig_res3_x(k), ref_res3_x(k)));
        legend('Location', 'best', 'FontSize', 7); grid on;

        % Y cut
        subplot(1, 3, 2);
        plot(yGrid, cut_y, 'bo-', 'MarkerSize', 4, 'LineWidth', 1, 'DisplayName', 'Original (5 cm)');
        hold on;
        plot(yFine, cut_y_fine, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Sinc interp (1 cm)');
        yline(thresh_3dB, 'k--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
        yline(thresh_6dB, 'k:', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
        xline(ref_peakY(k) - ref_res3_y(k)/2, 'g-', 'LineWidth', 1.5);
        xline(ref_peakY(k) + ref_res3_y(k)/2, 'g-', 'LineWidth', 1.5);
        hold off;
        xlim([yGrid(1) yGrid(end)]); ylim([0 1.05]);
        xlabel('y [m]'); ylabel('Normalized Amplitude');
        title(sprintf('Y Cut  |  Orig: %.4f m  |  Refined: %.4f m', orig_res3_y(k), ref_res3_y(k)));
        legend('Location', 'best', 'FontSize', 7); grid on;

        % Z cut
        subplot(1, 3, 3);
        plot(zGrid, cut_z, 'bo-', 'MarkerSize', 4, 'LineWidth', 1, 'DisplayName', 'Original (5 cm)');
        hold on;
        plot(zFine, cut_z_fine, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Sinc interp (1 cm)');
        yline(thresh_3dB, 'k--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
        yline(thresh_6dB, 'k:', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
        xline(ref_peakZ(k) - ref_res3_z(k)/2, 'g-', 'LineWidth', 1.5);
        xline(ref_peakZ(k) + ref_res3_z(k)/2, 'g-', 'LineWidth', 1.5);
        hold off;
        xlim([zGrid(1) zGrid(end)]); ylim([0 1.05]);
        xlabel('z [m]'); ylabel('Normalized Amplitude');
        title(sprintf('Z Cut  |  Orig: %.4f m  |  Refined: %.4f m', orig_res3_z(k), ref_res3_z(k)));
        legend('Location', 'best', 'FontSize', 7); grid on;

        sgtitle(sprintf('Sinc Refinement: n_2=%.2f, n_3=%.2f, d_1=%.2f m', ...
            n2_val, n3_val, depth1_val), 'FontSize', 13, 'FontWeight', 'bold');

        saveas(fig, fullfile(plotDir, 'sinc_refined_cuts.png'));
        close(fig);

        if mod(nNewlyProcessed, 10) == 0
            fprintf('  Newly processed %d trials so far (%.0f s elapsed)\n', nNewlyProcessed, toc);
        end

    catch ME
        status_list{k} = sprintf('error: %s', ME.message);
        fprintf('  Trial %d FAILED: %s\n', k, ME.message);
    end
end

elapsed = toc;
nSuccess = sum(strcmp(status_list, 'success') | strcmp(status_list, 'copied'));
fprintf('\nTotal available: %d / %d trials  (newly processed: %d, previously cached: %d)  [%.1f s]\n', ...
    nSuccess, nTotal, nNewlyProcessed, nPrev, elapsed);

%% ======================== BUILD COMPARISON TABLE =========================
T = table( ...
    n2_list, n3_list, depth1_list, ...
    orig_res3_x, orig_res3_y, orig_res3_z, ...
    orig_res6_x, orig_res6_y, orig_res6_z, ...
    ref_res3_x,  ref_res3_y,  ref_res3_z, ...
    ref_res6_x,  ref_res6_y,  ref_res6_z, ...
    ref_res3_vol, ref_res3_diam, ...
    ref_peakX, ref_peakY, ref_peakZ, ...
    status_list, ...
    'VariableNames', { ...
        'n2', 'n3', 'depth1_m', ...
        'Orig_Res3dB_X', 'Orig_Res3dB_Y', 'Orig_Res3dB_Z', ...
        'Orig_Res6dB_X', 'Orig_Res6dB_Y', 'Orig_Res6dB_Z', ...
        'Sinc_Res3dB_X', 'Sinc_Res3dB_Y', 'Sinc_Res3dB_Z', ...
        'Sinc_Res6dB_X', 'Sinc_Res6dB_Y', 'Sinc_Res6dB_Z', ...
        'Sinc_Res3dB_Vol', 'Sinc_Res3dB_EquivDiam', ...
        'Sinc_PeakX', 'Sinc_PeakY', 'Sinc_PeakZ', ...
        'Status'});

% Filter to successful trials
okMask = strcmp(T.Status, 'success') | strcmp(T.Status, 'copied');
T_ok = T(okMask, :);

% Add delta columns (difference between original and refined)
T_ok.Delta_Res3dB_X = T_ok.Orig_Res3dB_X - T_ok.Sinc_Res3dB_X;
T_ok.Delta_Res3dB_Y = T_ok.Orig_Res3dB_Y - T_ok.Sinc_Res3dB_Y;
T_ok.Delta_Res3dB_Z = T_ok.Orig_Res3dB_Z - T_ok.Sinc_Res3dB_Z;
T_ok.Delta_Res6dB_X = T_ok.Orig_Res6dB_X - T_ok.Sinc_Res6dB_X;
T_ok.Delta_Res6dB_Y = T_ok.Orig_Res6dB_Y - T_ok.Sinc_Res6dB_Y;
T_ok.Delta_Res6dB_Z = T_ok.Orig_Res6dB_Z - T_ok.Sinc_Res6dB_Z;

% Save
writetable(T_ok, fullfile(refinedDir, 'resolution_comparison.csv'));
save(fullfile(refinedDir, 'resolution_comparison.mat'), 'T_ok', ...
    'n2_values', 'n3_values', 'depth1_values');
fprintf('\nSaved: %s\n', fullfile(refinedDir, 'resolution_comparison.csv'));
fprintf('Saved: %s\n', fullfile(refinedDir, 'resolution_comparison.mat'));

%% ======================== PRINT SUMMARY TABLE ============================
% Deduplicate n2==n3 cases (keep only first depth1) for summary stats
sameN_mask = abs(T_ok.n2 - T_ok.n3) < 1e-10;
firstDepth = min(T_ok.depth1_m);
redundant  = sameN_mask & (abs(T_ok.depth1_m - firstDepth) > 1e-10);
T_unique   = T_ok(~redundant, :);

fprintf('\n');
fprintf('================================================================\n');
fprintf('   RESOLUTION COMPARISON: ORIGINAL (5 cm) vs SINC-REFINED (1 cm)\n');
fprintf('================================================================\n');
fprintf('%-5s %-5s %-7s | %-8s %-8s %-8s | %-8s %-8s %-8s | %-7s %-7s %-7s\n', ...
    'n2', 'n3', 'depth1', ...
    'Orig_X', 'Orig_Y', 'Orig_Z', ...
    'Sinc_X', 'Sinc_Y', 'Sinc_Z', ...
    'dX_cm', 'dY_cm', 'dZ_cm');
fprintf('%s\n', repmat('-', 1, 105));

nShow = min(40, height(T_unique));
for r = 1:nShow
    fprintf('%-5.2f %-5.2f %-7.2f | %-8.4f %-8.4f %-8.4f | %-8.4f %-8.4f %-8.4f | %-+7.2f %-+7.2f %-+7.2f\n', ...
        T_unique.n2(r), T_unique.n3(r), T_unique.depth1_m(r), ...
        T_unique.Orig_Res3dB_X(r), T_unique.Orig_Res3dB_Y(r), T_unique.Orig_Res3dB_Z(r), ...
        T_unique.Sinc_Res3dB_X(r), T_unique.Sinc_Res3dB_Y(r), T_unique.Sinc_Res3dB_Z(r), ...
        T_unique.Delta_Res3dB_X(r)*100, T_unique.Delta_Res3dB_Y(r)*100, T_unique.Delta_Res3dB_Z(r)*100);
end
if height(T_unique) > nShow
    fprintf('  ... and %d more rows (see CSV file)\n', height(T_unique) - nShow);
end
fprintf('================================================================\n');

% Summary statistics of the delta
fprintf('\n  Delta statistics (Original - Sinc, in cm):\n');
for ax = {'X', 'Y', 'Z'}
    col = sprintf('Delta_Res3dB_%s', ax{1});
    vals = T_unique.(col) * 100;  % convert m to cm
    fprintf('    %s:  mean=%+.2f cm, std=%.2f cm, max=%.2f cm\n', ...
        ax{1}, mean(vals, 'omitnan'), std(vals, 'omitnan'), max(abs(vals), [], 'omitnan'));
end

%% ======================== HEATMAP FIGURES ================================
% Generate n2-vs-n3 heatmaps for refined resolution metrics at each depth1,
% matching the style of existing sweep heatmaps.

hmDefs = {
    'Sinc_Res3dB_X',    'Refined Resolution X -3dB (m)',       'jet'
    'Sinc_Res3dB_Y',    'Refined Resolution Y -3dB (m)',       'jet'
    'Sinc_Res3dB_Z',    'Refined Resolution Z -3dB (m)',       'jet'
    'Sinc_Res3dB_Vol',  'Refined Resolution Volume (m^3)',     'jet'
    'Sinc_Res6dB_X',    'Refined Resolution X -6dB (m)',       'jet'
    'Sinc_Res6dB_Y',    'Refined Resolution Y -6dB (m)',       'jet'
    'Sinc_Res6dB_Z',    'Refined Resolution Z -6dB (m)',       'jet'
    'Delta_Res3dB_X',   '\Delta Res X -3dB (Orig-Sinc) (m)',  'hot'
    'Delta_Res3dB_Y',   '\Delta Res Y -3dB (Orig-Sinc) (m)',  'hot'
    'Delta_Res3dB_Z',   '\Delta Res Z -3dB (Orig-Sinc) (m)',  'hot'
};

fprintf('\nGenerating heatmap figures ...\n');

for di = 1:numel(depth1_values)
    d1 = depth1_values(di);
    dMask = abs(T_ok.depth1_m - d1) < 1e-6;
    subT  = T_ok(dMask, :);
    if isempty(subT), continue; end

    for mi = 1:size(hmDefs, 1)
        colName  = hmDefs{mi, 1};
        colTitle = hmDefs{mi, 2};
        cmap     = hmDefs{mi, 3};

        if ~ismember(colName, subT.Properties.VariableNames), continue; end

        fig = figure('Color', 'w', 'Position', [100 100 650 520], 'Visible', 'off');
        mat = NaN(numel(n3_values), numel(n2_values));
        for row = 1:height(subT)
            ci = find(abs(n2_values - subT.n2(row)) < 1e-6);
            ri = find(abs(n3_values - subT.n3(row)) < 1e-6);
            if ~isempty(ci) && ~isempty(ri)
                mat(ri, ci) = subT.(colName)(row);
            end
        end

        imagesc(n2_values, n3_values, mat);
        axis xy; colorbar; colormap(cmap);
        xlabel('n_2 (soil layer 1)');
        ylabel('n_3 (soil layer 2)');
        title(sprintf('%s | depth_1 = %.2f m', colTitle, d1));

        % Add value annotations on each cell
        for ri = 1:numel(n3_values)
            for ci = 1:numel(n2_values)
                if ~isnan(mat(ri, ci))
                    if abs(mat(ri, ci)) < 0.01
                        txt = sprintf('%.1e', mat(ri, ci));
                    else
                        txt = sprintf('%.3f', mat(ri, ci));
                    end
                    text(n2_values(ci), n3_values(ri), txt, ...
                        'HorizontalAlignment', 'center', 'FontSize', 6, 'Color', 'k');
                end
            end
        end

        fname = sprintf('heatmap_sinc_%s_d1_%.2f', lower(colName), d1);
        saveas(fig, fullfile(refinedDir, [fname '.png']));
        close(fig);
    end
end

fprintf('Heatmaps saved to: %s\n', refinedDir);

%% ======================== DONE ===========================================
fprintf('\n============================================================\n');
fprintf('   SINC REFINEMENT COMPLETE\n');
fprintf('============================================================\n');
fprintf('Successful trials: %d\n', nSuccess);
fprintf('Output directory:  %s\n', refinedDir);
fprintf('Table:  resolution_comparison.csv  (%d rows x %d cols)\n', height(T_ok), width(T_ok));
fprintf('============================================================\n');

%% ========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function f_fine = sinc_interp(x_orig, f_orig, x_fine, dx)
% SINC_INTERP  Whittaker-Shannon sinc interpolation of a 1D signal.
%
% Reconstructs the continuous signal from uniformly-spaced samples using:
%   f(x') = sum_k  f(x_k) * sinc( (x' - x_k) / dx )
%
% This is the theoretically optimal interpolation for bandlimited signals
% (which backprojected SAR images are, given the finite aperture bandwidth).
%
% Inputs:
%   x_orig  - Original sample coordinates [N x 1] or [1 x N]
%   f_orig  - Original sample values [N x 1] or [1 x N]
%   x_fine  - Fine grid coordinates [M x 1] or [1 x M]
%   dx      - Original sample spacing (scalar)
%
% Output:
%   f_fine  - Interpolated values at x_fine [M x 1]

    x_orig = x_orig(:);
    f_orig = f_orig(:);
    x_fine = x_fine(:);

    N = numel(x_orig);
    M = numel(x_fine);

    % Vectorized: build [M x N] matrix of sinc arguments
    % Each row = one fine-grid point, each column = one original sample
    arg = (x_fine - x_orig.') / dx;   % [M x N]
    S   = sinc(arg);                   % sinc(x) = sin(pi*x)/(pi*x), sinc(0)=1

    f_fine = S * f_orig;               % [M x 1] = [M x N] * [N x 1]
end

function width = computeResolution_local(axis_vec, cut, threshold)
% COMPUTERESOLUTION_LOCAL  Measure the width of a 1D profile at a threshold.
%
% Finds the two points where the profile crosses the threshold level on
% either side of the peak, using linear interpolation between adjacent
% samples. Returns the distance between these crossing points.
%
% This is identical to the version in analyze_bp_quality_fn.m.
    cut = cut(:);
    axis_vec = axis_vec(:);
    [~, ipk] = max(cut);

    left_cut  = cut(1:ipk);
    left_axis = axis_vec(1:ipk);
    idx_left  = find(left_cut < threshold, 1, 'last');
    if isempty(idx_left) || idx_left >= length(left_cut)
        x_left = left_axis(1);
    else
        x_left = interp1(left_cut(idx_left:idx_left+1), ...
                         left_axis(idx_left:idx_left+1), threshold, 'linear', 'extrap');
    end

    right_cut  = cut(ipk:end);
    right_axis = axis_vec(ipk:end);
    idx_right  = find(right_cut < threshold, 1, 'first');
    if isempty(idx_right) || idx_right <= 1
        x_right = right_axis(end);
    else
        x_right = interp1(right_cut(idx_right-1:idx_right), ...
                          right_axis(idx_right-1:idx_right), threshold, 'linear', 'extrap');
    end

    width = abs(x_right - x_left);
end
