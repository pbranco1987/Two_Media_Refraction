% =========================================================================
% RESUME PARAMETER SWEEP: pick up where the interrupted sweep left off
% =========================================================================
% This script loads the existing sweep_summary.mat, identifies all trials
% with 'pending' status, and runs only those. Results are written back
% into the same sweep_summary.mat and sweep_summary.csv files, preserving
% the original trial numbering and the n2==n3 skip optimisation.
%
% Usage: simply run this script in MATLAB from the project directory.
%        It will automatically detect which trials remain and resume.
% =========================================================================

clc; close all;

%% ======================== LOAD EXISTING PROGRESS ========================
baseOutputDir = fullfile(pwd, 'sweep_results');
summaryMatFile = fullfile(baseOutputDir, 'sweep_summary.mat');
summaryCsvFile = fullfile(baseOutputDir, 'sweep_summary.csv');

if ~isfile(summaryMatFile)
    error('Cannot find %s — nothing to resume from.', summaryMatFile);
end

fprintf('Loading existing sweep progress from:\n  %s\n\n', summaryMatFile);
S = load(summaryMatFile);
sweepTable     = S.sweepTable;
n2_values      = S.n2_values;
n3_values      = S.n3_values;
depth1_values  = S.depth1_values;
nCombinations  = height(sweepTable);

%% ======================== IDENTIFY PENDING TRIALS =======================
% A trial is pending if its Status is not 'success'.
pendingMask = ~strcmp(sweepTable.Status, 'success');
pendingIdx  = find(pendingMask);
nPending    = numel(pendingIdx);

% Rebuild the full parameter lists from the table
n2_list     = sweepTable.n2;
n3_list     = sweepTable.n3;
depth1_list = sweepTable.depth1_m;

% Rebuild skip mask for n2==n3 optimisation
sameN      = (abs(n2_list - n3_list) < 1e-10);
refDepth1  = depth1_values(1);
skipMask   = sameN & (abs(depth1_list - refDepth1) > 1e-10);

% Build reference trial lookup (for n2==n3 copies)
refTrialIdx = zeros(nCombinations, 1);
for k = 1:nCombinations
    if skipMask(k)
        match = find(abs(n2_list - n2_list(k)) < 1e-10 & ...
                     abs(n3_list - n3_list(k)) < 1e-10 & ...
                     abs(depth1_list - refDepth1) < 1e-10);
        refTrialIdx(k) = match(1);
    end
end

% Count how many pending trials actually need computation (not skip-copies)
nPendingCompute = sum(pendingMask & ~skipMask);
nPendingSkip    = sum(pendingMask & skipMask);
nAlreadyDone    = sum(~pendingMask);

fprintf('============================================\n');
fprintf('       RESUME PARAMETER SWEEP               \n');
fprintf('============================================\n');
fprintf('Total combinations: %d\n', nCombinations);
fprintf('Already completed:  %d\n', nAlreadyDone);
fprintf('Pending:            %d\n', nPending);
fprintf('  - To compute:     %d\n', nPendingCompute);
fprintf('  - To copy (n2==n3 skip): %d\n', nPendingSkip);
fprintf('============================================\n\n');

if nPending == 0
    fprintf('All trials already completed. Nothing to do.\n');
    return;
end

%% ======================== UNPACK TABLE INTO COLUMN VECTORS ==============
% The original script works with separate column vectors. Extract them
% from the loaded table so we can update them in place.
trialIdx           = sweepTable.Trial;
n2_col             = sweepTable.n2;
n3_col             = sweepTable.n3;
depth1_col         = sweepTable.depth1_m;
compTime           = sweepTable.ComputationTime_s;
peakMag            = sweepTable.PeakMagnitude;
peakX              = sweepTable.PeakX;
peakY              = sweepTable.PeakY;
peakZ              = sweepTable.PeakZ;
entropy_amp_norm   = sweepTable.EntropyAmpNorm;
entropy_int_norm   = sweepTable.EntropyIntNorm;
entropy_amp_raw    = sweepTable.EntropyAmpRaw;
entropy_int_raw    = sweepTable.EntropyIntRaw;
sharp_peak_mean    = sweepTable.SharpnessPkMean;
sharp_peak_median  = sweepTable.SharpnessPkMedian;
sharp_peak_rms     = sweepTable.SharpnessPkRMS;
res3_x             = sweepTable.Res3dB_X;
res3_y             = sweepTable.Res3dB_Y;
res3_z             = sweepTable.Res3dB_Z;
res3_vol           = sweepTable.Res3dB_Volume;
res3_equiv_diam    = sweepTable.Res3dB_EquivDiam;
res6_x             = sweepTable.Res6dB_X;
res6_y             = sweepTable.Res6dB_Y;
res6_z             = sweepTable.Res6dB_Z;
TBR_dB_col         = sweepTable.TBR_dB;
SCR_dB_col         = sweepTable.SCR_dB;
CNR_col            = sweepTable.CNR;
PSLR_dB_col        = sweepTable.PSLR_dB;
ISLR_dB_col        = sweepTable.ISLR_dB;
kurtosis_col       = sweepTable.Kurtosis;
skewness_col       = sweepTable.Skewness;
CV_col             = sweepTable.CV;
gini_col           = sweepTable.Gini;
pct_vox_50         = sweepTable.PctVox50;
pct_vox_90         = sweepTable.PctVox90;
pct_vox_99         = sweepTable.PctVox99;
eff_scatterers     = sweepTable.EffScatterers;
eff_scatterers_pct = sweepTable.EffScatterersPct;
grad_mean          = sweepTable.GradMean;
grad_mean_peak     = sweepTable.GradMeanPeak;
grad_max           = sweepTable.GradMax;
psf_asym_max       = sweepTable.PSF_AsymMax;
dyn_range_dB       = sweepTable.DynamicRange_dB;
status_col         = sweepTable.Status;
folder_col         = sweepTable.Folder;

% Column names (same order as the original script)
sweepVarNames = { ...
    'Trial', 'n2', 'n3', 'depth1_m', 'ComputationTime_s', ...
    'PeakMagnitude', 'PeakX', 'PeakY', 'PeakZ', ...
    'EntropyAmpNorm', 'EntropyIntNorm', 'EntropyAmpRaw', 'EntropyIntRaw', ...
    'SharpnessPkMean', 'SharpnessPkMedian', 'SharpnessPkRMS', ...
    'Res3dB_X', 'Res3dB_Y', 'Res3dB_Z', 'Res3dB_Volume', 'Res3dB_EquivDiam', ...
    'Res6dB_X', 'Res6dB_Y', 'Res6dB_Z', ...
    'TBR_dB', 'SCR_dB', 'CNR', ...
    'PSLR_dB', 'ISLR_dB', ...
    'Kurtosis', 'Skewness', 'CV', 'Gini', ...
    'PctVox50', 'PctVox90', 'PctVox99', 'EffScatterers', 'EffScatterersPct', ...
    'GradMean', 'GradMeanPeak', 'GradMax', ...
    'PSF_AsymMax', ...
    'DynamicRange_dB', ...
    'Status', 'Folder'};

%% ======================== RUN REMAINING TRIALS ==========================
sweepStartTime = tic;
nComputed = 0;
nProcessed = 0;

for qi = 1:nPending
    k = pendingIdx(qi);   % index into the full 648-row table

    n2_val     = n2_list(k);
    n3_val     = n3_list(k);
    depth1_val = depth1_list(k);

    folderName = sprintf('n2_%.2f_n3_%.2f_d1_%.2f', n2_val, n3_val, depth1_val);
    trialOutputDir = fullfile(baseOutputDir, folderName);
    folder_col{k} = folderName;
    nProcessed = nProcessed + 1;

    % ---- Skip redundant trial when n2 == n3 ----
    if skipMask(k)
        ref = refTrialIdx(k);

        % Only copy if the reference trial is already done
        if ~strcmp(status_col{ref}, 'success')
            fprintf('=== Trial %d (pending %d/%d): n2=%.2f, n3=%.2f, d1=%.2f -> DEFERRED (ref trial %d not yet done)\n', ...
                k, qi, nPending, n2_val, n3_val, depth1_val, ref);
            continue;
        end

        compTime(k)           = compTime(ref);
        peakMag(k)            = peakMag(ref);
        peakX(k)              = peakX(ref);
        peakY(k)              = peakY(ref);
        peakZ(k)              = peakZ(ref);
        entropy_amp_norm(k)   = entropy_amp_norm(ref);
        entropy_int_norm(k)   = entropy_int_norm(ref);
        entropy_amp_raw(k)    = entropy_amp_raw(ref);
        entropy_int_raw(k)    = entropy_int_raw(ref);
        sharp_peak_mean(k)    = sharp_peak_mean(ref);
        sharp_peak_median(k)  = sharp_peak_median(ref);
        sharp_peak_rms(k)     = sharp_peak_rms(ref);
        res3_x(k)             = res3_x(ref);
        res3_y(k)             = res3_y(ref);
        res3_z(k)             = res3_z(ref);
        res3_vol(k)           = res3_vol(ref);
        res3_equiv_diam(k)    = res3_equiv_diam(ref);
        res6_x(k)             = res6_x(ref);
        res6_y(k)             = res6_y(ref);
        res6_z(k)             = res6_z(ref);
        TBR_dB_col(k)         = TBR_dB_col(ref);
        SCR_dB_col(k)         = SCR_dB_col(ref);
        CNR_col(k)            = CNR_col(ref);
        PSLR_dB_col(k)        = PSLR_dB_col(ref);
        ISLR_dB_col(k)        = ISLR_dB_col(ref);
        kurtosis_col(k)       = kurtosis_col(ref);
        skewness_col(k)       = skewness_col(ref);
        CV_col(k)             = CV_col(ref);
        gini_col(k)           = gini_col(ref);
        pct_vox_50(k)         = pct_vox_50(ref);
        pct_vox_90(k)         = pct_vox_90(ref);
        pct_vox_99(k)         = pct_vox_99(ref);
        eff_scatterers(k)     = eff_scatterers(ref);
        eff_scatterers_pct(k) = eff_scatterers_pct(ref);
        grad_mean(k)          = grad_mean(ref);
        grad_mean_peak(k)     = grad_mean_peak(ref);
        grad_max(k)           = grad_max(ref);
        psf_asym_max(k)       = psf_asym_max(ref);
        dyn_range_dB(k)       = dyn_range_dB(ref);
        status_col{k}         = status_col{ref};
        folder_col{k}         = folder_col{ref};

        fprintf('=== Trial %d (pending %d/%d): n2=%.2f, n3=%.2f, d1=%.2f -> SKIPPED (n2==n3, copied from trial %d) ===\n', ...
            k, qi, nPending, n2_val, n3_val, depth1_val, ref);
    else
        % ---- Actually compute this trial ----
        nComputed = nComputed + 1;
        if ~exist(trialOutputDir, 'dir')
            mkdir(trialOutputDir);
        end

        fprintf('=== Trial %d (pending %d/%d) [compute %d/%d]: n2=%.2f, n3=%.2f, d1=%.2f ===\n', ...
            k, qi, nPending, nComputed, nPendingCompute, n2_val, n3_val, depth1_val);

        try
            % ---- Step 1: Run backprojection ----
            [Img, tTotal] = bp_hybrid(n2_val, n3_val, depth1_val, trialOutputDir);
            compTime(k) = tTotal;

            % Load grid variables from saved .mat
            matPath = fullfile(trialOutputDir, 'mat', 'bp_hybrid_results.mat');
            loaded = load(matPath, 'xGrid', 'yGrid', 'zGrid');
            xGrid = loaded.xGrid;
            yGrid = loaded.yGrid;
            zGrid = loaded.zGrid;

            % ---- Step 2: Run full quality analysis ----
            R = analyze_bp_quality_fn(Img, xGrid, yGrid, zGrid, ...
                    n2_val, n3_val, depth1_val, trialOutputDir);

            % ---- Step 3: Extract all metrics into columns ----
            peakMag(k) = R.peak.value;
            peakX(k)   = R.peak.location(1);
            peakY(k)   = R.peak.location(2);
            peakZ(k)   = R.peak.location(3);

            entropy_amp_norm(k) = R.entropy.normalized;
            entropy_int_norm(k) = R.entropy.intensity_normalized;
            entropy_amp_raw(k)  = R.entropy.shannon;
            entropy_int_raw(k)  = R.entropy.intensity;

            sharp_peak_mean(k)   = R.sharpness.peak_mean;
            sharp_peak_median(k) = R.sharpness.peak_median;
            sharp_peak_rms(k)    = R.sharpness.peak_rms;

            res3_x(k)          = R.resolution_3dB.x;
            res3_y(k)          = R.resolution_3dB.y;
            res3_z(k)          = R.resolution_3dB.z;
            res3_vol(k)        = R.resolution_3dB.volume;
            res3_equiv_diam(k) = R.resolution_3dB.equiv_diameter;

            res6_x(k) = R.resolution_6dB.x;
            res6_y(k) = R.resolution_6dB.y;
            res6_z(k) = R.resolution_6dB.z;

            TBR_dB_col(k) = R.contrast.TBR_dB;
            SCR_dB_col(k) = R.contrast.SCR_dB;
            CNR_col(k)    = R.contrast.CNR;

            PSLR_dB_col(k) = R.sidelobes.PSLR_dB;
            ISLR_dB_col(k) = R.sidelobes.ISLR_dB;

            kurtosis_col(k) = R.statistical.kurtosis;
            skewness_col(k) = R.statistical.skewness;
            CV_col(k)       = R.statistical.CV;
            gini_col(k)     = R.statistical.gini;

            pct_vox_50(k)         = R.energy.pct_voxels_50;
            pct_vox_90(k)         = R.energy.pct_voxels_90;
            pct_vox_99(k)         = R.energy.pct_voxels_99;
            eff_scatterers(k)     = R.energy.effective_scatterers;
            eff_scatterers_pct(k) = R.energy.effective_scatterers_pct;

            grad_mean(k)      = R.gradient.mean;
            grad_mean_peak(k) = R.gradient.mean_peak;
            grad_max(k)       = R.gradient.max;

            psf_asym_max(k) = R.psf.asymmetry_max;

            dyn_range_dB(k) = R.dynamic_range_dB;

            status_col{k} = 'success';
            fprintf('  -> Done in %.1fs | Peak=%.4f at (%.2f, %.2f, %.2f)\n', ...
                tTotal, peakMag(k), peakX(k), peakY(k), peakZ(k));

        catch ME
            status_col{k} = sprintf('error: %s', ME.message);
            fprintf('  -> FAILED: %s\n', ME.message);

            errorFile = fullfile(trialOutputDir, 'error_log.txt');
            fid = fopen(errorFile, 'w');
            fprintf(fid, 'Error for n2=%.2f, n3=%.2f, depth1=%.2f\n', n2_val, n3_val, depth1_val);
            fprintf(fid, 'Message: %s\n', ME.message);
            fprintf(fid, 'Identifier: %s\n', ME.identifier);
            for s = 1:numel(ME.stack)
                fprintf(fid, '  File: %s, Line: %d, Function: %s\n', ...
                    ME.stack(s).file, ME.stack(s).line, ME.stack(s).name);
            end
            fclose(fid);
        end

        close all;
    end

    % ---- INCREMENTAL SAVE (same format as original) ----
    elapsedSoFar = toc(sweepStartTime);
    sweepTable = table( ...
        trialIdx, n2_col, n3_col, depth1_col, compTime, ...
        peakMag, peakX, peakY, peakZ, ...
        entropy_amp_norm, entropy_int_norm, entropy_amp_raw, entropy_int_raw, ...
        sharp_peak_mean, sharp_peak_median, sharp_peak_rms, ...
        res3_x, res3_y, res3_z, res3_vol, res3_equiv_diam, ...
        res6_x, res6_y, res6_z, ...
        TBR_dB_col, SCR_dB_col, CNR_col, ...
        PSLR_dB_col, ISLR_dB_col, ...
        kurtosis_col, skewness_col, CV_col, gini_col, ...
        pct_vox_50, pct_vox_90, pct_vox_99, eff_scatterers, eff_scatterers_pct, ...
        grad_mean, grad_mean_peak, grad_max, ...
        psf_asym_max, dyn_range_dB, ...
        status_col, folder_col, ...
        'VariableNames', sweepVarNames);
    save(summaryMatFile, 'sweepTable', 'n2_values', 'n3_values', 'depth1_values', ...
        'elapsedSoFar', 'nCombinations');
    writetable(sweepTable, summaryCsvFile);

    nDone = sum(strcmp(status_col, 'success'));
    fprintf('  [Saved progress: %d/%d done | pending %d/%d processed | %.1f min elapsed]\n\n', ...
        nDone, nCombinations, qi, nPending, elapsedSoFar / 60);
end

totalResumeTime = toc(sweepStartTime);

%% ======================== HANDLE DEFERRED n2==n3 COPIES =================
% Some skip-copies may have been deferred if their reference trial was not
% yet done when first encountered. Now that all computable trials are done,
% do a second pass to fill in any remaining deferred copies.
stillPending = find(~strcmp(status_col, 'success') & skipMask);
if ~isempty(stillPending)
    fprintf('\n--- Second pass: filling %d deferred n2==n3 copies ---\n', numel(stillPending));
    for qi = 1:numel(stillPending)
        k = stillPending(qi);
        ref = refTrialIdx(k);
        if strcmp(status_col{ref}, 'success')
            compTime(k)           = compTime(ref);
            peakMag(k)            = peakMag(ref);
            peakX(k)              = peakX(ref);
            peakY(k)              = peakY(ref);
            peakZ(k)              = peakZ(ref);
            entropy_amp_norm(k)   = entropy_amp_norm(ref);
            entropy_int_norm(k)   = entropy_int_norm(ref);
            entropy_amp_raw(k)    = entropy_amp_raw(ref);
            entropy_int_raw(k)    = entropy_int_raw(ref);
            sharp_peak_mean(k)    = sharp_peak_mean(ref);
            sharp_peak_median(k)  = sharp_peak_median(ref);
            sharp_peak_rms(k)     = sharp_peak_rms(ref);
            res3_x(k)             = res3_x(ref);
            res3_y(k)             = res3_y(ref);
            res3_z(k)             = res3_z(ref);
            res3_vol(k)           = res3_vol(ref);
            res3_equiv_diam(k)    = res3_equiv_diam(ref);
            res6_x(k)             = res6_x(ref);
            res6_y(k)             = res6_y(ref);
            res6_z(k)             = res6_z(ref);
            TBR_dB_col(k)         = TBR_dB_col(ref);
            SCR_dB_col(k)         = SCR_dB_col(ref);
            CNR_col(k)            = CNR_col(ref);
            PSLR_dB_col(k)        = PSLR_dB_col(ref);
            ISLR_dB_col(k)        = ISLR_dB_col(ref);
            kurtosis_col(k)       = kurtosis_col(ref);
            skewness_col(k)       = skewness_col(ref);
            CV_col(k)             = CV_col(ref);
            gini_col(k)           = gini_col(ref);
            pct_vox_50(k)         = pct_vox_50(ref);
            pct_vox_90(k)         = pct_vox_90(ref);
            pct_vox_99(k)         = pct_vox_99(ref);
            eff_scatterers(k)     = eff_scatterers(ref);
            eff_scatterers_pct(k) = eff_scatterers_pct(ref);
            grad_mean(k)          = grad_mean(ref);
            grad_mean_peak(k)     = grad_mean_peak(ref);
            grad_max(k)           = grad_max(ref);
            psf_asym_max(k)       = psf_asym_max(ref);
            dyn_range_dB(k)       = dyn_range_dB(ref);
            status_col{k}         = status_col{ref};
            folder_col{k}         = folder_col{ref};
            fprintf('  Deferred copy: trial %d <- ref %d (n2=n3=%.2f, d1=%.2f)\n', ...
                k, ref, n2_list(k), depth1_list(k));
        else
            fprintf('  WARNING: ref trial %d still not success for trial %d\n', ref, k);
        end
    end

    % Final save after deferred copies
    sweepTable = table( ...
        trialIdx, n2_col, n3_col, depth1_col, compTime, ...
        peakMag, peakX, peakY, peakZ, ...
        entropy_amp_norm, entropy_int_norm, entropy_amp_raw, entropy_int_raw, ...
        sharp_peak_mean, sharp_peak_median, sharp_peak_rms, ...
        res3_x, res3_y, res3_z, res3_vol, res3_equiv_diam, ...
        res6_x, res6_y, res6_z, ...
        TBR_dB_col, SCR_dB_col, CNR_col, ...
        PSLR_dB_col, ISLR_dB_col, ...
        kurtosis_col, skewness_col, CV_col, gini_col, ...
        pct_vox_50, pct_vox_90, pct_vox_99, eff_scatterers, eff_scatterers_pct, ...
        grad_mean, grad_mean_peak, grad_max, ...
        psf_asym_max, dyn_range_dB, ...
        status_col, folder_col, ...
        'VariableNames', sweepVarNames);
    save(summaryMatFile, 'sweepTable', 'n2_values', 'n3_values', 'depth1_values', ...
        'totalResumeTime', 'nCombinations');
    writetable(sweepTable, summaryCsvFile);
end

%% ======================== SUMMARY VISUALIZATIONS ========================
successMask = strcmp(status_col, 'success');

if any(successMask)
    successTable = sweepTable(successMask, :);

    hmDefs = {
        'SharpnessPkMean',     'Sharpness (Peak/Mean)',      'jet'
        'EntropyAmpNorm',      'Entropy (Amplitude, Norm)',  'jet'
        'EntropyIntNorm',      'Entropy (Intensity, Norm)',  'jet'
        'Res3dB_X',            'Resolution X -3dB (m)',      'jet'
        'Res3dB_Y',            'Resolution Y -3dB (m)',      'jet'
        'Res3dB_Z',            'Resolution Z -3dB (m)',      'jet'
        'Res3dB_Volume',       'Resolution Volume (m^3)',    'jet'
        'SCR_dB',              'SCR (dB)',                   'jet'
        'PSLR_dB',             'PSLR (dB)',                  'jet'
        'ISLR_dB',             'ISLR (dB)',                  'jet'
        'CNR',                 'CNR',                        'jet'
        'Gini',                'Gini Coefficient',           'jet'
        'PctVox90',            '% Voxels for 90% Energy',    'jet'
        'DynamicRange_dB',     'Dynamic Range (dB)',         'hot'
        'ComputationTime_s',   'Computation Time (s)',       'hot'
    };

    for di = 1:numel(depth1_values)
        d1 = depth1_values(di);
        dMask = abs(successTable.depth1_m - d1) < 1e-6;
        subT = successTable(dMask, :);
        if isempty(subT), continue; end

        for mi = 1:size(hmDefs, 1)
            colName  = hmDefs{mi, 1};
            colTitle = hmDefs{mi, 2};
            cmap     = hmDefs{mi, 3};

            figure('Color', 'w', 'Position', [100 100 650 520], 'Visible', 'off');
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
            xlabel('n2 (soil layer 1)');
            ylabel('n3 (soil layer 2)');
            title(sprintf('%s | depth1 = %.2f m', colTitle, d1));

            fname = sprintf('heatmap_%s_d1_%.2f', lower(colName), d1);
            saveas(gcf, fullfile(baseOutputDir, [fname '.png']));
            saveas(gcf, fullfile(baseOutputDir, [fname '.fig']));
            close(gcf);
        end
    end

    % Cross-depth summary
    figure('Color', 'w', 'Position', [100 100 700 450], 'Visible', 'off');
    bestSharp = NaN(numel(depth1_values), 1);
    bestN2    = NaN(numel(depth1_values), 1);
    bestN3    = NaN(numel(depth1_values), 1);
    for di = 1:numel(depth1_values)
        d1 = depth1_values(di);
        dMask = abs(successTable.depth1_m - d1) < 1e-6;
        subT = successTable(dMask, :);
        if isempty(subT), continue; end
        [bestSharp(di), idx] = max(subT.SharpnessPkMean);
        bestN2(di) = subT.n2(idx);
        bestN3(di) = subT.n3(idx);
    end

    subplot(2,1,1);
    bar(depth1_values, bestSharp, 'FaceColor', [0.2 0.6 0.9]);
    xlabel('depth1 (m)'); ylabel('Best Sharpness');
    title('Best Sharpness vs DEM2 Depth'); grid on;

    subplot(2,1,2);
    plot(depth1_values, bestN2, '-o', 'DisplayName', 'n2', 'LineWidth', 1.5);
    hold on;
    plot(depth1_values, bestN3, '-s', 'DisplayName', 'n3', 'LineWidth', 1.5);
    xlabel('depth1 (m)'); ylabel('Refractive index');
    title('Optimal n2, n3 at Best Sharpness per depth1');
    legend('Location', 'best'); grid on;

    saveas(gcf, fullfile(baseOutputDir, 'summary_best_sharpness_vs_depth.png'));
    saveas(gcf, fullfile(baseOutputDir, 'summary_best_sharpness_vs_depth.fig'));
    close(gcf);
end

%% ======================== FINAL REPORT ==================================
nFinalSuccess = sum(strcmp(status_col, 'success'));
nFinalError   = sum(startsWith(status_col, 'error'));
nFinalPending = sum(strcmp(status_col, 'pending'));

fprintf('\n============================================\n');
fprintf('        RESUME SWEEP COMPLETE               \n');
fprintf('============================================\n');
fprintf('Total combinations: %d\n', nCombinations);
fprintf('Previously completed: %d\n', nAlreadyDone);
fprintf('Newly computed in this run: %d\n', nComputed);
fprintf('Now successful: %d / %d\n', nFinalSuccess, nCombinations);
fprintf('Errors: %d\n', nFinalError);
fprintf('Still pending: %d\n', nFinalPending);
fprintf('Resume time: %.1f min (%.1f hrs)\n', totalResumeTime/60, totalResumeTime/3600);
fprintf('Results: %s\n', baseOutputDir);
fprintf('============================================\n');
