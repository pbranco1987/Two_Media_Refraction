%% ANALYZE_BP_QUALITY.m
% =========================================================================
% ANALYZE BACKPROJECTION IMAGE QUALITY (Standalone Script Version)
% =========================================================================
%
% This standalone script performs a comprehensive quality analysis of a
% backprojection (BP) image volume. It is the original interactive version
% of the analysis; for automated use in the parameter sweep pipeline,
% see the function version: analyze_bp_quality_fn.m.
%
% Metrics computed:
%   - Entropy (Shannon, amplitude & intensity based) - lower = more focused
%   - Sharpness (peak/mean, peak/median, peak/RMS) - higher = better
%   - Resolution (-3dB and -6dB widths in x, y, z) - smaller = better
%   - Contrast (TBR, SCR, CNR) - higher = better
%   - Sidelobes (PSLR, ISLR) - higher ratios = cleaner image
%   - Statistical (kurtosis, skewness, CV, Gini) - characterize distribution shape
%   - Energy concentration (% voxels for 50/90/99% energy)
%   - Gradient (edge sharpness metrics)
%   - PSF asymmetry and dynamic range
%
% Plots:
%   - Figure 1: Image quality analysis (linear scale, 8 subplots)
%   - Figure 2: DEM interfaces visualization (6 subplots)
%
% Usage:
%   1. Set resultFile to point to your bp_hybrid_results.mat
%   2. Set dataFile to point to the original IMOC_Inputs.mat (for DEM)
%   3. Run the script
%
% =========================================================================

close all;

%% ======================== LOAD RESULTS ==================================
% Path to the backprojection result file (output of bp_hybrid.m)
resultFile = 'bp_hybrid_fast_results.mat';  % Change to match your output filename
% Path to the original radar data file (needed for the DEM surface data)
dataFile = 'IMOC_Inputs_vs2.mat';           % Change to match your input filename

fprintf('Loading %s...\n', resultFile);
data = load(resultFile);

% Extract the complex 3D image volume and coordinate grids
Img = data.Img;
xGrid = data.xGrid;
yGrid = data.yGrid;
zGrid = data.zGrid;

% Extract refractive indices and depth from the results file if available;
% otherwise fall back to default values for standalone use.
if isfield(data, 'n2'), n2 = data.n2; else, n2 = NaN; end
if isfield(data, 'n3'), n3 = data.n3; else, n3 = NaN; end
if isfield(data, 'depth1'), depth1 = data.depth1; else, depth1 = 0.5; end

% Load the DEM (Digital Elevation Model) from the original input data.
% The DEM is needed to overlay the air-soil and soil-soil interfaces
% on the image slices in Figure 2.
fprintf('Loading DEM from %s...\n', dataFile);
S = load(dataFile);
if isfield(S, 'S'), S = S.S; end
[xDEM, yDEM, zDEM1] = extractDEM(S.DEM);
zDEM2 = zDEM1 - depth1;  % Second interface (parallel to DEM1, shifted down by depth1)

%% ======================== BASIC INFO ====================================
% Compute magnitude of the complex image volume. All quality metrics
% operate on the magnitude (absolute value), not the complex data.
M = abs(Img);
[Nx, Ny, Nz] = size(M);

% Grid spacing in each direction (assumed uniform)
dx = xGrid(2) - xGrid(1);
dy = yGrid(2) - yGrid(1);
dz = zGrid(2) - zGrid(1);

fprintf('\n');
fprintf('══════════════════════════════════════════════════════════════\n');
fprintf('  IMAGE QUALITY ANALYSIS\n');
fprintf('══════════════════════════════════════════════════════════════\n');

if ~isnan(n2)
    fprintf('\n  Parameters:\n');
    fprintf('    n2 (soil layer 1): %.4f\n', n2);
    fprintf('    n3 (soil layer 2): %.4f\n', n3);
    fprintf('    depth1:            %.4f m\n', depth1);
end

fprintf('\n  Grid: %d × %d × %d voxels\n', Nx, Ny, Nz);
fprintf('  Spacing: %.3f × %.3f × %.3f m\n', dx, dy, dz);

%% ======================== FIND PEAK =====================================
% Locate the global maximum (strongest scatterer response) in the volume.
% The peak position is used as the reference for resolution, contrast, and
% sidelobe metrics.
[peakVal, peakIdx] = max(M(:));
[ix_peak, iy_peak, iz_peak] = ind2sub([Nx, Ny, Nz], peakIdx);

x_peak = xGrid(ix_peak);
y_peak = yGrid(iy_peak);
z_peak = zGrid(iz_peak);

fprintf('\n──────────────────────────────────────────────────────────────\n');
fprintf('  PEAK\n');
fprintf('──────────────────────────────────────────────────────────────\n');
fprintf('    Location: (%.4f, %.4f, %.4f) m\n', x_peak, y_peak, z_peak);
fprintf('    Indices:  (%d, %d, %d)\n', ix_peak, iy_peak, iz_peak);
fprintf('    Value:    %.6e\n', peakVal);

%% ======================== ENTROPY =======================================
% Entropy measures the dispersion of energy across the image volume.
% Lower entropy indicates more focused energy (desirable for well-focused images).
M_flat = M(:);
M_nonzero = M_flat(M_flat > 0);

% Amplitude-based Shannon entropy: treat normalized magnitudes as a
% probability distribution. Normalized to [0, 1] using H/H_max.
p = M_nonzero / sum(M_nonzero);
entropy_shannon = -sum(p .* log(p));
entropy_max = log(numel(M_nonzero));
entropy_normalized = entropy_shannon / entropy_max;

% Intensity-weighted entropy (power-based): uses squared magnitudes as
% the distribution. More sensitive to dominant scatterers than amplitude entropy.
M2 = M_nonzero.^2;
p2 = M2 / sum(M2);
entropy_intensity = -sum(p2 .* log(p2));
entropy_intensity_max = log(numel(M2));
entropy_intensity_normalized = entropy_intensity / entropy_intensity_max;

fprintf('\n──────────────────────────────────────────────────────────────\n');
fprintf('  ENTROPY (lower = better focused)\n');
fprintf('──────────────────────────────────────────────────────────────\n');
fprintf('  Amplitude-based:\n');
fprintf('    Shannon entropy:    %.4f\n', entropy_shannon);
fprintf('    Normalized (0-1):   %.4f\n', entropy_normalized);
fprintf('  Intensity-weighted (power):\n');
fprintf('    Shannon entropy:    %.4f\n', entropy_intensity);
fprintf('    Normalized (0-1):   %.4f\n', entropy_intensity_normalized);

%% ======================== SHARPNESS =====================================
sharpness_peak_mean = peakVal / mean(M_nonzero);
sharpness_peak_median = peakVal / median(M_nonzero);
sharpness_peak_rms = peakVal / sqrt(mean(M_nonzero.^2));

fprintf('\n──────────────────────────────────────────────────────────────\n');
fprintf('  SHARPNESS (higher = better)\n');
fprintf('──────────────────────────────────────────────────────────────\n');
fprintf('    Peak / Mean:        %.2f\n', sharpness_peak_mean);
fprintf('    Peak / Median:      %.2f\n', sharpness_peak_median);
fprintf('    Peak / RMS:         %.2f\n', sharpness_peak_rms);

%% ======================== ADDITIONAL SAR QUALITY METRICS ================

fprintf('\n──────────────────────────────────────────────────────────────\n');
fprintf('  ADDITIONAL SAR QUALITY METRICS\n');
fprintf('──────────────────────────────────────────────────────────────\n');

% --- Contrast Metrics ---
% Define three concentric spherical regions centered on the peak for
% contrast and sidelobe analysis:
%   - Mainlobe: within half the largest resolution dimension (core PSF)
%   - Signal: within 2x mainlobe (target + near sidelobes)
%   - Background: beyond 3x mainlobe (clutter/noise region)
[Xgrid, Ygrid, Zgrid] = ndgrid(xGrid, yGrid, zGrid);
dist_from_peak = sqrt((Xgrid - x_peak).^2 + (Ygrid - y_peak).^2 + (Zgrid - z_peak).^2);

% Mainlobe region: approximate -3dB envelope as a sphere
mainlobe_radius = max([res_x, res_y, res_z]) / 2;
mainlobe_mask = dist_from_peak <= mainlobe_radius;

% Background region: far from the peak (beyond 3x mainlobe radius)
background_radius = 3 * mainlobe_radius;
background_mask = dist_from_peak > background_radius;

% Signal region: extended area around the peak (for gradient analysis)
signal_mask = dist_from_peak <= 2 * mainlobe_radius;

if any(mainlobe_mask(:)) && any(background_mask(:))
    mainlobe_mean = mean(M(mainlobe_mask));
    background_mean = mean(M(background_mask));
    background_std = std(M(background_mask));
    
    % Target-to-Background Ratio (TBR): mean mainlobe amplitude / mean background
    TBR = mainlobe_mean / background_mean;
    TBR_dB = 20 * log10(TBR);

    % Signal-to-Clutter Ratio (SCR): peak amplitude / mean background
    SCR = peakVal / background_mean;
    SCR_dB = 20 * log10(SCR);

    % Contrast-to-Noise Ratio (CNR): contrast normalized by background variability
    CNR = (mainlobe_mean - background_mean) / background_std;
    
    fprintf('\n  Contrast Metrics:\n');
    fprintf('    Target-to-Background (TBR):   %.2f (%.1f dB)\n', TBR, TBR_dB);
    fprintf('    Signal-to-Clutter (SCR):      %.2f (%.1f dB)\n', SCR, SCR_dB);
    fprintf('    Contrast-to-Noise (CNR):      %.2f\n', CNR);
else
    TBR = NaN; TBR_dB = NaN; SCR = NaN; SCR_dB = NaN; CNR = NaN;
    fprintf('\n  Contrast Metrics: Could not compute (regions too small)\n');
end

% --- Sidelobe Metrics ---
% Zero out the mainlobe to isolate sidelobe contributions
M_outside_mainlobe = M;
M_outside_mainlobe(mainlobe_mask) = 0;

% Peak Sidelobe Level Ratio (PSLR): main peak / highest sidelobe peak
% Higher PSLR = better sidelobe suppression
peak_sidelobe = max(M_outside_mainlobe(:));
if peak_sidelobe > 0
    PSLR = peakVal / peak_sidelobe;
    PSLR_dB = 20 * log10(PSLR);
else
    PSLR = inf;
    PSLR_dB = inf;
end

% Integrated Sidelobe Ratio (ISLR)
mainlobe_energy = sum(M(mainlobe_mask).^2);
sidelobe_energy = sum(M_outside_mainlobe(:).^2);
if sidelobe_energy > 0
    ISLR = mainlobe_energy / sidelobe_energy;
    ISLR_dB = 10 * log10(ISLR);
else
    ISLR = inf;
    ISLR_dB = inf;
end

fprintf('\n  Sidelobe Metrics:\n');
fprintf('    Peak Sidelobe Ratio (PSLR):   %.2f (%.1f dB)\n', PSLR, PSLR_dB);
fprintf('    Integrated Sidelobe (ISLR):   %.2f (%.1f dB)\n', ISLR, ISLR_dB);

% --- Statistical Metrics ---
% Higher-order statistics characterize the shape of the amplitude distribution.

% Kurtosis: measures "peakedness" of distribution. Values > 3 indicate
% a sharper peak than Gaussian (leptokurtic) -- desirable for focused images.
M_centered = M_nonzero - mean(M_nonzero);
kurtosis_val = mean(M_centered.^4) / (mean(M_centered.^2)^2);

% Skewness: asymmetry of distribution. Positive skewness = long right tail
% (few strong scatterers dominate, typical for well-focused SAR images).
skewness_val = mean(M_centered.^3) / (mean(M_centered.^2)^1.5);

% Coefficient of Variation (CV): std/mean. Higher CV = more contrast between
% strong and weak scatterers, indicating better focusing.
CV = std(M_nonzero) / mean(M_nonzero);

% Gini coefficient: measures inequality of amplitude distribution.
% Values near 0 = uniform distribution; near 1 = all energy in few voxels.
M_sorted = sort(M_nonzero);
n_vox = numel(M_sorted);
gini = (2 * sum((1:n_vox)' .* M_sorted) / (n_vox * sum(M_sorted))) - (n_vox + 1) / n_vox;

fprintf('\n  Statistical Metrics:\n');
fprintf('    Kurtosis:                     %.2f (>3 = peaked)\n', kurtosis_val);
fprintf('    Skewness:                     %.2f (>0 = right tail)\n', skewness_val);
fprintf('    Coefficient of Variation:    %.2f\n', CV);
fprintf('    Gini Coefficient:             %.4f (1 = concentrated)\n', gini);

% --- Energy Concentration Metrics ---
% Determine what fraction of voxels contains a given percentage of total energy.
% Lower percentages = better energy concentration (tighter focus).
M2_sorted = sort(M_nonzero.^2, 'descend');  % Sort squared magnitudes (energy) descending
cumulative_energy = cumsum(M2_sorted) / sum(M2_sorted);  % Cumulative energy fraction

% Find the minimum number of voxels needed to capture 50%, 90%, and 99% of total energy
idx_50 = find(cumulative_energy >= 0.50, 1, 'first');
idx_90 = find(cumulative_energy >= 0.90, 1, 'first');
idx_99 = find(cumulative_energy >= 0.99, 1, 'first');

pct_voxels_50 = 100 * idx_50 / n_vox;  % % of voxels needed for 50% energy
pct_voxels_90 = 100 * idx_90 / n_vox;  % % of voxels needed for 90% energy
pct_voxels_99 = 100 * idx_99 / n_vox;  % % of voxels needed for 99% energy

fprintf('\n  Energy Concentration:\n');
fprintf('    50%% energy in %.2f%% of voxels\n', pct_voxels_50);
fprintf('    90%% energy in %.2f%% of voxels\n', pct_voxels_90);
fprintf('    99%% energy in %.2f%% of voxels\n', pct_voxels_99);

% --- Image Gradient (Edge Sharpness) ---
% The gradient magnitude measures how sharply the image transitions between
% high and low amplitude regions. High gradients near the peak indicate
% a well-focused PSF with sharp edges.
[Gx, Gy, Gz] = gradient(M, dx, dy, dz);  % 3D numerical gradient components
gradient_magnitude = sqrt(Gx.^2 + Gy.^2 + Gz.^2);  % Gradient magnitude volume
mean_gradient = mean(gradient_magnitude(:));         % Global mean gradient
max_gradient = max(gradient_magnitude(:));           % Maximum gradient anywhere

% Gradient specifically in the peak region (local edge sharpness)
gradient_at_peak_region = gradient_magnitude(signal_mask);
mean_gradient_peak = mean(gradient_at_peak_region(:));

fprintf('\n  Gradient (Edge Sharpness):\n');
fprintf('    Mean gradient (global):       %.2e\n', mean_gradient);
fprintf('    Mean gradient (near peak):    %.2e\n', mean_gradient_peak);
fprintf('    Max gradient:                 %.2e\n', max_gradient);

% --- Point Spread Function (PSF) Analysis ---
% Asymmetry: ratio of resolutions
asymmetry_xy = max(res_x, res_y) / min(res_x, res_y);
asymmetry_xz = max(res_x, res_z) / min(res_x, res_z);
asymmetry_yz = max(res_y, res_z) / min(res_y, res_z);
asymmetry_max = max([asymmetry_xy, asymmetry_xz, asymmetry_yz]);

% Resolution volume (already computed, but also compute equivalent sphere diameter)
equiv_diameter = 2 * (3 * res_volume / (4 * pi))^(1/3);

fprintf('\n  PSF Analysis:\n');
fprintf('    Resolution volume:            %.4e m³\n', res_volume);
fprintf('    Equivalent sphere diameter:   %.4f m\n', equiv_diameter);
fprintf('    Asymmetry (max ratio):        %.2f (1 = isotropic)\n', asymmetry_max);
fprintf('    Asymmetry XY/XZ/YZ:           %.2f / %.2f / %.2f\n', asymmetry_xy, asymmetry_xz, asymmetry_yz);

% --- Dynamic Range ---
% Ratio of peak amplitude to the noise floor. Voxels below 1% of peak
% are excluded as noise. Higher dynamic range = cleaner image.
M_above_noise = M(M > 0.01 * peakVal);  % Keep only above 1% of peak
if ~isempty(M_above_noise)
    dynamic_range = peakVal / min(M_above_noise);
    dynamic_range_dB = 20 * log10(dynamic_range);
else
    dynamic_range = NaN;
    dynamic_range_dB = NaN;
end

fprintf('\n  Dynamic Range:\n');
fprintf('    Peak / Min (>1%% peak):        %.1f dB\n', dynamic_range_dB);

% --- Effective Number of Scatterers ---
% Based on the "participation ratio" from statistical physics.
% Measures how many voxels effectively contribute to the total energy.
% Lower values = energy concentrated in fewer voxels = better focus.
M4_sum = sum(M_nonzero.^4);
M2_sum = sum(M_nonzero.^2);
if M4_sum > 0
    effective_scatterers = M2_sum^2 / M4_sum;
    effective_scatterers_pct = 100 * effective_scatterers / n_vox;
else
    effective_scatterers = NaN;
    effective_scatterers_pct = NaN;
end

fprintf('\n  Effective Number of Scatterers:\n');
fprintf('    Count:                        %.1f (%.2f%% of voxels)\n', effective_scatterers, effective_scatterers_pct);

%% ======================== RESOLUTION (-3dB) =============================
% Resolution is measured as the width of the Point Spread Function (PSF)
% at -3 dB and -6 dB thresholds along each axis direction.
M_norm = M / peakVal;  % Peak-normalized magnitude [0, 1]

% Extract 1D profiles (cuts) through the peak along each axis
cut_x = squeeze(M_norm(:, iy_peak, iz_peak));  % X-cut at peak Y, Z
cut_y = squeeze(M_norm(ix_peak, :, iz_peak));  % Y-cut at peak X, Z
cut_z = squeeze(M_norm(ix_peak, iy_peak, :));  % Z-cut at peak X, Y

% Thresholds in linear amplitude (corresponding to -3 dB and -6 dB)
thresh_3dB = 10^(-3/20);  % ~0.7079 (half-power point)
thresh_6dB = 10^(-6/20);  % ~0.5012 (quarter-power point)

% Compute widths by interpolating where the profile crosses the threshold
res_x = computeResolution(xGrid, cut_x, thresh_3dB);
res_y = computeResolution(yGrid, cut_y, thresh_3dB);
res_z = computeResolution(zGrid, cut_z, thresh_3dB);
res_volume = res_x * res_y * res_z;  % Resolution cell volume (m^3)

% -6 dB widths (wider, characterizes extended PSF shape)
res_x_6dB = computeResolution(xGrid, cut_x, thresh_6dB);
res_y_6dB = computeResolution(yGrid, cut_y, thresh_6dB);
res_z_6dB = computeResolution(zGrid, cut_z, thresh_6dB);

fprintf('\n──────────────────────────────────────────────────────────────\n');
fprintf('  RESOLUTION -3dB (smaller = better)\n');
fprintf('──────────────────────────────────────────────────────────────\n');
fprintf('    X direction:  %.4f m\n', res_x);
fprintf('    Y direction:  %.4f m\n', res_y);
fprintf('    Z direction:  %.4f m\n', res_z);
fprintf('    Volume:       %.4e m³\n', res_volume);

fprintf('\n  RESOLUTION -6dB:\n');
fprintf('    X: %.4f m, Y: %.4f m, Z: %.4f m\n', res_x_6dB, res_y_6dB, res_z_6dB);

%% ======================== SUMMARY =======================================
fprintf('\n══════════════════════════════════════════════════════════════\n');
fprintf('  SUMMARY - KEY METRICS FOR n2, n3 OPTIMIZATION\n');
fprintf('══════════════════════════════════════════════════════════════\n');
fprintf('  Lower is better:\n');
fprintf('    Entropy (amplitude):     %.4f (normalized 0-1)\n', entropy_normalized);
fprintf('    Entropy (intensity):     %.4f (normalized 0-1)\n', entropy_intensity_normalized);
fprintf('    Resolution avg:          %.4f m\n', (res_x + res_y + res_z)/3);
fprintf('    Resolution volume:       %.4e m³\n', res_volume);
fprintf('    %% voxels for 90%% energy: %.2f%%\n', pct_voxels_90);
fprintf('  Higher is better:\n');
fprintf('    Sharpness (peak/mean):   %.2f\n', sharpness_peak_mean);
fprintf('    Gini coefficient:        %.4f\n', gini);
fprintf('    SCR:                     %.1f dB\n', SCR_dB);
fprintf('    PSLR:                    %.1f dB\n', PSLR_dB);
fprintf('    ISLR:                    %.1f dB\n', ISLR_dB);
fprintf('══════════════════════════════════════════════════════════════\n\n');

%% ======================== STORE RESULTS =================================
results.n2 = n2;
results.n3 = n3;
results.depth1 = depth1;

% Peak
results.peak.value = peakVal;
results.peak.location = [x_peak, y_peak, z_peak];

% Entropy
results.entropy.shannon = entropy_shannon;
results.entropy.normalized = entropy_normalized;
results.entropy.intensity = entropy_intensity;
results.entropy.intensity_normalized = entropy_intensity_normalized;

% Sharpness
results.sharpness.peak_mean = sharpness_peak_mean;
results.sharpness.peak_median = sharpness_peak_median;
results.sharpness.peak_rms = sharpness_peak_rms;

% Resolution
results.resolution_3dB.x = res_x;
results.resolution_3dB.y = res_y;
results.resolution_3dB.z = res_z;
results.resolution_3dB.volume = res_volume;
results.resolution_3dB.equiv_diameter = equiv_diameter;
results.resolution_6dB.x = res_x_6dB;
results.resolution_6dB.y = res_y_6dB;
results.resolution_6dB.z = res_z_6dB;

% Contrast
results.contrast.TBR = TBR;
results.contrast.TBR_dB = TBR_dB;
results.contrast.SCR = SCR;
results.contrast.SCR_dB = SCR_dB;
results.contrast.CNR = CNR;

% Sidelobes
results.sidelobes.PSLR = PSLR;
results.sidelobes.PSLR_dB = PSLR_dB;
results.sidelobes.ISLR = ISLR;
results.sidelobes.ISLR_dB = ISLR_dB;

% Statistical
results.statistical.kurtosis = kurtosis_val;
results.statistical.skewness = skewness_val;
results.statistical.CV = CV;
results.statistical.gini = gini;

% Energy concentration
results.energy.pct_voxels_50 = pct_voxels_50;
results.energy.pct_voxels_90 = pct_voxels_90;
results.energy.pct_voxels_99 = pct_voxels_99;
results.energy.effective_scatterers = effective_scatterers;

% Gradient
results.gradient.mean = mean_gradient;
results.gradient.mean_peak = mean_gradient_peak;
results.gradient.max = max_gradient;

% PSF
results.psf.asymmetry_max = asymmetry_max;
results.psf.asymmetry = [asymmetry_xy, asymmetry_xz, asymmetry_yz];

% Dynamic range
results.dynamic_range_dB = dynamic_range_dB;

%% ========================================================================
% FIGURE 1: IMAGE QUALITY ANALYSIS (LINEAR SCALE)
% =========================================================================

figure('Color', 'w', 'Position', [50 50 1600 900], 'Name', 'BP Image Quality Analysis (Linear Scale)');

% Row 1: 2D Image Slices
subplot(2, 4, 1);
slice_xy = squeeze(M_norm(:, :, iz_peak)).';
imagesc(xGrid, yGrid, slice_xy);
axis xy image; colorbar;
colormap(gca, 'jet');
hold on;
plot(x_peak, y_peak, 'k+', 'MarkerSize', 15, 'LineWidth', 2);
theta = linspace(0, 2*pi, 100);
plot(x_peak + res_x/2*cos(theta), y_peak + res_y/2*sin(theta), 'k--', 'LineWidth', 1.5);
hold off;
xlabel('x [m]'); ylabel('y [m]');
title(sprintf('XY Slice at Z = %.2f m', z_peak));

subplot(2, 4, 2);
slice_xz = squeeze(M_norm(:, iy_peak, :)).';
imagesc(xGrid, zGrid, slice_xz);
axis xy image; colorbar;
colormap(gca, 'jet');
hold on;
plot(x_peak, z_peak, 'k+', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
xlabel('x [m]'); ylabel('z [m]');
title(sprintf('XZ Slice at Y = %.2f m', y_peak));

subplot(2, 4, 3);
slice_yz = squeeze(M_norm(ix_peak, :, :)).';
imagesc(yGrid, zGrid, slice_yz);
axis xy image; colorbar;
colormap(gca, 'jet');
hold on;
plot(y_peak, z_peak, 'k+', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
xlabel('y [m]'); ylabel('z [m]');
title(sprintf('YZ Slice at X = %.2f m', x_peak));

% Metrics text box
subplot(2, 4, 4);
axis off;
text(0.05, 0.98, 'KEY METRICS', 'FontSize', 12, 'FontWeight', 'bold');
ypos = 0.88;
if ~isnan(n2)
    text(0.05, ypos, sprintf('n_2=%.3f, n_3=%.3f, d_1=%.2fm', n2, n3, depth1), 'FontSize', 9); 
    ypos = ypos - 0.08;
end
text(0.05, ypos, 'Entropy (norm):', 'FontSize', 9, 'FontWeight', 'bold'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('  Ampl: %.4f  Int: %.4f', entropy_normalized, entropy_intensity_normalized), 'FontSize', 8, 'Color', 'b'); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('Sharpness: %.2f', sharpness_peak_mean), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('Gini: %.4f', gini), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.08;
text(0.05, ypos, 'Resolution -3dB:', 'FontSize', 9, 'FontWeight', 'bold'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('  X:%.3f Y:%.3f Z:%.3fm', res_x, res_y, res_z), 'FontSize', 8); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('SCR: %.1f dB', SCR_dB), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('PSLR: %.1f dB', PSLR_dB), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('ISLR: %.1f dB', ISLR_dB), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('90%% energy in %.1f%% voxels', pct_voxels_90), 'FontSize', 8);

% Row 2: 1D Cuts
subplot(2, 4, 5);
plot(xGrid, cut_x, 'b-', 'LineWidth', 1.5);
hold on;
yline(thresh_3dB, 'r--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(thresh_6dB, 'm--', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
xline(x_peak - res_x/2, 'g:', 'LineWidth', 1.5);
xline(x_peak + res_x/2, 'g:', 'LineWidth', 1.5);
hold off;
xlim([xGrid(1) xGrid(end)]); ylim([0 1.05]);
xlabel('x [m]'); ylabel('Normalized Amplitude');
title(sprintf('X Cut (Res = %.4f m)', res_x));
grid on;

subplot(2, 4, 6);
plot(yGrid, cut_y, 'b-', 'LineWidth', 1.5);
hold on;
yline(thresh_3dB, 'r--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(thresh_6dB, 'm--', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
xline(y_peak - res_y/2, 'g:', 'LineWidth', 1.5);
xline(y_peak + res_y/2, 'g:', 'LineWidth', 1.5);
hold off;
xlim([yGrid(1) yGrid(end)]); ylim([0 1.05]);
xlabel('y [m]'); ylabel('Normalized Amplitude');
title(sprintf('Y Cut (Res = %.4f m)', res_y));
grid on;

subplot(2, 4, 7);
plot(zGrid, cut_z, 'b-', 'LineWidth', 1.5);
hold on;
yline(thresh_3dB, 'r--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(thresh_6dB, 'm--', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
xline(z_peak - res_z/2, 'g:', 'LineWidth', 1.5);
xline(z_peak + res_z/2, 'g:', 'LineWidth', 1.5);
hold off;
xlim([zGrid(1) zGrid(end)]); ylim([0 1.05]);
xlabel('z [m]'); ylabel('Normalized Amplitude');
title(sprintf('Z Cut (Res = %.4f m)', res_z));
grid on;

subplot(2, 4, 8);
histogram(M_norm(M_norm > 0), 100, 'FaceColor', 'b', 'EdgeColor', 'none', 'Normalization', 'probability');
xlabel('Normalized Amplitude'); ylabel('Probability');
title('Amplitude Distribution');
xlim([0 1]); grid on;

if ~isnan(n2)
    sgtitle(sprintf('Image Analysis: n_2 = %.3f, n_3 = %.3f', n2, n3), 'FontSize', 14, 'FontWeight', 'bold');
end

%% ========================================================================
% FIGURE 2: DEM INTERFACES
% =========================================================================

figure('Color', 'w', 'Position', [100 100 1500 800], 'Name', 'DEM Interfaces and Layers');

% --- Subplot 1: 3D view of both DEM surfaces ---
subplot(2, 3, 1);
[X_dem, Y_dem] = meshgrid(xDEM, yDEM);

% Plot DEM1 (air-soil1 interface)
h1 = surf(X_dem, Y_dem, zDEM1, 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'FaceColor', [0.6 0.4 0.2]);
hold on;

% Plot DEM2 (soil1-soil2 interface)
h2 = surf(X_dem, Y_dem, zDEM2, 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'FaceColor', [0.4 0.3 0.5]);

% Plot peak location
h3 = plot3(x_peak, y_peak, z_peak, 'r*', 'MarkerSize', 15, 'LineWidth', 2);

% Add vertical line from peak to surfaces
plot3([x_peak x_peak], [y_peak y_peak], [z_peak zGrid(end)], 'r:', 'LineWidth', 1);

hold off;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('3D View: DEM Interfaces');
legend([h1 h2 h3], {'DEM1 (Air-Soil1)', 'DEM2 (Soil1-Soil2)', 'Peak'}, 'Location', 'northeast');
view(45, 30);
axis tight;
grid on;

% --- Subplot 2: DEM1 surface (top view) ---
subplot(2, 3, 2);
imagesc(xDEM, yDEM, zDEM1);
axis xy image; 
cb = colorbar;
ylabel(cb, 'z [m]');
hold on;
plot(x_peak, y_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
% Add contour lines
contour(xDEM, yDEM, zDEM1, 10, 'k', 'LineWidth', 0.5);
hold off;
xlabel('x [m]'); ylabel('y [m]');
title('DEM1: Air-Soil1 Interface');
colormap(gca, 'copper');

% --- Subplot 3: DEM2 surface (top view) ---
subplot(2, 3, 3);
imagesc(xDEM, yDEM, zDEM2);
axis xy image;
cb = colorbar;
ylabel(cb, 'z [m]');
hold on;
plot(x_peak, y_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
contour(xDEM, yDEM, zDEM2, 10, 'k', 'LineWidth', 0.5);
hold off;
xlabel('x [m]'); ylabel('y [m]');
title(sprintf('DEM2: Soil1-Soil2 (depth_1 = %.2f m)', depth1));
colormap(gca, 'copper');

% --- Subplot 4: XZ slice with DEM interfaces overlaid ---
subplot(2, 3, 4);
slice_xz = squeeze(M_norm(:, iy_peak, :)).';
imagesc(xGrid, zGrid, slice_xz);
axis xy; colorbar;
hold on;

% Interpolate DEM at the Y slice location
[~, iy_dem] = min(abs(yDEM - y_peak));
dem1_slice = zDEM1(iy_dem, :);
dem2_slice = zDEM2(iy_dem, :);

% Plot DEM lines (with white background for visibility)
plot(xDEM, dem1_slice, 'w-', 'LineWidth', 3);
plot(xDEM, dem1_slice, 'k-', 'LineWidth', 1.5);
plot(xDEM, dem2_slice, 'w-', 'LineWidth', 3);
plot(xDEM, dem2_slice, 'm-', 'LineWidth', 1.5);

% Peak marker
plot(x_peak, z_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);

% Add text labels
text(xGrid(end)-0.1, mean(dem1_slice)+0.05, 'DEM1', 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
text(xGrid(end)-0.1, mean(dem2_slice)+0.05, 'DEM2', 'Color', 'm', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');

hold off;
xlabel('x [m]'); ylabel('z [m]');
title(sprintf('XZ Slice at Y = %.2f m', y_peak));
colormap(gca, 'jet');
ylim([zGrid(1) max(zGrid(end), max(dem1_slice(:))+0.1)]);

% --- Subplot 5: YZ slice with DEM interfaces overlaid ---
subplot(2, 3, 5);
slice_yz = squeeze(M_norm(ix_peak, :, :)).';
imagesc(yGrid, zGrid, slice_yz);
axis xy; colorbar;
hold on;

% Interpolate DEM at the X slice location
[~, ix_dem] = min(abs(xDEM - x_peak));
dem1_slice_y = zDEM1(:, ix_dem);
dem2_slice_y = zDEM2(:, ix_dem);

% Plot DEM lines
plot(yDEM, dem1_slice_y, 'w-', 'LineWidth', 3);
plot(yDEM, dem1_slice_y, 'k-', 'LineWidth', 1.5);
plot(yDEM, dem2_slice_y, 'w-', 'LineWidth', 3);
plot(yDEM, dem2_slice_y, 'm-', 'LineWidth', 1.5);

% Peak marker
plot(y_peak, z_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);

% Add text labels
text(yGrid(end)-0.1, mean(dem1_slice_y)+0.05, 'DEM1', 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
text(yGrid(end)-0.1, mean(dem2_slice_y)+0.05, 'DEM2', 'Color', 'm', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');

hold off;
xlabel('y [m]'); ylabel('z [m]');
title(sprintf('YZ Slice at X = %.2f m', x_peak));
colormap(gca, 'jet');
ylim([zGrid(1) max(zGrid(end), max(dem1_slice_y(:))+0.1)]);

% --- Subplot 6: Cross-section schematic ---
subplot(2, 3, 6);
hold on;

% Draw layers
xsch = [0 1];

% Air layer
fill([xsch fliplr(xsch)], [1 1 0.85 0.85], [0.7 0.9 1.0], 'EdgeColor', 'none');

% Soil 1 layer
fill([xsch fliplr(xsch)], [0.85 0.85 0.5 0.5], [0.76 0.60 0.42], 'EdgeColor', 'none');

% Soil 2 layer
fill([xsch fliplr(xsch)], [0.5 0.5 0.05 0.05], [0.55 0.45 0.35], 'EdgeColor', 'none');

% Interface lines
plot(xsch, [0.85 0.85], 'k-', 'LineWidth', 2);
plot(xsch, [0.5 0.5], 'm-', 'LineWidth', 2);

% Labels for layers
text(0.5, 0.925, 'AIR', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0 0 0.5]);
text(0.5, 0.675, sprintf('SOIL 1 (n_2 = %.2f)', n2), 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', 'w');
text(0.5, 0.275, sprintf('SOIL 2 (n_3 = %.2f)', n3), 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', 'w');

% Interface labels
text(1.05, 0.85, 'DEM1', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
text(1.05, 0.5, 'DEM2', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'm');

% Depth annotation with arrow
annotation('textarrow', [0.87 0.87], [0.35 0.28], 'String', sprintf('depth_1 = %.2f m', depth1), ...
    'FontSize', 10, 'HeadStyle', 'vback2');

% Target marker
plot(0.5, 0.35, 'rp', 'MarkerSize', 20, 'MarkerFaceColor', 'r');
text(0.58, 0.35, 'Target', 'FontSize', 11, 'Color', 'r', 'FontWeight', 'bold');

% Refractive index labels on the right
text(1.12, 0.925, 'n_1 = 1.0', 'FontSize', 10);
text(1.12, 0.675, sprintf('n_2 = %.2f', n2), 'FontSize', 10);
text(1.12, 0.275, sprintf('n_3 = %.2f', n3), 'FontSize', 10);

hold off;
title('Layer Model', 'FontSize', 12, 'FontWeight', 'bold');
xlim([-0.05 1.35]);
ylim([0 1.05]);
axis off;
box off;

sgtitle(sprintf('DEM Interfaces: n_2 = %.2f, n_3 = %.2f, depth_1 = %.2f m', n2, n3, depth1), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% ======================== SAVE ANALYSIS =================================
% Save the compact results struct for later comparison or loading
analysisFile = strrep(resultFile, '.mat', '_analysis.mat');
save(analysisFile, 'results');
fprintf('Analysis saved to: %s\n\n', analysisFile);

%% ========================================================================
% HELPER FUNCTIONS
% =========================================================================

function width = computeResolution(axis_vec, cut, threshold)
% COMPUTERESOLUTION  Measure the width of a 1D profile at a given threshold.
%
% Finds the two points where the profile crosses the threshold level on
% either side of the peak, using linear interpolation between samples.
% Returns the distance between these crossing points.
%
% Inputs:
%   axis_vec  - Coordinate vector (e.g., xGrid)
%   cut       - 1D amplitude profile through the peak
%   threshold - Normalized threshold level (e.g., 0.7079 for -3 dB)
%
% Output:
%   width     - Width of the profile at the given threshold (same units as axis_vec)
    cut = cut(:);
    axis_vec = axis_vec(:);
    [~, ipk] = max(cut);
    
    left_cut = cut(1:ipk);
    left_axis = axis_vec(1:ipk);
    idx_left = find(left_cut < threshold, 1, 'last');
    if isempty(idx_left) || idx_left >= length(left_cut)
        x_left = left_axis(1);
    else
        x_left = interp1(left_cut(idx_left:idx_left+1), left_axis(idx_left:idx_left+1), threshold, 'linear', 'extrap');
    end
    
    right_cut = cut(ipk:end);
    right_axis = axis_vec(ipk:end);
    idx_right = find(right_cut < threshold, 1, 'first');
    if isempty(idx_right) || idx_right <= 1
        x_right = right_axis(end);
    else
        x_right = interp1(right_cut(idx_right-1:idx_right), right_axis(idx_right-1:idx_right), threshold, 'linear', 'extrap');
    end
    
    width = abs(x_right - x_left);
end

function [x, y, Z] = extractDEM(D)
% EXTRACTDEM  Extract DEM data and ensure consistent orientation.
%
% Handles two common DEM formats:
%   1) D.x and D.y are vectors, D.z is a 2D matrix
%   2) D.x and D.y are meshgrid-style 2D matrices
%
% Guarantees output has:
%   - x: row vector with strictly increasing values
%   - y: column vector with strictly increasing values
%   - Z: size [numel(y) x numel(x)] (rows = y, columns = x)
    if isvector(D.x) && isvector(D.y)
        x = double(D.x(:)).';
        y = double(D.y(:));
        Z = double(D.z);
        if size(Z,1) == numel(x) && size(Z,2) == numel(y)
            Z = Z.';
        end
    else
        x = double(D.x(1,:));
        y = double(D.y(:,1));
        Z = double(D.z);
        if ~isequal(size(Z), [numel(y), numel(x)])
            Z = Z.';
        end
    end
    % Ensure monotonically increasing axes (flip if descending)
    if any(diff(x) < 0), x = fliplr(x); Z = Z(:, end:-1:1); end
    if any(diff(y) < 0), y = flipud(y); Z = Z(end:-1:1, :); end
end
