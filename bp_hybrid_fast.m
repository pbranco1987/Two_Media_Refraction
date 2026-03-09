function [Img, tTotal] = bp_hybrid_fast(n2_in, n3_in, depth1_in, outputDir)
% =========================================================================
% FAST MULTI-LAYER SAR BACKPROJECTION (SNELL INIT + NEWTON REFINE)
% =========================================================================
% Drop-in replacement for bp_hybrid with major performance improvements:
%
% 1) ANALYTICAL SNELL INIT replaces DEM-IVS (Phase 1):
%    Solves flat-surface Snell's law via 3 iterations of 1D Newton to get
%    an excellent initial guess for refraction points P1 and P2. This
%    eliminates the costly 3D [nv x B x Nc] candidate expansion entirely.
%
% 2) DYNAMIC BATCH SIZING:
%    Without DEM-IVS 3D arrays, peak memory drops ~9x, enabling much
%    larger voxel chunks and radar batches. Auto-calculates optimal sizes
%    from actual free VRAM, potentially processing everything in 1 pass.
%
% 3) BROADCASTING (no repmat/reshape):
%    Newton section uses implicit expansion:
%      Xc/Yc/Zc [nv x 1] broadcast against [nv x B]
%      rx/ry/rz  [1 x B]  broadcast against [nv x B]
%      isL3_s    [nv x 1] broadcast against [nv x B]
%
% 4) anyL3 GUARD: skips Newton P2 + L3 cost paths for L2-only chunks.
%
% 4x oversampling via FFT zero-padding for sub-sample delay accuracy.
%
% Inputs/Outputs: identical interface to bp_hybrid.
% =========================================================================

clc; close all;

%% ======================== SETUP =========================================
g = gpuDevice; reset(g);
fprintf('GPU: %s (%.1f GB free)\n', g.Name, g.AvailableMemory/1e9);

% -------------------------- Physical parameters --------------------------
dataFile = 'IMOC_Inputs.mat';

if nargin < 1 || isempty(n2_in),     n2_in = 3.8;  end
if nargin < 2 || isempty(n3_in),     n3_in = 3.4;  end
if nargin < 3 || isempty(depth1_in), depth1_in = 0.85; end
if nargin < 4 || isempty(outputDir), outputDir = '.'; end

n2 = single(n2_in);
n3 = single(n3_in);
depth1 = single(depth1_in);
n_air = single(1.0);
c = single(299792458);
fc = single(425e6);

% Precomputed squares for Snell's law
n_air_sq = n_air^2;   % = 1
n2_sq    = n2^2;
n3_sq    = n3^2;

% -------------------------- 3D imaging grid ------------------------------
step = 0.05;
xGrid = single(3.2:step:5.4);
yGrid = single(3.2:step:5.4);
zGrid = single(-2.7:step:-0.1);

% -------------------------- Signal oversampling ----------------------------
oversampFactor = 4;  % 4x FFT zero-padding for sub-sample accuracy

% -------------------------- Newton parameters ----------------------------
nNewtonIter = 4;     % Newton-Raphson iterations (well-initialized by Snell)
nSnellIter  = 3;     % 1D Newton iterations for flat-surface Snell solve

%% ======================== LOAD DATA =====================================
fprintf('  FAST BACKPROJECTION (Snell init + Newton refine)\n');

S = loadData(dataFile);

radarData = single(S.radarData);
fs = single(S.samplingRate);
radarPos = single(S.radarPositions);
if size(radarPos,2) ~= 3, radarPos = radarPos.'; end
Nrad = size(radarPos, 1);
if size(radarData,2) ~= Nrad, radarData = radarData.'; end
Nt = size(radarData, 1);

t0 = single(0);
if isfield(S,'TimeAxis') && numel(S.TimeAxis) == Nt
    t0 = single(S.TimeAxis(1));
    if t0 > 1e-3, t0 = t0 * 1e-9; end
end

omega_c = single(2 * pi * fc);
applyPhase = ~isreal(S.radarData);

fprintf('Data: Nt=%d, Nrad=%d, fs=%.1f MHz\n', Nt, Nrad, double(fs)/1e6);
fprintf('Params: n2=%.1f, n3=%.1f, d1=%.2fm\n', n2, n3, depth1);
fprintf('Init: %d Snell + %d Newton\n', nSnellIter, nNewtonIter);

% --------------------- Analytic signal + oversampling --------------------
if isreal(radarData)
    H = [1; 2*ones(floor((Nt-1)/2),1,'single'); ...
         1+mod(Nt+1,2); zeros(floor((Nt-1)/2),1,'single')];
    radarData = ifft(fft(radarData,[],1).*H,[],1);
end

% ----- FFT-based 4x oversampling -----
if oversampFactor > 1
    Nt_orig = Nt;
    Nt_up   = Nt_orig * oversampFactor;
    F = fft(radarData, [], 1);
    halfN = floor(Nt_orig / 2);
    nZeros = Nt_up - Nt_orig;
    if mod(Nt_orig, 2) == 0
        F_up = [F(1:halfN, :); ...
                F(halfN+1, :) / 2; ...
                zeros(nZeros - 1, Nrad, 'single'); ...
                F(halfN+1, :) / 2; ...
                F(halfN+2:end, :)];
    else
        F_up = [F(1:halfN+1, :); ...
                zeros(nZeros, Nrad, 'single'); ...
                F(halfN+2:end, :)];
    end
    radarData = ifft(F_up, [], 1) * single(oversampFactor);
    Nt = Nt_up;
    fs = fs * single(oversampFactor);
    fprintf('Oversampling: %dx (Nt: %d -> %d, fs: %.1f -> %.1f MHz)\n', ...
        oversampFactor, Nt_orig, Nt, double(fs)/1e6/oversampFactor, double(fs)/1e6);
end

radarData = [radarData; zeros(1,Nrad,'single')];
Nt_pad = Nt + 1;
idxOff = single(1 - t0 * fs);
twoOverC = single(2) / c;   % Precomputed factor for tau = 2*pathLen/c

%% ======================== DEM + VOXEL GRID ===============================
[xD, yD, zD] = extractDEM(S.DEM);

dem.Zv = single(zD(:));
dem.Nx = int32(size(zD,2));  dem.Ny = int32(size(zD,1));
dem.x0 = single(xD(1));     dem.y0 = single(yD(1));
dem.xMax = single(xD(end));  dem.yMax = single(yD(end));
dem.dx = single(xD(2)-xD(1)); dem.dy = single(yD(2)-yD(1));
dem.invDx = single(1/dem.dx);  dem.invDy = single(1/dem.dy);

Nx = numel(xGrid); Ny = numel(yGrid); Nz = numel(zGrid);
[Xg, Yg, Zg] = ndgrid(xGrid, yGrid, zGrid);
Xg = Xg(:); Yg = Yg(:); Zg = Zg(:);

zS1 = demCPU(dem, Xg, Yg);
valid = Zg < zS1;
vidx = find(valid);
Nv = numel(vidx);

Xv = Xg(vidx); Yv = Yg(vidx); Zv = Zg(vidx);
zS1v = zS1(vidx);
zS2v = zS1v - depth1;
isL3 = Zv < zS2v;

fprintf('Grid: %dx%dx%d, Valid: %d voxels (L2=%d, L3=%d)\n', ...
    Nx, Ny, Nz, Nv, sum(~isL3), sum(isL3));

%% ======================== GPU SETUP ======================================
demG.Zv = gpuArray(dem.Zv);
demG.Nx = dem.Nx; demG.Ny = dem.Ny;
demG.x0 = dem.x0; demG.y0 = dem.y0;
demG.xMax = dem.xMax; demG.yMax = dem.yMax;
demG.dx = dem.dx; demG.dy = dem.dy;
demG.invDx = dem.invDx; demG.invDy = dem.invDy;

xMin = demG.x0;  xMax_ = demG.xMax;
yMin = demG.y0;  yMax_ = demG.yMax;

%% ======================== DATA TRANSFER TO GPU ===========================
radarData_g = gpuArray(radarData);
radarPos_g  = gpuArray(radarPos);

%% ======================== BATCH SIZE ESTIMATION ==========================
% Peak GPU memory occurs in Newton phase: ~30 live [nv x B] arrays.
% Use conservative estimate matching bp_hybrid's proven approach.
availMem = 0.50 * g.AvailableMemory;
bytesPerArray = 4;
numArrays = 30;           % Newton peak: ~30 live [nv x B] arrays

maxNvB = floor(availMem / (bytesPerArray * numArrays));

radarBatchSize = 512;     % Proven batch size from bp_hybrid
voxelChunkSize = floor(maxNvB / radarBatchSize);
voxelChunkSize = max(1000, min(20000, voxelChunkSize));

nVoxelChunks = ceil(Nv / voxelChunkSize);
nRadarBatches = ceil(Nrad / radarBatchSize);
totalBatches = nVoxelChunks * nRadarBatches;

fprintf('Batching: %d x %d = %d batches (vChunk=%d, rBatch=%d)\n', ...
    nVoxelChunks, nRadarBatches, totalBatches, voxelChunkSize, radarBatchSize);

%% ======================== CONSTANTS =====================================
EPS = single(1e-9);

%% ======================== MAIN PROCESSING LOOP ===========================
fprintf('\n--- Processing ---\n');
tStart = tic;

ImgAccum = zeros(Nv, 1, 'single');
batchCount = 0;
lastPrintTime = 0;

for vc = 1:nVoxelChunks
    v1 = (vc-1)*voxelChunkSize + 1;
    v2 = min(vc*voxelChunkSize, Nv);
    nv = v2 - v1 + 1;

    Xc    = gpuArray(Xv(v1:v2));       % [nv x 1]
    Yc    = gpuArray(Yv(v1:v2));       % [nv x 1]
    Zc    = gpuArray(Zv(v1:v2));       % [nv x 1]
    isL3c = gpuArray(isL3(v1:v2));     % [nv x 1] logical
    zS1_c = gpuArray(single(zS1v(v1:v2)));  % [nv x 1] DEM1 elevation

    isL3_s = single(isL3c);   % [nv x 1] pre-computed mask
    isL2_s = single(~isL3c);  % [nv x 1] pre-computed mask
    anyL3  = any(isL3c);

    chunkAccum = complex(zeros(nv, 1, 'single', 'gpuArray'));

    for rb = 1:nRadarBatches
        r1 = (rb-1)*radarBatchSize + 1;
        r2 = min(rb*radarBatchSize, Nrad);
        B  = r2 - r1 + 1;
        batchCount = batchCount + 1;

        rx = reshape(radarPos_g(r1:r2, 1), 1, B);   % [1 x B]
        ry = reshape(radarPos_g(r1:r2, 2), 1, B);
        rz = reshape(radarPos_g(r1:r2, 3), 1, B);

        traces = radarData_g(:, r1:r2);              % [Nt_pad x B]

        % ==============================================================
        % PHASE 1: ANALYTICAL SNELL'S LAW INITIALIZATION
        % ==============================================================
        % Solve the flat-surface refraction problem analytically using
        % the ray parameter p = n_air * sin(theta_air). For a multi-layer
        % flat model, Snell's law gives a 1D equation in p:
        %   f(p) = sum_i p*h_i/sqrt(n_i^2 - p^2) - Lh = 0
        % Solved with 3 Newton iterations (quadratic convergence).
        % ==============================================================

        % Heights for flat-surface model
        h1 = max(EPS, rz - zS1_c);                                  % [nv x B] air height
        h2 = (zS1_c - Zc) .* isL2_s + depth1 * isL3_s;             % [nv x 1] soil1 segment
        h3 = max(single(0), zS1_c - depth1 - Zc) .* isL3_s;        % [nv x 1] soil2 segment (0 for L2)

        % Horizontal distance radar -> voxel
        Lh = sqrt((Xc - rx).^2 + (Yc - ry).^2) + EPS;              % [nv x B]

        % Initial guess: sin(theta) for a straight ray (no refraction)
        htot = h1 + h2 + h3;
        p = n_air * Lh ./ sqrt(Lh.^2 + htot.^2);                   % [nv x B]

        % Solve for ray parameter p via 1D Newton
        for si = 1:nSnellIter
            t1_sq = n_air_sq - p.^2;
            t2_sq = n2_sq    - p.^2;
            t3_sq = n3_sq    - p.^2;

            t1 = sqrt(t1_sq);
            t2 = sqrt(t2_sq);
            t3 = sqrt(t3_sq);

            f_val = p .* (h1 ./ t1 + h2 ./ t2 + h3 ./ t3) - Lh;
            f_der = h1 .* n_air_sq ./ (t1_sq .* t1) + ...
                    h2 .* n2_sq    ./ (t2_sq .* t2) + ...
                    h3 .* n3_sq    ./ (t3_sq .* t3) + EPS;

            p = p - f_val ./ f_der;
            p = max(single(0.001), min(single(0.999), p));
        end

        % Convert ray parameter to horizontal distances
        t1 = sqrt(n_air_sq - p.^2);
        t2 = sqrt(n2_sq    - p.^2);

        u1 = p .* h1 ./ t1;           % R -> P1 horizontal distance
        u2 = p .* h2 ./ t2;           % P1 -> P2 (L3) or P1 -> V (L2)

        % Horizontal direction unit vector
        invLh = 1 ./ Lh;
        dir_x = (Xc - rx) .* invLh;   % [nv x B]
        dir_y = (Yc - ry) .* invLh;   % [nv x B]

        % P1 refraction point on DEM1
        x1 = rx + dir_x .* u1;
        y1 = ry + dir_y .* u1;

        % P2 refraction point on DEM2 (L3: P1->P2; L2: reaches V, P2 unused)
        x2 = rx + dir_x .* (u1 + u2);
        y2 = ry + dir_y .* (u1 + u2);

        % Clamp to DEM domain
        x1 = max(xMin, min(xMax_, x1));
        y1 = max(yMin, min(yMax_, y1));
        x2 = max(xMin, min(xMax_, x2));
        y2 = max(yMin, min(yMax_, y2));

        % Free Snell-phase intermediates to reclaim GPU memory for Newton
        clear h1 h2 h3 Lh htot p t1_sq t2_sq t3_sq t1 t2 t3 f_val f_der u1 u2 invLh dir_x dir_y;

        % ==============================================================
        % PHASE 2: NEWTON-RAPHSON LOCAL REFINEMENT ON ACTUAL DEM
        % ==============================================================
        % Snell init assumes flat surface; Newton refines on the real DEM
        % with surface gradients. Broadcasting: Xc/Yc/Zc [nv x 1],
        % rx/ry/rz [1 x B], isL3_s/isL2_s [nv x 1] all broadcast to [nv x B].
        % ==============================================================

        for iter = 1:nNewtonIter
            optimizeP1 = (mod(iter, 2) == 1);

            if optimizeP1
                % ===================== Newton for P1 ===================
                [z1, hx1, hy1] = demGPU_with_grad(demG, x1, y1);
                z2 = demGPU(demG, x2, y2) - depth1;

                dx_r = x1 - rx; dy_r = y1 - ry; dz_r = z1 - rz;
                r1 = sqrt(dx_r.^2 + dy_r.^2 + dz_r.^2) + EPS;

                dx_2 = (x1 - Xc) .* isL2_s + (x1 - x2) .* isL3_s;
                dy_2 = (y1 - Yc) .* isL2_s + (y1 - y2) .* isL3_s;
                dz_2 = (z1 - Zc) .* isL2_s + (z1 - z2) .* isL3_s;
                r2 = sqrt(dx_2.^2 + dy_2.^2 + dz_2.^2) + EPS;

                A1 = dx_r + dz_r .* hx1;
                B1 = dy_r + dz_r .* hy1;
                A2 = dx_2 + dz_2 .* hx1;
                B2 = dy_2 + dz_2 .* hy1;

                F1 = n_air * A1 ./ r1 + n2 * A2 ./ r2;
                F2 = n_air * B1 ./ r1 + n2 * B2 ./ r2;

                hx2 = hx1.^2; hy2 = hy1.^2; hxy = hx1 .* hy1;
                r1_sq = r1.^2; r2_sq = r2.^2;

                J11 = n_air*(1+hx2-A1.^2./r1_sq)./r1 + n2*(1+hx2-A2.^2./r2_sq)./r2;
                J22 = n_air*(1+hy2-B1.^2./r1_sq)./r1 + n2*(1+hy2-B2.^2./r2_sq)./r2;
                J12 = n_air*(hxy-A1.*B1./r1_sq)./r1 + n2*(hxy-A2.*B2./r2_sq)./r2;

                det_J  = J11.*J22 - J12.^2 + EPS;
                delta_x = (J12.*F2 - J22.*F1) ./ det_J;
                delta_y = (J12.*F1 - J11.*F2) ./ det_J;

                step_mag = sqrt(delta_x.^2 + delta_y.^2);
                max_step = single(0.2);
                scale = min(single(1), max_step ./ (step_mag + EPS));

                x1 = max(xMin, min(xMax_, x1 + scale .* delta_x));
                y1 = max(yMin, min(yMax_, y1 + scale .* delta_y));

            else
                % ===================== Newton for P2 ===================
                if anyL3
                    z1 = demGPU(demG, x1, y1);
                    [z2_s1, hx2, hy2] = demGPU_with_grad(demG, x2, y2);
                    z2 = z2_s1 - depth1;

                    dx_12 = x2 - x1; dy_12 = y2 - y1; dz_12 = z2 - z1;
                    r12 = sqrt(dx_12.^2 + dy_12.^2 + dz_12.^2) + EPS;

                    dx_2v = x2 - Xc; dy_2v = y2 - Yc; dz_2v = z2 - Zc;
                    r2v = sqrt(dx_2v.^2 + dy_2v.^2 + dz_2v.^2) + EPS;

                    A1 = dx_12 + dz_12 .* hx2;
                    B1 = dy_12 + dz_12 .* hy2;
                    A2 = dx_2v + dz_2v .* hx2;
                    B2 = dy_2v + dz_2v .* hy2;

                    F1 = (n2 * A1 ./ r12 + n3 * A2 ./ r2v) .* isL3_s;
                    F2 = (n2 * B1 ./ r12 + n3 * B2 ./ r2v) .* isL3_s;

                    hx2_sq = hx2.^2; hy2_sq = hy2.^2; hxy2 = hx2 .* hy2;
                    r12_sq = r12.^2; r2v_sq = r2v.^2;

                    J11 = n2*(1+hx2_sq-A1.^2./r12_sq)./r12 + n3*(1+hx2_sq-A2.^2./r2v_sq)./r2v + isL2_s;
                    J22 = n2*(1+hy2_sq-B1.^2./r12_sq)./r12 + n3*(1+hy2_sq-B2.^2./r2v_sq)./r2v + isL2_s;
                    J12 = n2*(hxy2-A1.*B1./r12_sq)./r12 + n3*(hxy2-A2.*B2./r2v_sq)./r2v;

                    det_J  = J11.*J22 - J12.^2 + EPS;
                    delta_x = (J12.*F2 - J22.*F1) ./ det_J;
                    delta_y = (J12.*F1 - J11.*F2) ./ det_J;

                    step_mag = sqrt(delta_x.^2 + delta_y.^2);
                    max_step = single(0.2);
                    scale = min(single(1), max_step ./ (step_mag + EPS));

                    x2 = max(xMin, min(xMax_, x2 + scale .* delta_x));
                    y2 = max(yMin, min(yMax_, y2 + scale .* delta_y));
                end
            end
        end

        % ==============================================================
        % FINAL PATH LENGTH + SIGNAL INTERPOLATION
        % ==============================================================
        z1 = demGPU(demG, x1, y1);
        z2 = demGPU(demG, x2, y2) - depth1;

        Ra     = sqrt((x1 - rx).^2 + (y1 - ry).^2 + (z1 - rz).^2);
        Rs1_l2 = sqrt((Xc - x1).^2 + (Yc - y1).^2 + (Zc - z1).^2);
        Rs1_l3 = sqrt((x2 - x1).^2 + (y2 - y1).^2 + (z2 - z1).^2);
        Rs2_l3 = sqrt((Xc - x2).^2 + (Yc - y2).^2 + (Zc - z2).^2);

        % Fused pathLen -> tau (eliminates pathLen intermediate)
        tau = twoOverC * ((n_air*Ra + n2*Rs1_l2) .* isL2_s + ...
                          (n_air*Ra + n2*Rs1_l3 + n3*Rs2_l3) .* isL3_s);

        idxF = tau * fs + idxOff;
        validI = isfinite(idxF) & (idxF >= 1) & (idxF <= Nt);

        iFl = int32(max(1, min(Nt, floor(idxF))));
        iCe = int32(min(Nt_pad, iFl + 1));
        wCe = idxF - single(iFl);

        base = int32(0:B-1) * int32(Nt_pad);
        val = traces(iFl + base) .* (1 - wCe) + traces(iCe + base) .* wCe;

        if applyPhase
            val = val .* exp(1j * omega_c * tau);
        end

        val(~validI) = 0;
        chunkAccum = chunkAccum + sum(val, 2);

        % --------------------- Progress display -----------------------
        elapsed = toc(tStart);
        if elapsed - lastPrintTime >= 5 || batchCount == totalBatches
            pct = 100 * batchCount / totalBatches;
            eta = elapsed / batchCount * (totalBatches - batchCount);
            if eta > 60
                etaStr = sprintf('%dm %02.0fs', floor(eta/60), mod(eta,60));
            else
                etaStr = sprintf('%.0fs', eta);
            end
            fprintf('[%5.1f%%] Batch %d/%d | %.1fs | ETA: %s\n', ...
                pct, batchCount, totalBatches, elapsed, etaStr);
            lastPrintTime = elapsed;
        end
    end

    ImgAccum(v1:v2) = gather(chunkAccum) / Nrad;
end

tTotal = toc(tStart);
if tTotal > 60
    fprintf('  COMPLETED: %dm %.1fs\n', floor(tTotal/60), mod(tTotal,60));
else
    fprintf('  COMPLETED: %.2fs\n', tTotal);
end

%% ======================== OUTPUT ========================================
Img = complex(zeros(Nx, Ny, Nz, 'single'));
Img(vidx) = ImgAccum;

M = abs(ImgAccum); M = M(M > 0);
if ~isempty(M)
    Mn = M / sum(M);
    fprintf('Entropy: %.3f, Sharpness: %.1f\n', -sum(Mn.*log(Mn+eps)), max(M)/mean(M));
end

[~, mi] = max(abs(Img(:)));
[ix, iy, iz] = ind2sub([Nx, Ny, Nz], mi);
fprintf('Peak: (%.3f, %.3f, %.3f) m\n', xGrid(ix), yGrid(iy), zGrid(iz));

xDEM = xD;
yDEM = yD;
zDEM1 = zD;
zDEM2 = zD - double(depth1);

matDir = fullfile(outputDir, 'mat');
if ~exist(matDir, 'dir'), mkdir(matDir); end
matFile = fullfile(matDir, 'bp_hybrid_results.mat');
save(matFile, 'Img', 'xGrid', 'yGrid', 'zGrid', 'n2', 'n3', 'depth1', 'tTotal', ...
    'xDEM', 'yDEM', 'zDEM1', 'zDEM2', 'oversampFactor', '-v7.3');
fprintf('Saved: %s\n', matFile);

plotDir = fullfile(outputDir, 'plots');
if ~exist(plotDir, 'dir'), mkdir(plotDir); end
visualize(Img, xGrid, yGrid, zGrid, 40, n2, n3, depth1);
saveas(gcf, fullfile(plotDir, 'bp_hybrid_slices.png'));
saveas(gcf, fullfile(plotDir, 'bp_hybrid_slices.fig'));
fprintf('Saved plots to: %s\n', plotDir);
end

%% ========================================================================
% DEM INTERPOLATION FUNCTIONS
% ========================================================================

function [z, hx, hy] = demGPU_with_grad(d, xq, yq)
tx = (xq - d.x0) * d.invDx;
ty = (yq - d.y0) * d.invDy;
j = max(int32(0), min(d.Nx-2, int32(floor(tx))));
i = max(int32(0), min(d.Ny-2, int32(floor(ty))));
ax = tx - single(j);
ay = ty - single(i);
j = j + 1; i = i + 1;
Ny = d.Ny;
idx = i + (j-1)*Ny;

z00 = d.Zv(idx);
z10 = d.Zv(idx + Ny);
z01 = d.Zv(idx + 1);
z11 = d.Zv(idx + Ny + 1);

z = (1-ax).*(1-ay).*z00 + ax.*(1-ay).*z10 + (1-ax).*ay.*z01 + ax.*ay.*z11;
hx = ((z10 - z00).*(1-ay) + (z11 - z01).*ay) * d.invDx;
hy = ((z01 - z00).*(1-ax) + (z11 - z10).*ax) * d.invDy;
end

function z = demGPU(d, xq, yq)
tx = (xq - d.x0) * d.invDx;
ty = (yq - d.y0) * d.invDy;
j = max(int32(0), min(d.Nx-2, int32(floor(tx))));
i = max(int32(0), min(d.Ny-2, int32(floor(ty))));
ax = tx - single(j);
ay = ty - single(i);
j = j + 1; i = i + 1;
Ny = d.Ny;
idx = i + (j-1)*Ny;

z = (1-ax).*(1-ay).*d.Zv(idx) + ax.*(1-ay).*d.Zv(idx+Ny) + ...
    (1-ax).*ay.*d.Zv(idx+1) + ax.*ay.*d.Zv(idx+Ny+1);
end

function z = demCPU(d, xq, yq)
tx = double((xq - d.x0) * d.invDx);
ty = double((yq - d.y0) * d.invDy);
j = max(0, min(double(d.Nx-2), floor(tx)));
i = max(0, min(double(d.Ny-2), floor(ty)));
ax = tx - j; ay = ty - i;
j = int32(j)+1; i = int32(i)+1;
Ny = double(d.Ny); Z = double(d.Zv);

z = single((1-ax).*(1-ay).*Z(i+(j-1)*Ny) + ax.*(1-ay).*Z(i+j*Ny) + ...
           (1-ax).*ay.*Z(i+1+(j-1)*Ny) + ax.*ay.*Z(i+1+j*Ny));
end

function S = loadData(f)
r = load(f);
if isfield(r,'S'), S = r.S;
    while isstruct(S) && isfield(S,'S'), S = S.S; end
else, S = r;
end
end

function [x, y, Z] = extractDEM(D)
if isvector(D.x) && isvector(D.y)
    x = double(D.x(:)).'; y = double(D.y(:)); Z = double(D.z);
    if size(Z,1)==numel(x) && size(Z,2)==numel(y), Z = Z.'; end
else
    x = double(D.x(1,:)); y = double(D.y(:,1)); Z = double(D.z);
    if ~isequal(size(Z), [numel(y), numel(x)]), Z = Z.'; end
end
if any(diff(x)<0), x = fliplr(x); Z = Z(:, end:-1:1); end
if any(diff(y)<0), y = flipud(y); Z = Z(end:-1:1, :); end
end

function visualize(Img, xG, yG, zG, dyn, n2, n3, d1)
M = abs(Img); pk = max(M(:));
if pk <= 0, return; end
MdB = 20*log10(M/pk + eps); MdB(MdB < -dyn) = -dyn;

[~, mi] = max(M(:));
[ix, iy, iz] = ind2sub(size(M), mi);

figure('Color', 'w', 'Position', [100 100 1400 400]); colormap('jet');

subplot(1,3,1);
imagesc(xG, yG, MdB(:,:,iz).'); axis xy image; colorbar; clim([-dyn 0]);
title(sprintf('XY (Z=%.2f)', zG(iz))); xlabel('x'); ylabel('y');

subplot(1,3,2);
imagesc(xG, zG, squeeze(MdB(:,iy,:)).'); axis xy image; colorbar; clim([-dyn 0]);
title(sprintf('XZ (Y=%.2f)', yG(iy))); xlabel('x'); ylabel('z');

subplot(1,3,3);
imagesc(yG, zG, squeeze(MdB(ix,:,:)).'); axis xy image; colorbar; clim([-dyn 0]);
title(sprintf('YZ (X=%.2f)', xG(ix))); xlabel('y'); ylabel('z');

sgtitle(sprintf('Fast BP: n2=%.2f, n3=%.2f, d1=%.2fm', n2, n3, d1));
end
