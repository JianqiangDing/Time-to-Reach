function run_cora(configPath, settingsPath, outPath)
% RUN_CORA - CORA reachability on one benchmark; called by tools/cora/run.py.
%
% The vector field is generated from the config's equation strings (-f for
% backward benchmarks). Every time-interval reachable set is compared with
% each target: 'disjoint' (certainly outside), 'inside' (certainly inside) or
% 'possible'. Box targets use CORA's exact zonotope/interval tests, balls the
% interval hull of the set, level sets interval arithmetic on the hull. Initial
% balls and level sets are covered by boxes (optionally split); 'inside' proves
% reachability only for boxes that certainly meet the initial set. A reachable
% set that does not cover the horizon (CORA stops on set explosion) is reported
% as such.

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, 'CORA')));
cfg = jsondecode(fileread(configPath));
st = jsondecode(fileread(settingsPath));
if isfield(st, 'threads') && ~isempty(st.threads)
    maxNumCompThreads(st.threads);
end

eqs = cellstr(cfg.system.equations);
n = numel(eqs);
backward = isfield(cfg, 'direction') && strcmp(cfg.direction, 'backward');
sgn = '';
if backward
    sgn = '-';
end
body = strjoin(cellfun(@(e) [sgn '(' aux_matlab(e) ')'], eqs, 'UniformOutput', false), '; ');
f = str2func(['@(x,u) [' body ']']);
sys = nonlinearSys(['ttr_' lower(cfg.id)], f, n, 1);

options = struct('timeStep', st.timeStep, 'taylorTerms', st.taylorTerms, ...
    'zonotopeOrder', st.zonotopeOrder, 'alg', st.alg, 'tensorOrder', st.tensorOrder);
t0 = cfg.time_horizon(1);
t1 = cfg.time_horizon(2);

% initial boxes (with a flag: certainly meets the initial set)
splits = 1;
if isfield(st, 'initial_splits')
    splits = st.initial_splits;
end
boxes = {};
meets = [];
overInit = false;
initSets = aux_sets(cfg.initial_sets);
for k = 1:numel(initSets)
    [b, m] = aux_initial_boxes(initSets{k}, n, splits);
    boxes = [boxes, b]; %#ok<AGROW>
    meets = [meets, m]; %#ok<AGROW>
    overInit = overInit || ~strcmp(initSets{k}.type, 'box');
end
targetSets = aux_sets(cfg.target_sets);

% warm-up: derivative generation and first calls are not part of the timed reach
params = struct('tStart', t0, 'tFinal', t0 + 2 * st.timeStep, 'R0', zonotope(boxes{1}), 'U', zonotope(0));
tw = tic;
reach(sys, params, options);
warmup = toc(tw);

statuses = {};
segments = zeros(0, 2);
covered = true;
reachedUntil = Inf;
provenInside = false;
reachTime = 0;
checkTime = 0;
for b = 1:numel(boxes)
    params = struct('tStart', t0, 'tFinal', t1, 'R0', zonotope(boxes{b}), 'U', zonotope(0));
    tr = tic;
    R = reach(sys, params, options);
    reachTime = reachTime + toc(tr);

    tc = tic;
    tEnd = -Inf;
    for r = 1:numel(R)
        if isempty(R(r).timeInterval)
            continue
        end
        for i = 1:numel(R(r).timeInterval.set)
            S = R(r).timeInterval.set{i};
            T = R(r).timeInterval.time{i};
            a = infimum(T);
            e = supremum(T);
            tEnd = max(tEnd, e);
            s = aux_check(S, targetSets, n);
            if strcmp(s, 'inside') && ~meets(b)
                s = 'possible';
            end
            provenInside = provenInside || strcmp(s, 'inside');
            statuses{end + 1} = s; %#ok<AGROW>
            if ~strcmp(s, 'disjoint')
                segments(end + 1, :) = [max(a, t0), min(e, t1)]; %#ok<AGROW>
            end
        end
    end
    checkTime = checkTime + toc(tc);
    reachedUntil = min(reachedUntil, tEnd);
    covered = covered && tEnd >= t1 - 1e-9;
end

if provenInside
    verdict = 'reachable_proven';
elseif ~covered
    verdict = 'incomplete';
elseif all(strcmp(statuses, 'disjoint'))
    verdict = 'unreachable_proven';
else
    verdict = 'reachable_possible';
end

out = struct();
out.verdict = verdict;
out.horizon_covered = covered;
out.reachable_set_end_time = reachedUntil;
out.num_segments = numel(statuses);
out.num_possible = sum(strcmp(statuses, 'possible'));
out.num_inside = sum(strcmp(statuses, 'inside'));
out.reach_time_bounds = aux_merge(segments);
out.initial_set_overapproximated = overInit;
out.num_initial_boxes = numel(boxes);
out.times = struct('warmup', warmup, 'reach', reachTime, 'check', checkTime);
out.versions = struct('matlab', version, 'cora', CORAVERSION());
fid = fopen(outPath, 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
end


% Auxiliary functions -----------------------------------------------------

function s = aux_matlab(e)
% 'x12^2' -> 'x(12)^2'
s = regexprep(strrep(char(e), '**', '^'), '\<x(\d+)\>', 'x($1)');
end

function sets = aux_sets(raw)
% jsondecode gives a struct array for identical fields, else a cell array
if iscell(raw)
    sets = raw;
else
    sets = num2cell(raw);
end
end

function I = aux_box(bounds)
I = interval(bounds(:, 1), bounds(:, 2));
end

function [boxes, meets] = aux_initial_boxes(s, n, k)
switch s.type
    case 'box'
        bb = aux_box(s.bounds);
    case 'ball'
        bb = interval(s.center - s.radius, s.center + s.radius);
    case 'level_set'
        bb = aux_box(s.bounding_box);
end
lo = infimum(bb);
hi = supremum(bb);
counts = ones(n, 1);
counts(hi > lo) = k;
boxes = {};
meets = [];
idx = ones(n, 1);
while true
    blo = lo + (idx - 1) .* (hi - lo) ./ counts;
    bhi = lo + idx .* (hi - lo) ./ counts;
    B = interval(blo, bhi);
    switch s.type
        case 'box'
            boxes{end + 1} = B; %#ok<AGROW>
            meets(end + 1) = true; %#ok<AGROW>
        case 'ball'
            c = s.center(:);
            if norm(max(max(blo - c, 0), c - bhi)) <= s.radius
                boxes{end + 1} = B; %#ok<AGROW>
                meets(end + 1) = true; %#ok<AGROW>
            end
        case 'level_set'
            g = str2func(['@(x) ' aux_matlab(s.function)]);
            v = g(B);
            if infimum(v) <= 0
                boxes{end + 1} = B; %#ok<AGROW>
                meets(end + 1) = supremum(v) <= 0; %#ok<AGROW>
            end
    end
    % next multi-index
    d = 1;
    while d <= n && idx(d) == counts(d)
        idx(d) = 1;
        d = d + 1;
    end
    if d > n
        break
    end
    idx(d) = idx(d) + 1;
end
end

function s = aux_check(S, targets, n)
statuses = cell(1, numel(targets));
H = interval(S);
for k = 1:numel(targets)
    t = targets{k};
    switch t.type
        case 'box'
            T = aux_box(t.bounds);
            if ~isIntersecting(S, T)
                statuses{k} = 'disjoint';
            elseif contains(T, H)
                statuses{k} = 'inside';
            else
                statuses{k} = 'possible';
            end
        case 'ball'
            c = t.center(:);
            lo = infimum(H);
            hi = supremum(H);
            nearest = norm(max(max(lo - c, 0), c - hi));
            farthest = norm(max(abs(lo - c), abs(hi - c)));
            if nearest > t.radius
                statuses{k} = 'disjoint';
            elseif farthest <= t.radius
                statuses{k} = 'inside';
            else
                statuses{k} = 'possible';
            end
        case 'level_set'
            bb = aux_box(t.bounding_box);
            if ~isIntersecting(H, bb)
                statuses{k} = 'disjoint';
                continue
            end
            g = str2func(['@(x) ' aux_matlab(t.function)]);
            v = g(and(H, bb));
            if infimum(v) > 0
                statuses{k} = 'disjoint';
            elseif supremum(v) <= 0 && contains(bb, H)
                statuses{k} = 'inside';
            else
                statuses{k} = 'possible';
            end
    end
end
if any(strcmp(statuses, 'inside'))
    s = 'inside';
elseif all(strcmp(statuses, 'disjoint'))
    s = 'disjoint';
else
    s = 'possible';
end
end

function m = aux_merge(seg)
m = zeros(0, 2);
if isempty(seg)
    return
end
seg = sortrows(seg);
m = seg(1, :);
for i = 2:size(seg, 1)
    if seg(i, 1) <= m(end, 2)
        m(end, 2) = max(m(end, 2), seg(i, 2));
    else
        m(end + 1, :) = seg(i, :); %#ok<AGROW>
    end
end
end
