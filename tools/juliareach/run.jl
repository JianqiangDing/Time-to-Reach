# JuliaReach (ReachabilityAnalysis.jl, TMJets) on one benchmark.
#
#   julia --project=tools/juliareach tools/juliareach/run.jl <config.json> <settings.json> <out.json>
#
# Called by tools/juliareach/run.py, which turns the output into a result record.
# The vector field is generated from the config's equation strings. Every
# flowpipe segment is enclosed in a box and compared with each target set:
#   disjoint  the segment's states are certainly outside the target
#   inside    the segment's states are certainly inside the target
#   possible  otherwise
# Level-set targets are decided by interval evaluation of their function on the
# box. Initial balls and level sets are over-approximated by boxes (sound for
# 'disjoint'; 'inside' still proves reachability because every trajectory from
# the box is then in the target).

using ReachabilityAnalysis, LazySets, JSON3
using IntervalArithmetic: interval, inf, sup
using Pkg

const cfg = JSON3.read(read(ARGS[1], String))
const st = JSON3.read(read(ARGS[2], String))
const out_path = ARGS[3]

const n = length(cfg.system.equations)
const backward = get(cfg, :direction, "forward") == "backward"

julia_expr(s) = replace(replace(String(s), "**" => "^"), r"\bx(\d+)\b" => s"x[\1]")

# --- vector field -----------------------------------------------------------
let lines = ["dx[$i] = " * (backward ? "-(" : "(") * julia_expr(e) * ")" for (i, e) in enumerate(cfg.system.equations)]
    body = join(lines, "\n    ")
    macro_ = get(st, :taylorize, true) ? "@taylorize " : ""
    eval(Meta.parse("""begin
    $(macro_)function field!(dx, x, p, t)
        $body
        return dx
    end
    end"""))
end

# --- sets -------------------------------------------------------------------
box_of(bounds) = Hyperrectangle(low=Float64[b[1] for b in bounds], high=Float64[b[2] for b in bounds])

# Initial boxes for one initial set: its bounding box split into `k` parts per
# dimension, keeping the parts that may meet the set. Each part comes with a flag
# that is true when it certainly meets the set (needed to prove reachability).
function initial_boxes(s, k)
    bb = s.type == "box" ? box_of(s.bounds) :
         s.type == "ball" ? Hyperrectangle(Float64.(collect(s.center)), fill(Float64(s.radius), n)) :
         s.type == "level_set" ? box_of(s.bounding_box) : error("unknown set type $(s.type)")
    edges = [range(low(bb, i), high(bb, i); length=(high(bb, i) > low(bb, i) ? k + 1 : 2)) for i in 1:n]
    parts = Tuple{Hyperrectangle{Float64,Vector{Float64},Vector{Float64}},Bool}[]
    for idx in Iterators.product((1:length(e)-1 for e in edges)...)
        lo = [edges[i][idx[i]] for i in 1:n]
        hi = [edges[i][idx[i]+1] for i in 1:n]
        B = Hyperrectangle(low=lo, high=hi)
        if s.type == "box"
            push!(parts, (B, true))
        elseif s.type == "ball"
            c, r = Float64.(collect(s.center)), Float64(s.radius)
            sqrt(sum(max.(lo .- c, 0.0, c .- hi) .^ 2)) <= r && push!(parts, (B, true))
        else
            f = eval(Meta.parse("x -> " * julia_expr(s.function)))
            v = Base.invokelatest(f, [interval(lo[i], hi[i]) for i in 1:n])
            inf(v) <= 0 && push!(parts, (B, sup(v) <= 0))
        end
    end
    return parts, s.type != "box"
end

# A target is a function box -> :disjoint / :inside / :possible.
function target_check(s)
    if s.type == "box"
        T = box_of(s.bounds)
        return B -> isdisjoint(B, T) ? :disjoint : (issubset(B, T) ? :inside : :possible)
    elseif s.type == "ball"
        c, r = Float64.(collect(s.center)), Float64(s.radius)
        return function (B)
            lo, hi = low(B), high(B)
            nearest = sqrt(sum(max.(lo .- c, 0.0, c .- hi) .^ 2))  # distance from c to the box
            farthest = sqrt(sum(max.(abs.(lo .- c), abs.(hi .- c)) .^ 2))  # farthest box vertex
            nearest > r ? :disjoint : (farthest <= r ? :inside : :possible)
        end
    elseif s.type == "level_set"
        bb = box_of(s.bounding_box)
        f = eval(Meta.parse("x -> " * julia_expr(s.function)))
        return function (B)
            isdisjoint(B, bb) && return :disjoint
            C = intersection(B, bb)
            X = [interval(low(C, i), high(C, i)) for i in 1:n]
            v = Base.invokelatest(f, X)
            inf(v) > 0 && return :disjoint
            (sup(v) <= 0 && issubset(B, bb)) && return :inside
            return :possible
        end
    end
    error("unknown set type $(s.type)")
end

combine(statuses) = any(==(:inside), statuses) ? :inside : (all(==(:disjoint), statuses) ? :disjoint : :possible)

# --- analysis ---------------------------------------------------------------
const t0, t1 = Float64(cfg.time_horizon[1]), Float64(cfg.time_horizon[2])
const targets = [target_check(s) for s in cfg.target_sets]
const alg = TMJets(abstol=st.abstol, orderT=st.orderT, orderQ=st.orderQ, maxsteps=st.maxsteps)

function solve_from(X0, T)
    prob = @ivp(x' = field!(x), dim: n, x(0) ∈ X0)
    return solve(prob, T=T, alg=alg)
end

const splits = get(st, :initial_splits, 1)
boxes0 = Tuple{Hyperrectangle{Float64,Vector{Float64},Vector{Float64}},Bool}[]
overapprox_initial = false
for s in cfg.initial_sets
    global overapprox_initial
    parts, over = initial_boxes(s, splits)
    append!(boxes0, parts)
    overapprox_initial |= over
end

# warm-up on a tiny horizon so that compilation is not part of the timed solve
warmup = @elapsed Base.invokelatest(solve_from, first(boxes0[1]), min(t1, 1e-3))

statuses = Symbol[]
segments = Tuple{Float64,Float64}[]
proven_inside = false
covered = true
reached_until = Inf
solve_time = 0.0
check_time = 0.0
for (X0, meets_initial_set) in boxes0
    global solve_time, check_time, covered, reached_until, proven_inside
    sol = nothing
    solve_time += @elapsed (sol = Base.invokelatest(solve_from, X0, t1))
    t_end = tend(sol)
    reached_until = min(reached_until, t_end)
    covered &= t_end >= t1 - 1e-9
    check_time += @elapsed for R in sol
        a, b = tstart(R), tend(R)
        b < t0 && continue
        B = set(overapproximate(R, Hyperrectangle))
        s = combine([Base.invokelatest(chk, B) for chk in targets])
        # 'inside' proves reachability only for parts that certainly meet the initial set
        s == :inside && !meets_initial_set && (s = :possible)
        proven_inside |= s == :inside
        push!(statuses, s)
        s != :disjoint && push!(segments, (max(a, t0), min(b, t1)))
    end
end

verdict = if proven_inside
    "reachable_proven"
elseif !covered
    "incomplete"
elseif all(==(:disjoint), statuses)
    "unreachable_proven"
else
    "reachable_possible"
end

function merged(iv)
    out = Tuple{Float64,Float64}[]
    for (a, b) in sort(iv)
        if !isempty(out) && a <= out[end][2]
            out[end] = (out[end][1], max(out[end][2], b))
        else
            push!(out, (a, b))
        end
    end
    out
end

versions = Dict(info.name => string(info.version) for (_, info) in Pkg.dependencies()
                if info.is_direct_dep && info.version !== nothing)
versions["julia"] = string(VERSION)

result = Dict(
    "verdict" => verdict,
    "horizon_covered" => covered,
    "flowpipe_end_time" => reached_until,
    "num_segments" => length(statuses),
    "num_possible" => count(==(:possible), statuses),
    "num_inside" => count(==(:inside), statuses),
    "reach_time_bounds" => [[a, b] for (a, b) in merged(segments)],
    "initial_set_overapproximated" => overapprox_initial,
    "num_initial_boxes" => length(boxes0),
    "times" => Dict("warmup_compile" => warmup, "solve" => solve_time, "check" => check_time),
    "versions" => versions,
)
open(out_path, "w") do io
    JSON3.write(io, result)
end
