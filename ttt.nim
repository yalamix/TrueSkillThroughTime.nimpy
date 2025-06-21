import math
import nimpy
import nimpy/[py_types, py_utils]  # For PyObject handling
import algorithm
import strformat

# Constants - we'll create getter functions instead
const BETA = 1.0
const MU = 0.0
const SIGMA = BETA * 6.0
const GAMMA = BETA * 0.03
const P_DRAW = 0.0
const EPSILON = 1e-6
const ITERATIONS = 30.0

let sqrt2 = sqrt(2.0)
let sqrt2pi = sqrt(2.0 * PI)
let inf = Inf

# Use pow for negative exponents
let PI_val = 1.0 / (SIGMA * SIGMA)
let TAU_val = PI_val * MU

# Getter functions for constants
proc get_BETA*(): float {.exportpy.} = BETA
proc get_MU*(): float {.exportpy.} = MU
proc get_SIGMA*(): float {.exportpy.} = SIGMA
proc get_GAMMA*(): float {.exportpy.} = GAMMA
proc get_P_DRAW*(): float {.exportpy.} = P_DRAW
proc get_EPSILON*(): float {.exportpy.} = EPSILON
proc get_ITERATIONS*(): float {.exportpy.} = ITERATIONS
proc get_sqrt2*(): float {.exportpy.} = sqrt2
proc get_sqrt2pi*(): float {.exportpy.} = sqrt2pi
proc get_inf*(): float {.exportpy.} = inf
proc get_PI*(): float {.exportpy.} = PI_val
proc get_TAU*(): float {.exportpy.} = TAU_val

# Erfc function
proc erfc*(x: float): float {.exportpy.} =
  let z = abs(x)
  let t = 1.0 / (1.0 + z / 2.0)
  
  let a = -0.82215223 + t * 0.17087277
  let b = 1.48851587 + t * a
  let c = -1.13520398 + t * b
  let d = 0.27886807 + t * c
  let e = -0.18628806 + t * d
  let f = 0.09678418 + t * e
  let g = 0.37409196 + t * f
  let h = 1.00002368 + t * g
  
  let r = t * exp(-z*z - 1.26551223 + t*h)
  result = if x >= 0: r else: 2.0 - r

# Inverse complementary error function
proc erfcinv*(y: float): float {.exportpy.} =
  if y >= 2: 
    return -inf
  if y < 0: 
    raise newException(ValueError, "argument must be nonnegative")
  if y == 0: 
    return inf
  
  var y_local = y
  if not (y_local < 1): 
    y_local = 2 - y_local
  
  var t = sqrt(-2 * ln(y_local / 2.0))
  var x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
  
  for _ in 0..2:
    let err = erfc(x) - y_local
    x += err / (1.12837916709551257 * exp(-(x*x)) - x * err)
  
  result = if y_local < 1: x else: -x

# Tau and Pi conversion
proc tau_pi*(mu, sigma: float): tuple[tau, pi: float] {.exportpy.} =
  if sigma > 0.0:
    result.pi = 1.0 / (sigma * sigma)
    result.tau = result.pi * mu
  elif sigma + 1e-5 < 0.0:
    raise newException(ValueError, "sigma should be >= 0")
  else:
    result.pi = inf
    result.tau = inf

# Mu and Sigma conversion
proc mu_sigma*(tau, pi: float): tuple[mu, sigma: float] {.exportpy.} =
  if pi > 0.0:
    result.sigma = sqrt(1.0/pi)
    result.mu = tau / pi
  elif pi + 1e-5 < 0.0:
    raise newException(ValueError, "pi should be >= 0")
  else:
    result.sigma = inf
    result.mu = 0.0

# Cumulative distribution function
proc cdf*(x, mu, sigma: float): float {.exportpy.} =
  let z = -(x - mu) / (sigma * sqrt2)
  0.5 * erfc(z)

# Probability density function
proc pdf*(x, mu, sigma: float): float {.exportpy.} =
  let normalizer = 1.0 / (sqrt2pi * sigma)
  let functional = exp(-pow(x - mu, 2) / (2.0*pow(sigma, 2)))
  normalizer * functional

# Percent point function (inverse CDF)
proc ppf*(p, mu, sigma: float): float {.exportpy.} =
  mu - sigma * sqrt2 * erfcinv(2.0 * p)

# V and W helper function
proc v_w*(mu, sigma, margin: float, tie: bool): tuple[v, w: float] {.exportpy.} =
  if not tie:
    let alpha = (margin - mu)/sigma
    let v = pdf(-alpha, 0.0, 1.0) / cdf(-alpha, 0.0, 1.0)
    result.v = v
    result.w = v * (v - alpha)
  else:
    let alpha = (-margin - mu)/sigma
    let beta = (margin - mu)/sigma
    result.v = (pdf(alpha, 0.0, 1.0) - pdf(beta, 0.0, 1.0)) / 
               (cdf(beta, 0.0, 1.0) - cdf(alpha, 0.0, 1.0))
    let u = (alpha*pdf(alpha, 0.0, 1.0) - beta*pdf(beta, 0.0, 1.0)) / 
            (cdf(beta, 0.0, 1.0) - cdf(alpha, 0.0, 1.0))
    result.w = -(u - pow(result.v, 2))

# Truncation function
proc trunc*(mu, sigma, margin: float, tie: bool): tuple[mu_trunc, sigma_trunc: float] {.exportpy.} =
  let (v, w) = v_w(mu, sigma, margin, tie)
  result.mu_trunc = mu + sigma * v
  result.sigma_trunc = sigma * sqrt(1.0 - w)

# Instead of a class, represent Gaussian as a tuple
type GaussianTuple* = tuple[mu: float, sigma: float]

# Helper to create a Gaussian tuple
proc gaussian*(mu: float = get_MU(), sigma: float = get_SIGMA()): GaussianTuple {.exportpy.} =
    if sigma < 0.0:
        raise newException(ValueError, "sigma must be >= 0.0")
    (mu, sigma)

# Gaussian operations as pure functions
proc gaussian_add*(a, b: GaussianTuple): GaussianTuple {.exportpy.} = 
    let new_mu = a.mu + b.mu
    let new_sigma = sqrt(a.sigma*a.sigma + b.sigma*b.sigma)
    (new_mu, new_sigma)

proc gaussian_sub*(a, b: GaussianTuple): GaussianTuple {.exportpy.} = 
    let new_mu = a.mu - b.mu
    let new_sigma = sqrt(a.sigma*a.sigma + b.sigma*b.sigma)
    (new_mu, new_sigma)

proc gaussian_mul*(a, b: GaussianTuple): GaussianTuple {.exportpy.} = 
    let (tau1, pi1) = tau_pi(a.mu, a.sigma)
    let (tau2, pi2) = tau_pi(b.mu, b.sigma)
    let tau = tau1 + tau2
    let pi = pi1 + pi2
    let (mu, sigma) = mu_sigma(tau, pi)
    (mu, sigma)

proc gaussian_div*(a, b: GaussianTuple): GaussianTuple {.exportpy.} = 
    let (tau1, pi1) = tau_pi(a.mu, a.sigma)
    let (tau2, pi2) = tau_pi(b.mu, b.sigma)
    let tau = tau1 - tau2
    let pi = pi1 - pi2
    let (mu, sigma) = mu_sigma(tau, pi)
    (mu, sigma)

proc gaussian_scale*(a: GaussianTuple, k: float): GaussianTuple {.exportpy.} = 
    (k * a.mu, abs(k) * a.sigma)

proc gaussian_forget*(a: GaussianTuple, gamma: float, t: int): GaussianTuple {.exportpy.} = 
    (a.mu, sqrt(a.sigma*a.sigma + float(t)*gamma*gamma))

proc gaussian_delta*(a, b: GaussianTuple): tuple[a: float, b: float] {.exportpy.} = 
    (abs(a.mu - b.mu), abs(a.sigma - b.sigma))

proc gaussian_exclude*(a, b: GaussianTuple): GaussianTuple {.exportpy.} = 
    let new_sigma_sq = a.sigma*a.sigma - b.sigma*b.sigma
    if new_sigma_sq < 0.0:
        raise newException(ValueError, "Resulting sigma would be imaginary")
    (a.mu - b.mu, sqrt(new_sigma_sq))

proc isapprox*(a, b: GaussianTuple, tol: float = 1e-4): bool {.exportpy.} =
    (abs(a.mu - b.mu) < tol) and (abs(a.sigma - b.sigma) < tol)

proc approx*(n: GaussianTuple, margin: float, tie: bool): GaussianTuple {.exportpy.} =
    let (mu, sigma) = trunc(n.mu, n.sigma, margin, tie)
    (mu, sigma)

# Constants as functions
proc N00*(): GaussianTuple {.exportpy.} = (0.0, 0.0)
proc N01*(): GaussianTuple {.exportpy.} = (0.0, 1.0)
proc Ninf*(): GaussianTuple {.exportpy.} = (0.0, inf)
proc Nms*(): GaussianTuple {.exportpy.} = (get_MU(), get_SIGMA())

# Other utility functions
proc compute_margin*(p_draw, sd: float): float {.exportpy.} =
    abs(ppf(0.5 - p_draw/2.0, 0.0, sd))

proc max_tuple*(t1, t2: tuple[a, b: float]): tuple[a, b: float] {.exportpy.} =
    (max(t1.a, t2.a), max(t1.b, t2.b))

proc gr_tuple*(tup: tuple[a, b: float], threshold: float): bool {.exportpy.} =
    tup.a > threshold or tup.b > threshold

proc sortperm*(xs: seq[float], reverse: bool = false): seq[int] {.exportpy.} =
    var indexed = newSeq[(float, int)](xs.len)
    for i, x in xs:
        indexed[i] = (x, i)
    
    indexed.sort(proc (a, b: (float, int)): int =
        if a[0] < b[0]: -1
        elif a[0] > b[0]: 1
        else: 0
    )
    
    if reverse:
        indexed.reverse()
    
    result = newSeq[int](xs.len)
    for i in 0..<indexed.len:
        result[i] = indexed[i][1]

proc podium*(xs: seq[float]): seq[int] {.exportpy.} =
    sortperm(xs)

proc dict_diff*(old_dict: PyObject, new_dict: PyObject): tuple[a, b: float] {.exportpy.} =
    var step = (0.0, 0.0)
    for key in old_dict:
        # Convert PyObject to GaussianTuple
        let g_old = old_dict[key].to(GaussianTuple)
        let g_new = new_dict[key].to(GaussianTuple)
        let d = gaussian_delta(g_old, g_new)
        step = max_tuple(step, d)
    result = step

type Player* = tuple
    prior: GaussianTuple
    beta: float
    gamma: float
    prior_draw: GaussianTuple

proc newPlayer*(
    prior: GaussianTuple = Nms(), 
    beta: float = get_BETA(), 
    gamma: float = get_GAMMA(), 
    prior_draw: GaussianTuple = Ninf()
): Player {.exportpy.} =
    (prior: prior, beta: beta, gamma: gamma, prior_draw: prior_draw)

# For a single player
proc player_performance*(player: Player): GaussianTuple {.exportpy.} =
    let (mu, sigma) = player.prior
    let new_sigma = sqrt(sigma*sigma + player.beta*player.beta)
    (mu, new_sigma)

proc reprPlayer*(player: Player): string {.exportpy.} =
    let (mu, sigma) = player.prior
    fmt"Player(prior=Gaussian(mu={mu:.3f}, sigma={sigma:.3f}), beta={player.beta}, gamma={player.gamma})"
  
type TeamVariable* = tuple
    prior: GaussianTuple
    likelihood_lose: GaussianTuple
    likelihood_win: GaussianTuple
    likelihood_draw: GaussianTuple

proc newTeamVariable*(
    prior: GaussianTuple = Ninf(),
    likelihood_lose: GaussianTuple = Ninf(),
    likelihood_win: GaussianTuple = Ninf(),
    likelihood_draw: GaussianTuple = Ninf()
): TeamVariable {.exportpy.} =
    (prior: prior, 
     likelihood_lose: likelihood_lose,
     likelihood_win: likelihood_win,
     likelihood_draw: likelihood_draw)

# Team Variable operations
proc team_variable_p*(tv: TeamVariable): GaussianTuple {.exportpy.} =
    var res = tv.prior
    res = gaussian_mul(res, tv.likelihood_lose)
    res = gaussian_mul(res, tv.likelihood_win)
    res = gaussian_mul(res, tv.likelihood_draw)
    res

proc team_variable_posterior_win*(tv: TeamVariable): GaussianTuple {.exportpy.} =
    var res = tv.prior
    res = gaussian_mul(res, tv.likelihood_lose)
    res = gaussian_mul(res, tv.likelihood_draw)
    res

proc team_variable_posterior_lose*(tv: TeamVariable): GaussianTuple {.exportpy.} =
    var res = tv.prior
    res = gaussian_mul(res, tv.likelihood_win)
    res = gaussian_mul(res, tv.likelihood_draw)
    res

proc team_variable_likelihood*(tv: TeamVariable): GaussianTuple {.exportpy.} =
    var res = tv.likelihood_win
    res = gaussian_mul(res, tv.likelihood_lose)
    res = gaussian_mul(res, tv.likelihood_draw)
    res

# For a team
proc team_performance*(team: seq[Player], weights: seq[float]): GaussianTuple {.exportpy.} =
    var res = N00()
    for i in 0..<team.len:
        let perf = player_performance(team[i])
        let scaled = gaussian_scale(perf, weights[i])
        res = gaussian_add(res, scaled)
    res

type DrawMessages* = tuple
    prior: GaussianTuple
    prior_team: GaussianTuple
    likelihood_lose: GaussianTuple
    likelihood_win: GaussianTuple

proc newDrawMessages*(
    prior: GaussianTuple = Ninf(),
    prior_team: GaussianTuple = Ninf(),
    likelihood_lose: GaussianTuple = Ninf(),
    likelihood_win: GaussianTuple = Ninf()
): DrawMessages {.exportpy.} =
    (prior: prior, 
     prior_team: prior_team,
     likelihood_lose: likelihood_lose,
     likelihood_win: likelihood_win)

# Draw Messages operations
proc draw_messages_p*(dm: DrawMessages): GaussianTuple {.exportpy.} =
    var res = dm.prior_team
    res = gaussian_mul(res, dm.likelihood_lose)
    res = gaussian_mul(res, dm.likelihood_win)
    res

proc draw_messages_posterior_win*(dm: DrawMessages): GaussianTuple {.exportpy.} =
    gaussian_mul(dm.prior_team, dm.likelihood_lose)

proc draw_messages_posterior_lose*(dm: DrawMessages): GaussianTuple {.exportpy.} =
    gaussian_mul(dm.prior_team, dm.likelihood_win)

proc draw_messages_likelihood*(dm: DrawMessages): GaussianTuple {.exportpy.} =
    gaussian_mul(dm.likelihood_win, dm.likelihood_lose)

type DiffMessages* = tuple
    prior: GaussianTuple
    likelihood: GaussianTuple

proc newDiffMessages*(
    prior: GaussianTuple = Ninf(),
    likelihood: GaussianTuple = Ninf()
): DiffMessages {.exportpy.} =
    (prior: prior, likelihood: likelihood)

# Diff Messages operations
proc diff_messages_p*(dm: DiffMessages): GaussianTuple {.exportpy.} =
    gaussian_mul(dm.prior, dm.likelihood)