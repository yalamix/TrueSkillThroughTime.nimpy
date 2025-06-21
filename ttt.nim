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

# Integer version of sortperm
proc sortperm_int*(xs: seq[int], reverse: bool = false): seq[int] {.exportpy.} =
    var indexed = newSeq[(int, int)](xs.len)
    for i, x in xs:
        indexed[i] = (x, i)
    
    indexed.sort(proc (a, b: (int, int)): int =
        if a[0] < b[0]: -1
        elif a[0] > b[0]: 1
        else: 0
    )
    
    if reverse:
        indexed.reverse()
    
    result = newSeq[int](xs.len)
    for i in 0..<indexed.len:
        result[i] = indexed[i][1]

# Float version of sortperm (for podium function)
proc sortperm_float*(xs: seq[float], reverse: bool = false): seq[int] {.exportpy.} =
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

# Update podium to use float version
proc podium*(xs: seq[float]): seq[int] {.exportpy.} =
    sortperm_float(xs)

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

type Game* = tuple
    teams: seq[seq[Player]]
    game_result: seq[int]
    p_draw: float
    weights: seq[seq[float]]

proc newGame*(
    teams: seq[seq[Player]], 
    game_result: seq[int] = @[],  # Renamed from 'result'
    p_draw: float = 0.0, 
    weights: seq[seq[float]] = @[]
): Game {.exportpy.} =
    # Default game_result if not provided
    var actual_result = game_result
    if len(actual_result) == 0:
        actual_result = newSeq[int](len(teams))
        for i in 0..<len(teams):
            actual_result[i] = len(teams) - i - 1
    
    # Default weights if not provided
    var actual_weights = weights
    if len(actual_weights) == 0:
        actual_weights = newSeq[seq[float]](len(teams))
        for i in 0..<len(teams):
            actual_weights[i] = newSeq[float](len(teams[i]))
            for j in 0..<len(teams[i]):
                actual_weights[i][j] = 1.0
    
    (teams: teams, game_result: actual_result, p_draw: p_draw, weights: actual_weights)

proc game_len*(game: Game): int {.exportpy.} =
    len(game.teams)

proc game_size*(game: Game): seq[int] {.exportpy.} =
    result = newSeq[int](len(game.teams))
    for i in 0..<len(game.teams):
        result[i] = len(game.teams[i])

proc performance*(game: Game, i: int): GaussianTuple {.exportpy.} =
    team_performance(game.teams[i], game.weights[i])

proc graphical_model*(game: Game): tuple[
    o: seq[int], 
    t: seq[TeamVariable], 
    d: seq[DiffMessages], 
    tie: seq[bool], 
    margin: seq[float],
    evidence: float
] {.exportpy.} =
    var o = sortperm_int(game.game_result, reverse = true)
    var t = newSeq[TeamVariable]()
    for e in 0..<game_len(game):
        t.add(newTeamVariable(prior=team_performance(game.teams[o[e]], game.weights[o[e]])))
    
    var d = newSeq[DiffMessages]()
    for e in 0..<len(t)-1:
        let diff = gaussian_sub(t[e].prior, t[e+1].prior)
        d.add(newDiffMessages(prior=diff))
    
    var tie = newSeq[bool]()
    for e in 0..<len(d):
        tie.add(game.game_result[o[e]] == game.game_result[o[e+1]])
    
    var margin = newSeq[float]()
    for e in 0..<len(d):
        if game.p_draw == 0.0:
            margin.add(0.0)
        else:
            var sd = 0.0
            for player in game.teams[o[e]]:
                sd += player.beta * player.beta
            for player in game.teams[o[e+1]]:
                sd += player.beta * player.beta
            sd = sqrt(sd)
            margin.add(compute_margin(game.p_draw, sd))
    
    # Correct evidence calculation
    var evidence = 1.0
    for e in 0..<len(d):
        let (mu, sigma) = d[e].prior
        if tie[e]:
            evidence *= (cdf(margin[e], mu, sigma) - cdf(-margin[e], mu, sigma))
        else:
            evidence *= (1.0 - cdf(margin[e], mu, sigma))
    
    result = (o: o, t: t, d: d, tie: tie, margin: margin, evidence: evidence)

proc partial_evidence*(
    d: GaussianTuple, 
    margin: float, 
    tie: bool, 
    evidence: float
): float {.exportpy.} =
    let (mu, sigma) = d
    if tie:
        evidence * (cdf(margin, mu, sigma) - cdf(-margin, mu, sigma))
    else:
        evidence * (1.0 - cdf(margin, mu, sigma))

proc likelihood_analitico*(game: Game): tuple[likelihoods: seq[seq[GaussianTuple]], evidence: float] {.exportpy.} =
    let (o, t, d, tie, margin, evidence) = graphical_model(game)
    
    # The evidence is already calculated, so we can use it directly
    var partial_ev = evidence
    
    # Rest of the implementation remains the same...
    let d0 = d[0].prior
    let (mu_trunc, sigma_trunc) = trunc(d0.mu, d0.sigma, margin[0], tie[0])
    
    # Calculate the analytical solution for each player
    var likelihoods = newSeq[seq[GaussianTuple]]()
    for i in 0..<game_len(game):
        var team_likelihoods = newSeq[GaussianTuple]()
        for j in 0..<len(game.teams[i]):
            let player = game.teams[o[i]][j]
            let w = game.weights[o[i]][j]
            
            # Calculate the new Gaussian parameters
            var new_mu: float
            var new_sigma: float
            
            if d0.sigma == sigma_trunc:
                new_mu = player.prior.mu
                new_sigma = player.prior.sigma
            else:
                # Calculate the parameters for the analytical solution
                let delta_div = (d0.sigma*d0.sigma*mu_trunc - sigma_trunc*sigma_trunc*d0.mu) /
                                (d0.sigma*d0.sigma - sigma_trunc*sigma_trunc)
                let theta_div_pow2 = (sigma_trunc*sigma_trunc * d0.sigma*d0.sigma) /
                                     (d0.sigma*d0.sigma - sigma_trunc*sigma_trunc)
                
                new_mu = player.prior.mu + (delta_div - d0.mu) * pow(-1.0, float(i == 1))
                new_sigma = sqrt(theta_div_pow2 + d0.sigma*d0.sigma - player.prior.sigma*player.prior.sigma)
            
            team_likelihoods.add((new_mu, new_sigma))
        likelihoods.add(team_likelihoods)
    
    # Reorder likelihoods if needed
    if o[0] > o[1]:
        swap(likelihoods[0], likelihoods[1])
    
    (likelihoods: likelihoods, evidence: partial_ev)

proc likelihood_teams*(game: Game): tuple[likelihoods: seq[seq[GaussianTuple]], evidence: float] {.exportpy.} =
    let (o, t, d, tie, margin, evidence) = graphical_model(game)
    
    var current_t = t
    var current_d = d
    var step = (inf, inf)
    var i = 0
    let max_iter = 10
    let epsilon = 1e-6
    
    while gr_tuple(step, epsilon) and i < max_iter:
        step = (0.0, 0.0)
        
        # Forward pass
        for e in 0..<len(current_d)-1:
            # Update difference prior
            let post_win = team_variable_posterior_win(current_t[e])
            let post_lose = team_variable_posterior_lose(current_t[e+1])
            let new_prior = gaussian_sub(post_win, post_lose)
            
            # Update difference message
            var new_d_e = current_d[e]
            new_d_e.prior = new_prior
            
            # Compute likelihood
            let approx_gauss = approx(new_prior, margin[e], tie[e])
            let likelihood_diff = gaussian_div(approx_gauss, new_prior)
            new_d_e.likelihood = likelihood_diff
            
            # Compute new likelihood_lose for next team
            let new_likelihood_lose = gaussian_sub(post_win, likelihood_diff)
            
            # Update step
            let delta = gaussian_delta(new_likelihood_lose, current_t[e+1].likelihood_lose)
            step = max_tuple(step, delta)
            
            # Update team variable
            var new_t_e1 = current_t[e+1]
            new_t_e1.likelihood_lose = new_likelihood_lose
            current_t[e+1] = new_t_e1
            current_d[e] = new_d_e
        
        # Backward pass
        for e in countdown(len(current_d)-1, 1):
            # Update difference prior
            let post_win = team_variable_posterior_win(current_t[e])
            let post_lose = team_variable_posterior_lose(current_t[e+1])
            let new_prior = gaussian_sub(post_win, post_lose)
            
            # Update difference message
            var new_d_e = current_d[e]
            new_d_e.prior = new_prior
            
            # Compute likelihood
            let approx_gauss = approx(new_prior, margin[e], tie[e])
            let likelihood_diff = gaussian_div(approx_gauss, new_prior)
            new_d_e.likelihood = likelihood_diff
            
            # Compute new likelihood_win for previous team
            let new_likelihood_win = gaussian_add(post_lose, likelihood_diff)
            
            # Update step
            let delta = gaussian_delta(new_likelihood_win, current_t[e].likelihood_win)
            step = max_tuple(step, delta)
            
            # Update team variable
            var new_t_e = current_t[e]
            new_t_e.likelihood_win = new_likelihood_win
            current_t[e] = new_t_e
            current_d[e] = new_d_e
        
        i += 1
    
    # Handle the last difference if it exists
    if len(current_d) > 0:
        let e = 0
        let post_win = team_variable_posterior_win(current_t[e])
        let post_lose = team_variable_posterior_lose(current_t[e+1])
        let new_prior = gaussian_sub(post_win, post_lose)
        
        var new_d_e = current_d[e]
        new_d_e.prior = new_prior
        
        let approx_gauss = approx(new_prior, margin[e], tie[e])
        let likelihood_diff = gaussian_div(approx_gauss, new_prior)
        new_d_e.likelihood = likelihood_diff
        
        # Update first team's likelihood_win
        let new_likelihood_win = gaussian_add(post_lose, likelihood_diff)
        var new_t0 = current_t[0]
        new_t0.likelihood_win = new_likelihood_win
        current_t[0] = new_t0
        
        # Update last team's likelihood_lose
        let last_index = len(current_t)-1
        let last_d_index = len(current_d)-1
        let last_post_win = team_variable_posterior_win(current_t[last_index-1])
        let last_likelihood_lose = gaussian_sub(last_post_win, current_d[last_d_index].likelihood)
        var new_t_last = current_t[last_index]
        new_t_last.likelihood_lose = last_likelihood_lose
        current_t[last_index] = new_t_last
    
    # Compute likelihoods for each player
    var likelihoods = newSeq[seq[GaussianTuple]]()
    for e in 0..<game_len(game):
        let team_likelihood = team_variable_likelihood(current_t[e])
        var player_likelihoods = newSeq[GaussianTuple]()
        
        let team_perf = team_performance(game.teams[o[e]], game.weights[o[e]])
        for i in 0..<len(game.teams[o[e]]):
            let player = game.teams[o[e]][i]
            let w = game.weights[o[e]][i]
            let player_perf = player_performance(player)
            let scaled_player_perf = gaussian_scale(player_perf, w)
            let other_perf = gaussian_exclude(team_perf, scaled_player_perf)
            let player_team_likelihood = gaussian_div(team_likelihood, other_perf)
            let player_likelihood = gaussian_scale(player_team_likelihood, 1.0 / w)
            player_likelihoods.add(player_likelihood)
        
        likelihoods.add(player_likelihoods)
    
    (likelihoods: likelihoods, evidence: evidence)

proc compute_likelihoods*(game: Game): tuple[likelihoods: seq[seq[GaussianTuple]], evidence: float] {.exportpy.} =
    # Check if we can use the analytical method
    if game_len(game) == 2:
        var all_weights_one = true
        for team_weights in game.weights:
            for w in team_weights:
                if w != 1.0:
                    all_weights_one = false
                    break
            if not all_weights_one:
                break
        
        if all_weights_one:
            return likelihood_analitico(game)
    
    # Otherwise use the team-based method
    return likelihood_teams(game)

proc posteriors*(game: Game, likelihoods: seq[seq[GaussianTuple]]): seq[seq[GaussianTuple]] {.exportpy.} =
    var posteriors = newSeq[seq[GaussianTuple]]()
    for e in 0..<len(game.teams):
        var team_posteriors = newSeq[GaussianTuple]()
        for i in 0..<len(game.teams[e]):
            let prior = game.teams[e][i].prior
            let likelihood = likelihoods[e][i]
            team_posteriors.add(gaussian_mul(prior, likelihood))
        posteriors.add(team_posteriors)
    posteriors