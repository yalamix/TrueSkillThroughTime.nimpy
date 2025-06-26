import math
import sets
import nimpy
import tables
import sequtils
import strformat
import algorithm

# Constants
const
  BETA = 1.0
  MU = 0.0
  SIGMA = BETA * 6.0
  GAMMA = BETA * 0.03
  P_DRAW = 0.0
  EPSILON = 1e-6
  ITERATIONS = 30
  SQRT2 = sqrt(2.0)
  SQRT2PI = sqrt(2.0 * PI)
  INF = Inf
  PI_CONSTANT = pow(SIGMA, -2.0)
  TAU_CONSTANT = PI_CONSTANT * MU

# Helper functions
proc erfc(x: float): float =
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
  let r = t * exp(-z * z - 1.26551223 + t * h)
  if x < 0: 2.0 - r else: r

proc erfcinv(y: float): float =
  if y >= 2: return -INF
  if y < 0: raise newException(ValueError, "argument must be nonnegative")
  if y == 0: return INF
  
  var y_work = y
  if y_work >= 1: y_work = 2 - y_work
  
  let t = sqrt(-2 * ln(y_work / 2.0))
  var x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
  
  for i in 0..2:
    let err = erfc(x) - y_work
    x += err / (1.12837916709551257 * exp(-(x * x)) - x * err)
  
  if y < 1: x else: -x

proc tauPi(mu, sigma: float): (float, float) =
  if sigma > 0.0:
    let pi = pow(sigma, -2.0)
    let tau = pi * mu
    (tau, pi)
  elif (sigma + 1e-5) < 0.0:
    raise newException(ValueError, "sigma should be greater than 0")
  else:
    (INF, INF)

proc muSigma(tau, pi: float): (float, float) =
  if pi > 0.0:
    let sigma = sqrt(1.0 / pi)
    let mu = tau / pi
    (mu, sigma)
  elif pi + 1e-5 < 0.0:
    raise newException(ValueError, "sigma should be greater than 0")
  else:
    (0.0, INF)

proc cdf(x: float, mu = 0.0, sigma = 1.0): float =
  let z = -(x - mu) / (sigma * SQRT2)
  0.5 * erfc(z)

proc pdf(x, mu, sigma: float): float =
  let normalizer = pow(SQRT2PI * sigma, -1.0)
  let functional = exp(-pow(x - mu, 2.0) / (2.0 * pow(sigma, 2.0)))
  normalizer * functional

proc ppf(p, mu, sigma: float): float =
  mu - sigma * SQRT2 * erfcinv(2 * p)

proc vW(mu, sigma, margin: float, tie: bool): (float, float) =
  if not tie:
    let alpha = (margin - mu) / sigma
    let cdfVal = cdf(-alpha, 0, 1)
    let pdfVal = pdf(-alpha, 0, 1)
    let v = pdfVal / cdfVal
    let w = v * (v + (-alpha))
    (v, w)
  else:
    let alpha = (-margin-mu) / sigma
    let beta = ( margin-mu) / sigma
    let v = (pdf(alpha,0,1)-pdf(beta,0,1))/(cdf(beta,0,1)-cdf(alpha,0,1))
    let u = (alpha*pdf(alpha,0,1)-beta*pdf(beta,0,1))/(cdf(beta,0,1)-cdf(alpha,0,1))
    let w =  - ( u - v^2 )
    (v, w)

proc trunc(mu, sigma, margin: float, tie: bool): (float, float) =
  let (v, w) = vW(mu, sigma, margin, tie)
  let muTrunc = mu + sigma * v
  let sigmaTrunc = sigma * sqrt(1 - w)  # Ensure non-negative  
  (muTrunc, sigmaTrunc)

proc computeMargin(pDraw, sd: float): float =
  abs(ppf(0.5 - p_draw/2.0, 0.0, sd))

proc maxTuple(t1, t2: (float, float)): (float, float) =
  (max(t1[0], t2[0]), max(t1[1], t2[1]))

proc grTuple(tup: (float, float), threshold: float): bool =
  (tup[0] > threshold) or (tup[1] > threshold)

proc sortPerm[T](xs: seq[T], reverse = false): seq[int] =
  var indexed = newSeq[(T, int)]()
  for i, x in xs:
    indexed.add((x, i))
  
  if reverse:
    indexed.sort(proc(a, b: (T, int)): int = cmp(b[0], a[0]))
  else:
    indexed.sort(proc(a, b: (T, int)): int = cmp(a[0], b[0]))
  
  result = newSeq[int]()
  for (_, i) in indexed:
    result.add(i)

# Gaussian class
type
  Gaussian* = ref object
    mu*: float
    sigma*: float

proc newGaussian*(mu = MU, sigma = SIGMA): Gaussian =
  if sigma >= 0.0:
    Gaussian(mu: mu, sigma: sigma)
  else:
    raise newException(ValueError, "sigma should be greater than 0")

proc tau*(g: Gaussian): float =
  if g.sigma > 0.0:
    g.mu * pow(g.sigma, -2.0)
  else:
    INF

proc pi*(g: Gaussian): float =
  if g.sigma > 0.0:
    pow(g.sigma, -2.0)
  else:
    INF

proc `$`*(g: Gaussian): string =
  fmt"N(mu={g.mu:.3f}, sigma={g.sigma:.3f})"

proc `+`*(g1, g2: Gaussian): Gaussian =
  newGaussian(g1.mu + g2.mu, sqrt(g1.sigma * g1.sigma + g2.sigma * g2.sigma))

proc `-`*(g1, g2: Gaussian): Gaussian =
  newGaussian(g1.mu - g2.mu, sqrt(g1.sigma * g1.sigma + g2.sigma * g2.sigma))

proc `*`*(g: Gaussian, m: float): Gaussian =
  if m == INF:
    newGaussian(0, INF)
  else:
    newGaussian(m * g.mu, abs(m) * g.sigma)

proc `*`*(m: float, g: Gaussian): Gaussian = g * m

proc `*`*(g1, g2: Gaussian): Gaussian =
  if g1.sigma == 0.0 or g2.sigma == 0.0:
    let mu = if g1.sigma == 0.0: g1.mu / ((g1.sigma * g1.sigma / g2.sigma / g2.sigma) + 1) 
             else: g2.mu / ((g2.sigma * g2.sigma / g1.sigma / g1.sigma) + 1)
    newGaussian(mu, 0.0)
  else:
    let tau = g1.tau + g2.tau
    let pi = g1.pi + g2.pi
    let (mu, sigma) = muSigma(tau, pi)
    newGaussian(mu, sigma)

proc `/`*(g1, g2: Gaussian): Gaussian =
  let tau = g1.tau - g2.tau
  let pi = g1.pi - g2.pi
  let (mu, sigma) = muSigma(tau, pi)
  newGaussian(mu, sigma)

proc forget*(g: Gaussian, gamma: float, t: float): Gaussian =
  newGaussian(g.mu, sqrt(g.sigma * g.sigma + t * gamma * gamma))

proc delta*(g1, g2: Gaussian): (float, float) =
  (abs(g1.mu - g2.mu), abs(g1.sigma - g2.sigma))

proc exclude*(g1, g2: Gaussian): Gaussian =
  newGaussian(g1.mu - g2.mu, sqrt(g1.sigma * g1.sigma - g2.sigma * g2.sigma))

proc isApprox*(g1, g2: Gaussian, tol = 1e-4): bool =
  (abs(g1.mu - g2.mu) < tol) and (abs(g1.sigma - g2.sigma) < tol)

# Predefined Gaussians
let N01* = newGaussian(0, 1)
let N00* = newGaussian(0, 0)
let Ninf* = newGaussian(0, INF)
let Nms* = newGaussian(MU, SIGMA)

# Player class
type
  Player* = ref object
    prior*: Gaussian
    beta*: float
    gamma*: float
    priorDraw*: Gaussian

proc newPlayer*(prior = newGaussian(MU, SIGMA), beta = BETA, gamma = GAMMA, priorDraw = Ninf): Player =
  Player(prior: prior, beta: beta, gamma: gamma, priorDraw: priorDraw)

proc performance*(p: Player): Gaussian =
  newGaussian(p.prior.mu, sqrt(p.prior.sigma * p.prior.sigma + p.beta * p.beta))

proc `$`*(p: Player): string =
  fmt"Player(Gaussian(mu={p.prior.mu:.3f}, sigma={p.prior.sigma:.3f}), beta={p.beta:.3f}, gamma={p.gamma:.3f})"

# Team variable
type
  TeamVariable* = ref object
    prior*: Gaussian
    likelihoodLose*: Gaussian
    likelihoodWin*: Gaussian
    likelihoodDraw*: Gaussian

proc newTeamVariable*(prior = Ninf, likelihoodLose = Ninf, likelihoodWin = Ninf, likelihoodDraw = Ninf): TeamVariable =
  TeamVariable(prior: prior, likelihoodLose: likelihoodLose, likelihoodWin: likelihoodWin, likelihoodDraw: likelihoodDraw)

proc p*(tv: TeamVariable): Gaussian =
  tv.prior * tv.likelihoodLose * tv.likelihoodWin * tv.likelihoodDraw

proc posteriorWin*(tv: TeamVariable): Gaussian =
  tv.prior * tv.likelihoodLose * tv.likelihoodDraw

proc posteriorLose*(tv: TeamVariable): Gaussian =
  tv.prior * tv.likelihoodWin * tv.likelihoodDraw

proc likelihood*(tv: TeamVariable): Gaussian =
  tv.likelihoodWin * tv.likelihoodLose * tv.likelihoodDraw

proc `$`*(tv: TeamVariable): string =
  fmt"TeamVariable({tv.prior}, {tv.likelihoodLose}, {tv.likelihoodWin}, {tv.likelihoodDraw})"

# Draw messages
type
  DrawMessages* = ref object
    prior*: Gaussian
    priorTeam*: Gaussian
    likelihoodLose*: Gaussian
    likelihoodWin*: Gaussian

proc newDrawMessages*(prior = Ninf, priorTeam = Ninf, likelihoodLose = Ninf, likelihoodWin = Ninf): DrawMessages =
  DrawMessages(prior: prior, priorTeam: priorTeam, likelihoodLose: likelihoodLose, likelihoodWin: likelihoodWin)

proc p*(dm: DrawMessages): Gaussian =
  dm.priorTeam * dm.likelihoodLose * dm.likelihoodWin

proc posteriorWin*(dm: DrawMessages): Gaussian =
  dm.priorTeam * dm.likelihoodLose

proc posteriorLose*(dm: DrawMessages): Gaussian =
  dm.priorTeam * dm.likelihoodWin

proc likelihood*(dm: DrawMessages): Gaussian =
  dm.likelihoodWin * dm.likelihoodLose

proc `$`*(dm: DrawMessages): string =
  fmt"DrawMessages({dm.prior}, {dm.priorTeam}, {dm.likelihoodLose}, {dm.likelihoodWin})"

# Diff messages
type
  DiffMessages* = ref object
    prior*: Gaussian
    likelihood*: Gaussian

proc newDiffMessages*(prior = Ninf, likelihood = Ninf): DiffMessages =
  DiffMessages(prior: prior, likelihood: likelihood)

proc p*(dm: DiffMessages): Gaussian =
  dm.prior * dm.likelihood

proc `$`*(dm: DiffMessages): string =
  fmt"DiffMessages({dm.prior}, {dm.likelihood})"

# Approximation function
proc approx*(N: Gaussian, margin: float, tie: bool): Gaussian =
  let (mu, sigma) = trunc(N.mu, N.sigma, margin, tie)
  newGaussian(mu, sigma)

# Game class
type
  Game* = ref object
    teams*: seq[seq[Player]]
    outcome*: seq[int]  # Changed from result to outcome
    pDraw*: float
    weights*: seq[seq[float]]
    likelihoods*: seq[seq[Gaussian]]
    evidence*: float

# Performance function
proc performanceTeam*(team: seq[Player], weights: seq[float]): Gaussian =
  var res = N00
  for i in 0..<team.len:
    res = res + team[i].performance() * weights[i]
  res

proc performance*(g: Game, i: int): Gaussian =
  performanceTeam(g.teams[i], g.weights[i])

proc graphicalModel*(g: Game): (seq[int], seq[TeamVariable], seq[DiffMessages], seq[bool], seq[float]) =
  let r = if g.outcome.len > 0: g.outcome else: toSeq(countdown(g.teams.len - 1, 0))
  let o = sortPerm(r, reverse = true)
  
  var t = newSeq[TeamVariable]()
  for e in 0..<g.teams.len:
    t.add(newTeamVariable(g.performance(o[e]), Ninf, Ninf, Ninf))
  
  var d = newSeq[DiffMessages]()
  for e in 0..<(g.teams.len - 1):
    d.add(newDiffMessages(t[e].prior - t[e+1].prior, Ninf))
  
  var tie = newSeq[bool]()
  for e in 0..<d.len:
    tie.add(r[o[e]] == r[o[e + 1]])
  
  var margin = newSeq[float]()
  for e in 0..<d.len:
    if g.pDraw == 0.0:
      margin.add(0.0)
    else:
      let sumBeta = sum(g.teams[o[e]].mapIt(it.beta * it.beta)) + sum(g.teams[o[e + 1]].mapIt(it.beta * it.beta))
      margin.add(computeMargin(g.pDraw, sqrt(sumBeta)))
  
  g.evidence = 1.0
  (o, t, d, tie, margin)

proc partialEvidence*(g: Game, d: seq[DiffMessages], margin: seq[float], tie: seq[bool], e: int) =
  let mu = d[e].prior.mu
  let sigma = d[e].prior.sigma
  if tie[e]:
    g.evidence *= cdf(margin[e], mu, sigma) - cdf(-margin[e], mu, sigma)
  else:
    g.evidence *= 1 - cdf(margin[e], mu, sigma)

proc likelihoodAnalytico*(g: Game): (seq[Gaussian], seq[Gaussian]) =
  let (o, t, d, tie, margin) = g.graphicalModel()
  g.partialEvidence(d, margin, tie, 0)

  let dPrior = d[0].prior
  let (muTrunc, sigmaTrunc) = trunc(dPrior.mu, dPrior.sigma, margin[0], tie[0])

  let (deltaDiv, thetaDivPow2) = 
    if dPrior.sigma == sigmaTrunc:
      (dPrior.sigma * dPrior.sigma * muTrunc - sigmaTrunc * sigmaTrunc * dPrior.mu, INF)
    else:
      let delta = (dPrior.sigma * dPrior.sigma * muTrunc - sigmaTrunc * sigmaTrunc * dPrior.mu) / 
                  (dPrior.sigma * dPrior.sigma - sigmaTrunc * sigmaTrunc)
      let theta = (sigmaTrunc * sigmaTrunc * dPrior.sigma * dPrior.sigma) / 
                  (dPrior.sigma * dPrior.sigma - sigmaTrunc * sigmaTrunc)
      (delta, theta)

  var res = newSeq[seq[Gaussian]]()
  for i in 0..<t.len:
    var team = newSeq[Gaussian]()
    for j in 0..<g.teams[o[i]].len:
      let mu = if dPrior.sigma == sigmaTrunc: 0.0 
               else: g.teams[o[i]][j].prior.mu + (deltaDiv - dPrior.mu) * (if i == 1: -1.0 else: 1.0)
      let sigmaAnalytico = sqrt(thetaDivPow2 + dPrior.sigma * dPrior.sigma - g.teams[o[i]][j].prior.sigma * g.teams[o[i]][j].prior.sigma)
      team.add(newGaussian(mu, sigmaAnalytico))
    res.add(team)

  if o[0] < o[1]: (res[0], res[1]) else: (res[1], res[0])

proc likelihoodTeams*(g: Game): seq[Gaussian] =
  let (o, t, d, tie, margin) = g.graphicalModel()
  var step = (INF, INF)
  var i = 0
  
  while grTuple(step, 1e-6) and i < 10:
    step = (0.0, 0.0)
    
    # Forward pass
    for e in 0..<(d.len - 1):
      d[e].prior = t[e].posteriorWin - t[e + 1].posteriorLose
      if i == 0:
        g.partialEvidence(d, margin, tie, e)
      d[e].likelihood = approx(d[e].prior, margin[e], tie[e]) / d[e].prior
      let likelihoodLose = t[e].posteriorWin - d[e].likelihood
      step = maxTuple(step, t[e + 1].likelihoodLose.delta(likelihoodLose))
      t[e + 1].likelihoodLose = likelihoodLose
    
    # Backward pass
    for e in countdown(d.len - 1, 1):
      d[e].prior = t[e].posteriorWin - t[e + 1].posteriorLose
      if i == 0 and e == d.len - 1:
        g.partialEvidence(d, margin, tie, e)
      d[e].likelihood = approx(d[e].prior, margin[e], tie[e]) / d[e].prior
      let likelihoodWin = t[e + 1].posteriorLose + d[e].likelihood
      step = maxTuple(step, t[e].likelihoodWin.delta(likelihoodWin))
      t[e].likelihoodWin = likelihoodWin
    
    i += 1
  
  if d.len == 1:
    g.partialEvidence(d, margin, tie, 0)
    d[0].prior = t[0].posteriorWin - t[1].posteriorLose
    d[0].likelihood = approx(d[0].prior, margin[0], tie[0]) / d[0].prior
  
  t[0].likelihoodWin = t[1].posteriorLose + d[0].likelihood
  t[^1].likelihoodLose = t[^2].posteriorWin - d[^1].likelihood
  
  var final_result = newSeq[Gaussian]()
  for e in 0..<t.len:
    final_result.add(t[o[e]].likelihood)
  final_result

proc computeLikelihoods*(g: Game) =
  # Check if we need complex computation
  let needsComplex = g.teams.len > 2 or 
                     (block:
                       var hasNonUnityWeight = false
                       for team in g.weights:
                         for w in team:
                           if w != 1.0:
                             hasNonUnityWeight = true
                             break
                         if hasNonUnityWeight: break
                       hasNonUnityWeight)
  
  if needsComplex:
    let mTFt = g.likelihoodTeams()
    g.likelihoods = newSeq[seq[Gaussian]]()
    for e in 0..<g.teams.len:
      var teamLikelihoods = newSeq[Gaussian]()
      for i in 0..<g.teams[e].len:
        let weight = if g.weights[e][i] != 0.0: 1.0 / g.weights[e][i] else: INF
        let excluded = g.performance(e).exclude(g.teams[e][i].prior * g.weights[e][i])
        teamLikelihoods.add(weight * (mTFt[e] - excluded))
      g.likelihoods.add(teamLikelihoods)
  else:
    let (team1, team2) = g.likelihoodAnalytico()
    g.likelihoods = @[team1, team2]

proc newGame*(teams: seq[seq[Player]], outcome: seq[int] = @[], pDraw = 0.0, weights: seq[seq[float]] = @[]): Game =
  if outcome.len > 0 and teams.len != outcome.len:
    raise newException(ValueError, "len(outcome) and len(teams) must be equal")
  if pDraw < 0.0 or pDraw >= 1.0:
    raise newException(ValueError, "0.0 <= pDraw < 1.0")
  if pDraw == 0.0 and outcome.len > 0:
    let uniqueOutcomes = outcome.deduplicate()
    if uniqueOutcomes.len != outcome.len:
      raise newException(ValueError, "No draws allowed when pDraw == 0.0")
  
  var finalWeights = weights
  if finalWeights.len == 0:
    finalWeights = newSeq[seq[float]]()
    for team in teams:
      finalWeights.add(newSeqWith(team.len, 1.0))
  
  var game = Game(teams: teams, outcome: outcome, pDraw: pDraw, weights: finalWeights, evidence: 0.0)
  game.computeLikelihoods()
  game

proc len*(g: Game): int = g.teams.len

proc size*(g: Game): seq[int] =
  result = newSeq[int]()
  for team in g.teams:
    result.add(team.len)

proc dictDiff*(old, new: TableRef[string, Gaussian]): (float, float) =
  var step = (0.0, 0.0)
  for a in old.keys:
    step = maxTuple(step, old[a].delta(new[a]))
  step

proc posteriors*(g: Game): seq[seq[Gaussian]] =
  result = newSeq[seq[Gaussian]]()
  for e in 0..<g.teams.len:
    var teamPosteriors = newSeq[Gaussian]()
    for i in 0..<g.teams[e].len:
      teamPosteriors.add(g.likelihoods[e][i] * g.teams[e][i].prior)
    result.add(teamPosteriors)

# Skill class
type
  Skill* = ref object
    forward*: Gaussian
    backward*: Gaussian
    likelihood*: Gaussian
    elapsed*: float

proc newSkill*(forward = Ninf, backward = Ninf, likelihood = Ninf, elapsed = 0.0): Skill =
  Skill(forward: forward, backward: backward, likelihood: likelihood, elapsed: elapsed)

# Agent class
type
  Agent* = ref object
    player*: Player
    message*: Gaussian
    lastTime*: float

proc newAgent*(player: Player, message: Gaussian, lastTime: float): Agent =
  Agent(player: player, message: message, lastTime: lastTime)

proc receive*(a: Agent, elapsed: float): Gaussian =
  if a.message != Ninf:
    a.message.forget(a.player.gamma, elapsed)
  else:
    a.player.prior

proc `$`*(a: Agent): string =
  fmt"Agent(player={a.player}, message={a.message}, last_time={a.last_time})"

proc clean*(agents: TableRef[string, Agent], lastTime = false) =
  for a in agents.values:
    a.message = Ninf
    if lastTime:
      a.lastTime = -INF

# Item class
type
  Item* = ref object
    name*: string
    likelihood*: Gaussian

proc newItem*(name: string, likelihood: Gaussian): Item =
  Item(name: name, likelihood: likelihood)

# Team class
type
  Team* = ref object
    items*: seq[Item]
    output*: int

proc newTeam*(items: seq[Item], output: int): Team =
  Team(items: items, output: output)

# Event class
type
  Event* = ref object
    teams*: seq[Team]
    evidence*: float
    weights*: seq[seq[float]]

proc newEvent*(teams: seq[Team], evidence: float, weights: seq[seq[float]]): Event =
  Event(teams: teams, evidence: evidence, weights: weights)

proc names*(e: Event): seq[seq[string]] =
  result = newSeq[seq[string]]()
  for team in e.teams:
    var teamNames = newSeq[string]()
    for item in team.items:
      teamNames.add(item.name)
    result.add(teamNames)

proc result*(e: Event): seq[int] =
  result = newSeq[int]()
  for team in e.teams:
    result.add(team.output)

proc `$`*(e: Event): string =
  fmt"Event({e.names}, {e.result})"

proc computeElapsed*(lastTime, actualTime: float): float =
  if lastTime == -INF: 0.0
  elif lastTime == INF: 1.0
  else: actualTime - lastTime

# Batch class
type
  Batch* = ref object
    skills*: TableRef[string, Skill]
    events*: seq[Event]
    time*: float
    agents*: TableRef[string, Agent]
    pDraw*: float

proc len*(b: Batch): int = b.events.len

proc `$`*(b: Batch): string =
  fmt"Batch(time={b.time}, events={b.events})"

proc posterior*(b: Batch, agent: string): Gaussian =
  b.skills[agent].likelihood * b.skills[agent].backward * b.skills[agent].forward

proc posteriors*(b: Batch): TableRef[string, Gaussian] =
  var res = newTable[string, Gaussian]()
  for a in b.skills.keys:
    res[a] = b.posterior(a)
  res

# Complete the withinPrior proc
proc withinPrior*(b: Batch, item: Item): Player =
  let agent = b.agents[item.name]
  let posterior = b.posterior(item.name) / item.likelihood
  newPlayer(newGaussian(posterior.mu, posterior.sigma), agent.player.beta, agent.player.gamma)

# Add withinPriors proc
proc withinPriors*(b: Batch, eventIndex: int): seq[seq[Player]] =
  result = newSeq[seq[Player]]()
  for team in b.events[eventIndex].teams:
    var teamPlayers = newSeq[Player]()
    for item in team.items:
      teamPlayers.add(b.withinPrior(item))
    result.add(teamPlayers)

# Add iteration proc
proc iteration*(b: Batch, startIndex: int = 0) =
  for e in startIndex ..< b.events.len:
    let teams = b.withinPriors(e)
    let result = b.events[e].result
    let weights = b.events[e].weights
    let g = newGame(teams, result, b.pDraw, weights)
    for t, team in b.events[e].teams.pairs:
      for i, item in team.items.pairs:
        let skill = b.skills[item.name]
        let newLikelihood = (skill.likelihood / item.likelihood) * g.likelihoods[t][i]
        skill.likelihood = newLikelihood
        item.likelihood = g.likelihoods[t][i]
    b.events[e].evidence = g.evidence

# Add convergence proc
proc convergence*(b: Batch, epsilon: float = EPSILON, iterations: int = ITERATIONS): int =
  var step: (float, float) = (Inf, Inf)
  var i = 0
  while step.grTuple(epsilon) and i < iterations:
    let old = b.posteriors()
    b.iteration()
    step = dictDiff(old, b.posteriors())
    inc i
  result = i

# Add forwardPriorOut proc
proc forwardPriorOut*(b: Batch, agent: string): Gaussian =
  b.skills[agent].forward * b.skills[agent].likelihood

# Add backwardPriorOut proc
proc backwardPriorOut*(b: Batch, agent: string): Gaussian =
  let N = b.skills[agent].likelihood * b.skills[agent].backward
  N.forget(b.agents[agent].player.gamma, b.skills[agent].elapsed)

# Add newBackwardInfo proc
proc newBackwardInfo*(b: Batch) =
  for agent, skill in b.skills:
    skill.backward = b.agents[agent].message
  b.iteration()

# Add newForwardInfo proc
proc newForwardInfo*(b: Batch) =
  for agent, skill in b.skills:
    skill.forward = b.agents[agent].receive(skill.elapsed)
  b.iteration()

proc newBatch*(
  composition: seq[seq[seq[string]]],
  outcomes: seq[seq[int]] = @[],
  time = 0.0,
  agents: TableRef[string, Agent],
  pDraw = 0.0,
  weights: seq[seq[seq[float]]] = @[]
): Batch =
  # Collect all agents
  var thisAgents = initHashSet[string]()
  for teams in composition:
    for team in teams:
      for a in team:
        thisAgents.incl(a)
  
  # Calculate elapsed time
  var elapsed = initTable[string, float]()
  for a in thisAgents:
    elapsed[a] = computeElapsed(agents[a].lastTime, time)
  
  # Initialize skills with proper likelihoods
  var skills = newTable[string, Skill]()
  for a in thisAgents:
    let prior = agents[a].receive(elapsed[a])
    skills[a] = newSkill(
      forward = prior,
      backward = Ninf,
      likelihood = Ninf, 
      elapsed = elapsed[a]
    )
  
  # Create events
  var events = newSeq[Event]()
  for e in 0..<composition.len:
    var teams = newSeq[Team]()
    for t in 0..<composition[e].len:
      var items = newSeq[Item]()
      for a in 0..<composition[e][t].len:
        items.add(newItem(composition[e][t][a], Ninf))  # Proper init
      let output = if outcomes.len > 0: outcomes[e][t] else: composition[e].len - t - 1
      teams.add(newTeam(items, output))
    
    let eventWeights = if weights.len > 0: weights[e] else: @[]
    events.add(newEvent(teams, 1.0, eventWeights))  # Start evidence at 1.0
  
  # Create batch
  var batch = Batch(
    skills: skills,
    events: events,
    time: time,
    agents: agents,
    pDraw: pDraw
  )
  
  # Perform initial iteration (MISSING STEP)
  batch.iteration()
  
  batch


type
  History* = ref object
    size*: int
    batches*: seq[Batch]
    agents*: TableRef[string, Agent]
    mu*: float
    sigma*: float
    gamma*: float
    p_draw*: float
    time*: bool

proc iteration*(h: History) =
  var step: (float, float) = (0.0, 0.0)
  
  # Clean agents for backward pass
  for agent in h.agents.values:
    agent.message = Ninf
  
  # Backward pass
  for j in countdown(h.batches.high - 1, 0):
    for agent in h.batches[j+1].skills.keys:
      h.agents[agent].message = h.batches[j+1].backwardPriorOut(agent)
    
    let old = h.batches[j].posteriors()
    h.batches[j].newBackwardInfo()
    step = maxTuple(step, dictDiff(old, h.batches[j].posteriors()))
  
  # Clean agents for forward pass
  for agent in h.agents.values:
    agent.message = Ninf
  
  # Forward pass
  for j in 1..h.batches.high:
    for agent in h.batches[j-1].skills.keys:
      h.agents[agent].message = h.batches[j-1].forwardPriorOut(agent)
    
    let old = h.batches[j].posteriors()
    h.batches[j].newForwardInfo()
    step = maxTuple(step, dictDiff(old, h.batches[j].posteriors()))
  
  # Handle single batch case
  if h.batches.len == 1:
    let old = h.batches[0].posteriors()
    discard h.batches[0].convergence()
    step = maxTuple(step, dictDiff(old, h.batches[0].posteriors()))

proc posteriors*(h: History): TableRef[string, Gaussian] =
  result = newTable[string, Gaussian]()
  for batch in h.batches:
    let batchPost = batch.posteriors()
    for agent, gaussian in batchPost.pairs:
      result[agent] = gaussian

proc convergence*(h: History, epsilon: float = EPSILON, iterations: int = ITERATIONS, verbose: bool = true): (float, float, int) =
  var step: (float, float) = (Inf, Inf)
  var i = 0
  while i < iterations:
    if verbose:
      echo "Iteration ", i
    
    # Store current state for comparison
    var oldPosteriors = newTable[string, Gaussian]()
    for batch in h.batches:
      let batchPost = batch.posteriors()
      for agent, gaussian in batchPost:
        oldPosteriors[agent] = gaussian
    
    # Run iteration
    h.iteration()
    
    # Calculate max change
    step = (0.0, 0.0)
    for batch in h.batches:
      let newPost = batch.posteriors()
      for agent in newPost.keys:
        let oldG = oldPosteriors.getOrDefault(agent, Ninf)
        let newG = newPost[agent]
        let delta = oldG.delta(newG)
        step = maxTuple(step, delta)
    
    if verbose:
      echo "  Step: mu=", step[0], " sigma=", step[1]
    
    # Check convergence
    if not step.grTuple(epsilon):
      break
      
    i.inc
  
  if verbose:
    echo "Converged in ", i, " iterations"
  (step[0], step[1], i)

proc learningCurves*(h: History): TableRef[string, seq[(float, Gaussian)]] =
  result = newTable[string, seq[(float, Gaussian)]]()
  for batch in h.batches:
    for agent, posterior in batch.posteriors():
      if not result.hasKey(agent):
        result[agent] = @[]
      result[agent].add((batch.time, posterior))

proc logEvidence*(h: History): float =
  result = 0.0
  for batch in h.batches:
    for event in batch.events:
      result += ln(event.evidence)

proc `$`*(h: History): string =
  fmt"History(Events={h.size}, Batches={h.batches}, Agents={h.agents})"

proc newHistory*(
  composition: seq[seq[seq[string]]],
  results: seq[seq[int]] = @[],
  times: seq[float] = @[],
  priors: TableRef[string, Player] = nil,
  mu: float = MU,
  sigma: float = SIGMA,
  beta: float = BETA,
  gamma: float = GAMMA,
  p_draw: float = P_DRAW,
  weights: seq[seq[seq[float]]] = @[]
): History =
  # Validate inputs
  if results.len > 0 and composition.len != results.len:
    raise newException(ValueError, "len(composition) != len(results)")
  if times.len > 0 and composition.len != times.len:
    raise newException(ValueError, "len(times) error")
  if weights.len > 0 and composition.len != weights.len:
    raise newException(ValueError, "len(weights) != len(composition)")

  result = History(
    size: composition.len,
    batches: @[],
    agents: newTable[string, Agent](),
    mu: mu,
    sigma: sigma,
    gamma: gamma,
    p_draw: p_draw,
    time: times.len > 0
  )

  # Create agents table
  var allAgents = initHashSet[string]()
  for event in composition:
    for team in event:
      for agent in team:
        allAgents.incl(agent)

  for agent in allAgents:
    let player = 
      if priors != nil and priors.hasKey(agent):
        priors[agent]
      else:
        newPlayer(newGaussian(mu, sigma), beta, gamma)
    result.agents[agent] = newAgent(player, Ninf, -Inf)

  # Create batches grouped by time
  var indices = toSeq(0..<composition.len)
  if times.len > 0:
    indices = indices.sortedByIt(times[it])

  var i = 0
  while i < indices.len:
    var j = i + 1
    let t = 
      if times.len > 0: times[indices[i]]
      else: float(i + 1)
    
    # Find events at same time
    if times.len > 0:
      while j < indices.len and times[indices[j]] == t:
        j.inc
    
    # Create batch for events at time t
    let batchIndices = indices[i..<j]
    var batchComp = newSeq[seq[seq[string]]]()
    var batchResults = newSeq[seq[int]]()
    var batchWeights = newSeq[seq[seq[float]]]()
    
    for idx in batchIndices:
      batchComp.add(composition[idx])
      if results.len > 0:
        batchResults.add(results[idx])
      if weights.len > 0:
        batchWeights.add(weights[idx])
    
    let batch = newBatch(
      batchComp,
      if results.len > 0: batchResults else: @[],
      t,
      result.agents,
      p_draw,
      if weights.len > 0: batchWeights else: @[]
    )
    
    result.batches.add(batch)
    
    # Update agent states
    for agent in batch.skills.keys:
      result.agents[agent].lastTime = 
        if result.time: t 
        else: Inf
      result.agents[agent].message = batch.forwardPriorOut(agent)
    
    i = j

when isMainModule:
  # let a1 = newPlayer(newGaussian(MU, SIGMA), BETA, GAMMA)
  # let a2 = newPlayer()
  # let a3 = newPlayer()
  # let a4 = newPlayer()

  # let team_a = @[a1, a2]
  # let team_b = @[a3, a4]
  # let teams = @[team_a, team_b]

  # let g = newGame(teams)

  # # echo team_a
  # # echo team_b
  # # echo "Result:", g.outcome
  # # echo "Draw:",g.pDraw 
  # # echo "Weights:", g.weights
  # echo()
  
  # let lhs = g.likelihoods[0][0]
  # let ev = g.evidence
  # echo ev

  # echo lhs

  # let pos = g.posteriors()
  
  # echo pos[0][0]
  # echo lhs * a1.prior

  let
    composition = @[
      @[@["a"], @["b"]],
      @[@["b"], @["c"]],
      @[@["c"], @["a"]]
    ]

  # disable time dynamics:
  var history = newHistory(
    composition,
    gamma   = 0.0
  )
  echo history, "\n"

  let initialCurves = history.learningCurves()
  echo initialCurves["a"]
  echo initialCurves["b"]

  # Run convergence
  echo "Running convergence..."
  let (stepMu, stepSigma, iterations) = history.convergence(verbose=true)
  echo "Converged in ", iterations, " iterations with max step (mu: ", stepMu, ", sigma: ", stepSigma, ")"
  
  # Get learning curves
  let curves = history.learningCurves()
  
  # Print skill evolution
  echo "\nLearning curves:"
  for player, curve in curves.pairs:
    echo "\nPlayer ", player, ":"
    for i, (time, gaussian) in curve.pairs:
      echo &"  Time {time}: μ = {gaussian.mu:.3f} σ = {gaussian.sigma:.3f}"
  
  # Print evidence
  echo "\nLog evidence: ", history.logEvidence()