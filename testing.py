import trueskillthroughtime as t
import ttt
import time

def compare_functions(func_a, func_b, *args, iterations=1_000_000, **kwargs):
    # Warm-up runs
    out_a = func_a(*args, **kwargs)
    out_b = func_b(*args, **kwargs)
    
    # Time func_a
    start_a = time.perf_counter()
    for _ in range(iterations):
        output_a = func_a(*args, **kwargs)
    end_a = time.perf_counter()
    time_a = (end_a - start_a) / iterations
    
    # Time func_b
    start_b = time.perf_counter()
    for _ in range(iterations):
        output_b = func_b(*args, **kwargs)
    end_b = time.perf_counter()
    time_b = (end_b - start_b) / iterations
    
    # Compare outputs
    # Compare outputs
    if isinstance(output_a, tuple) and len(output_a) == 2:
        # Handle Gaussian tuple comparison
        mu_a, sigma_a = output_a
        mu_b, sigma_b = output_b
        
        outputs_match = (abs(mu_a - mu_b) < 1e-9 and 
                         abs(sigma_a - sigma_b) < 1e-9)
        output_a_str = f"Gaussian(mu={mu_a:.16f}, sigma={sigma_a:.16f})"
        output_b_str = f"Gaussian(mu={mu_b:.16f}, sigma={sigma_b:.16f})"
    elif hasattr(output_a, 'mu') and hasattr(output_b, 'mu'):
        # Handle Python Gaussian objects
        mu_a = output_a.mu
        sigma_a = output_a.sigma
        mu_b = output_b.mu
        sigma_b = output_b.sigma
        
        outputs_match = (abs(mu_a - mu_b) < 1e-9 and 
                         abs(sigma_a - sigma_b) < 1e-9)
        output_a_str = f"Gaussian(mu={mu_a:.16f}, sigma={sigma_a:.16f})"
        output_b_str = f"Gaussian(mu={mu_b:.16f}, sigma={sigma_b:.16f})"
    elif isinstance(output_b, tuple):
        # General tuple comparison
        outputs_match = all(abs(a - b) < 1e-9 for a, b in zip(output_a, output_b))
        output_a_str = str(tuple(f"{x:.16f}" for x in output_a))
        output_b_str = str(tuple(f"{x:.16f}" for x in output_b))
    else:
        # Single value comparison
        outputs_match = abs(output_a - output_b) < 1e-9
        output_a_str = f"{output_a:.16f}"
        output_b_str = f"{output_b:.16f}"
    
    # Print results
    print(f"Original output: {output_a_str} | Time/call: {time_a:.9f} sec")
    print(f"Nim output:      {output_b_str} | Time/call: {time_b:.9f} sec")
    print(f"Outputs match:   {outputs_match}")
    print(f"Speedup:         {time_a/time_b:.2f}x")
    print(f"Total calls:     {iterations:,}\n")

def compare_gaussians(g1, g2, tol=1e-9):
    """Compare two Gaussian tuples (mu1, sigma1) and (mu2, sigma2) within tolerance."""
    return (abs(g1[0] - g2[0]) < tol) and (abs(g1[1] - g2[1]) < tol)

# print("\nerfc")
# compare_functions(t.erfc, ttt.erfc, 1)
# compare_functions(t.erfc, ttt.erfc, -1)
# print("\nerfcinv")
# compare_functions(t.erfcinv, ttt.erfcinv, 1)
# print("\ntau_pi")
# compare_functions(t.tau_pi, ttt.tau_pi, 0, 1)
# print("\nmu_sigma")
# compare_functions(t.mu_sigma, ttt.mu_sigma, 0, 1)
# print("\ncdf")
# compare_functions(t.cdf, ttt.cdf, 0.5, 0, 1)
# print("\npdf")
# compare_functions(t.pdf, ttt.pdf, 0.5, 0, 1)
# print("\nppf")
# compare_functions(t.ppf, ttt.ppf, 0.5, 0, 1)
# print("\nv_w")
# compare_functions(t.v_w, ttt.v_w, 0, 1, 0.5, False)
# compare_functions(t.v_w, ttt.v_w, 0, 1, 0.5, True)
# print("\ntrunc")
# compare_functions(t.trunc, ttt.trunc, 0, 1, 0.5, False)
# compare_functions(t.trunc, ttt.trunc, 0, 1, 0.5, True)

print("\nGaussian operations")
# Test addition
g1_py = t.Gaussian(1.0, 0.5)
g2_py = t.Gaussian(0.5, 0.5)  # Larger sigma makes division valid
g1_nim = ttt.gaussian(1.0, 0.5)
g2_nim = ttt.gaussian(0.5, 0.5)
g3_py = t.Gaussian(1.01, 0.51)
g3_nim = ttt.gaussian(1.01, 0.51)

print("\nAdd")
compare_functions(lambda: g1_py + g2_py, 
                  lambda: ttt.gaussian_add(g1_nim, g2_nim), 
                  iterations=100000)
print("\nMult")
compare_functions(lambda: g1_py * g2_py, 
                  lambda: ttt.gaussian_mul(g1_nim, g2_nim), 
                  iterations=100000)
print("\nDiv")
compare_functions(lambda: g1_py / g2_py, 
                  lambda: ttt.gaussian_div(g1_nim, g2_nim), 
                  iterations=100000)
print("\nk *")
compare_functions(lambda: 2.0 * g1_py, 
                  lambda: ttt.gaussian_scale(g1_nim, 2.0), 
                  iterations=100000)
print("\nforget")
compare_functions(lambda: g1_py.forget(0.1, 2), 
                  lambda: ttt.gaussian_forget(g1_nim, 0.1, 2), 
                  iterations=100000)
print("\napprox")
compare_functions(lambda: t.approx(g1_py, 0.5, False), 
                  lambda: ttt.approx(g1_nim, 0.5, False), 
                  iterations=100000)
print("\ndelta")
compare_functions(lambda: g1_py.delta(g3_py), 
                  lambda: ttt.gaussian_delta(g1_nim, g3_nim), 
                  iterations=100000)
print("\nexclude")
compare_functions(lambda: g1_py.exclude(g2_py), 
                  lambda: ttt.gaussian_exclude(g1_nim, g2_nim), 
                  iterations=100000)

# Test Player
player_nim = ttt.newPlayer(ttt.gaussian(25.0, 8.0), beta=1.0, gamma=0.1)
print("Player:", ttt.reprPlayer(player_nim))
print("Performance:", ttt.player_performance(player_nim))

# Test Team Performance with multiple players
players = [
    ttt.newPlayer(ttt.gaussian(25.0, 8.0)),
    ttt.newPlayer(ttt.gaussian(30.0, 4.0))
]
weights = [1.0, 0.5]
print("Team Performance:", ttt.team_performance(players, weights))

# Test Team Variable
tv_nim = ttt.newTeamVariable(
    prior=ttt.gaussian(25.0, 8.0),
    likelihood_lose=ttt.gaussian(0.0, 1.0),
    likelihood_win=ttt.gaussian(1.0, 1.0),
    likelihood_draw=ttt.gaussian(0.5, 0.5)
)

print("\nTeamVariable operations:")
print("p:", ttt.team_variable_p(tv_nim))
print("posterior_win:", ttt.team_variable_posterior_win(tv_nim))
print("posterior_lose:", ttt.team_variable_posterior_lose(tv_nim))
print("likelihood:", ttt.team_variable_likelihood(tv_nim))


# Recreate the same inputs in Python
prior_py = t.Gaussian(25.0, 8.0)
likelihood_lose_py = t.Gaussian(0.0, 1.0)
likelihood_win_py = t.Gaussian(1.0, 1.0)
likelihood_draw_py = t.Gaussian(0.5, 0.5)

# Create TeamVariable instance
tv_py = t.team_variable(
    prior=prior_py,
    likelihood_lose=likelihood_lose_py,
    likelihood_win=likelihood_win_py,
    likelihood_draw=likelihood_draw_py
)

# Get results
p_py = tv_py.p
posterior_win_py = tv_py.posterior_win
posterior_lose_py = tv_py.posterior_lose
likelihood_py = tv_py.likelihood

# Print Python results
print("\nPython TeamVariable operations:")
print(f"p: (mu={p_py.mu:.16f}, sigma={p_py.sigma:.16f})")
print(f"posterior_win: (mu={posterior_win_py.mu:.16f}, sigma={posterior_win_py.sigma:.16f})")
print(f"posterior_lose: (mu={posterior_lose_py.mu:.16f}, sigma={posterior_lose_py.sigma:.16f})")
print(f"likelihood: (mu={likelihood_py.mu:.16f}, sigma={likelihood_py.sigma:.16f})")
print()
# Create players with the same parameters in both libraries
mu1, sigma1 = 25.0, 8.0
mu2, sigma2 = 30.0, 4.0
mu3, sigma3 = 20.0, 6.0

# Python players
player1_py = t.Player(t.Gaussian(mu1, sigma1))
player2_py = t.Player(t.Gaussian(mu2, sigma2))
player3_py = t.Player(t.Gaussian(mu3, sigma3))

# Nim players
player1_nim = ttt.newPlayer(ttt.gaussian(mu1, sigma1))
player2_nim = ttt.newPlayer(ttt.gaussian(mu2, sigma2))
player3_nim = ttt.newPlayer(ttt.gaussian(mu3, sigma3))

# Create game: team1 = [player1_py, player2_py] vs team2 = [player3_py]
# team1 wins (result=[1, 0])
teams_py = [[player1_py, player2_py], [player3_py]]
teams_nim = [[player1_nim, player2_nim], [player3_nim]]
result = [1, 0]  # team1 wins
p_draw = 0.1

# Python game
game_py = t.Game(teams_py, result, p_draw)
game_py.compute_likelihoods()
evidence_py = game_py.evidence
likelihoods_py = game_py.likelihoods

# Nim game
game_nim = ttt.newGame(teams_nim, result, p_draw)
(likelihoods_nim, evidence_nim) = ttt.compute_likelihoods(game_nim)

# Compare evidence
print(f"Evidence match: {abs(evidence_py - evidence_nim) < 1e-9} (Python: {evidence_py}, Nim: {evidence_nim})")

# Compare likelihoods
for i in range(len(teams_py)):
    for j in range(len(teams_py[i])):
        l_py = likelihoods_py[i][j]
        l_nim = likelihoods_nim[i][j]
        match = compare_gaussians((l_py.mu, l_py.sigma), l_nim)
        print(f"Team {i}, Player {j} likelihood match: {match}")
        print(f"  Python: mu={l_py.mu}, sigma={l_py.sigma}")
        print(f"  Nim:    mu={l_nim[0]}, sigma={l_nim[1]}")

# Compare posteriors
posteriors_py = game_py.posteriors()
posteriors_nim = ttt.posteriors(game_nim, likelihoods_nim)

for i in range(len(teams_py)):
    for j in range(len(teams_py[i])):
        p_py = posteriors_py[i][j]
        p_nim = posteriors_nim[i][j]
        match = compare_gaussians((p_py.mu, p_py.sigma), p_nim)
        print(f"Team {i}, Player {j} posterior match: {match}")
        print(f"  Python: mu={p_py.mu}, sigma={p_py.sigma}")
        print(f"  Nim:    mu={p_nim[0]}, sigma={p_nim[1]}")