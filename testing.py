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