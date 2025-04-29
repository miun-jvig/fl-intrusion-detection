import dp_accounting


def get_epsilon(num):
    # privacy parameters
    target_delta = 1e-7

    # FL parameters
    total_clients = 137167
    clients_per_round = 5
    server_rounds = 25

    # sampling ratio
    q = clients_per_round / total_clients

    def compute_epsilon(noise_mult: float) -> float:
        # 1) make a fresh accountant
        acct = dp_accounting.rdp.RdpAccountant()

        # 2) build the per-round event: sample then add Gaussian noise
        gauss = dp_accounting.GaussianDpEvent(noise_mult)
        samp = dp_accounting.PoissonSampledDpEvent(q, gauss)
        # 3) compose it over all rounds
        composed = dp_accounting.SelfComposedDpEvent(samp, server_rounds)

        # 4) feed it to the accountant
        acct.compose(composed)

        # 5) ask “what ε for this δ?”
        return acct.get_epsilon(target_delta)

    eps = compute_epsilon(num)
    print(f"noise_mult = {num} ⇒ ε ≈ {eps:.1f} for (δ={target_delta})")


def get_noise_target():
    # privacy target
    target_eps = 150
    target_delta = 1e-7

    # fl parameters
    total_clients = 137167
    clients_per_round = 5
    server_rounds = 25

    def make_fresh_accountant():
        return dp_accounting.rdp.RdpAccountant()

    def make_event_from_noise(noise_mult):
        # each round you subsample Poisson(q) then add Gaussian(noise_mult)
        q = clients_per_round / total_clients
        gauss = dp_accounting.GaussianDpEvent(noise_mult)
        samp = dp_accounting.PoissonSampledDpEvent(q, gauss)
        return dp_accounting.SelfComposedDpEvent(samp, server_rounds)

    # bracket over a reasonable range of noise multipliers
    bracket = dp_accounting.ExplicitBracketInterval(0.01, 10.0)

    best_noise = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant,
        make_event_from_noise,
        target_eps,
        target_delta,
        bracket_interval=bracket,
        discrete=False,      # continuous search
    )

    print(f"Use noise_multiplier ≈ {best_noise:.3f} to get (ε={target_eps}, δ={target_delta})-DP")


# get_noise_target()
get_epsilon(0.293)
