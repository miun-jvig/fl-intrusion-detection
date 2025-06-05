import dp_accounting
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_noise_target():
    df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'DNN-EdgeIIoT-dataset.csv', low_memory=False)
    # privacy target
    target_eps = 10
    target_delta = 1e-7

    # fl parameters
    total_clients = df['ip.src_host'].nunique()
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


if __name__ == '__main__':
    get_noise_target()
