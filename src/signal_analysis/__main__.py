"""Run the complete signal-diagnostic stage."""

from .csmom_groups import make_csmom_outputs
from .data_split import save_splits
from .mom_percentiles import make_mom_percentile_outputs
from .momra_groups import make_momra_group_outputs
from .momra_percentiles import make_momra_percentile_outputs
from .scatterplots import make_scatterplots
from .signals import save_signal_data


def main() -> None:
    save_splits()
    save_signal_data()
    make_scatterplots()
    make_momra_group_outputs()
    make_mom_percentile_outputs()
    make_momra_percentile_outputs()
    make_csmom_outputs()


if __name__ == "__main__":
    main()
