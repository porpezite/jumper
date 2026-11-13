"""The cross-fade is `ref_free_jump`'s, imported. This task schedules two more
terms through the same window (the spin pair, beside the height pair); the
annealer itself is unchanged."""

from ...ref_free_jump.mdp.curriculum import anneal_reward_weight, report_feedforward

__all__ = ["anneal_reward_weight", "report_feedforward"]
