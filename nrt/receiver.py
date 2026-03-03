import numpy as np
from nrt import NRTConfig, RefinementPacket
from nrt.belief import BeliefState

class NRTReceiver:
    def __init__(self, config: NRTConfig, initial_belief: BeliefState):
        self.config = config
        self.belief = initial_belief

    def apply(self, packet: RefinementPacket, truth: np.ndarray) -> float:
        self.belief.apply_correction(packet.frames)
        self.belief.update_covariance(packet.frames)
        return self.belief.residual_norm(truth)

    def is_sufficient(self, truth: np.ndarray) -> bool:
        return self.belief.residual_norm(truth) < self.config.epsilon
