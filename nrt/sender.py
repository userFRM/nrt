import numpy as np
from nrt import NRTConfig, CorrectionFrame, RefinementPacket
from nrt.belief import BeliefState

class NRTSender:
    def __init__(self, data: np.ndarray, config: NRTConfig):
        self.data = data.astype(np.float32)
        self.config = config

    def _make_packet(self, belief: BeliefState, round_idx: int) -> RefinementPacket:
        """
        Compute one round of corrections via weighted-residual SVD.

        Rw = Σ_L^{1/2} R Σ_R^{1/2} — weights residual by receiver uncertainty.
        SVD of Rw gives optimal directions that balance uncertainty and correction magnitude.
        Degenerates to standard residual SVD when Σ = I (zero prior).
        """
        residual = self.data - belief.estimate
        sqrt_L, sqrt_R = belief.cov_sqrt()
        Rw = sqrt_L @ residual @ sqrt_R

        U, s, Vt = np.linalg.svd(Rw, full_matrices=False)
        k = min(self.config.k_per_round, len(s))

        frames = []
        for i in range(k):
            if s[i] < 1e-10:
                break
            # Map SVD directions back to original space via Σ^{1/2}
            u_orig = sqrt_L @ U[:, i]
            v_orig = sqrt_R @ Vt[i, :]
            u_norm = np.linalg.norm(u_orig)
            v_norm = np.linalg.norm(v_orig)
            if u_norm < 1e-10 or v_norm < 1e-10:
                continue
            u_hat = u_orig / u_norm
            v_hat = v_orig / v_norm
            c = float(u_hat @ residual @ v_hat)
            frames.append(CorrectionFrame(
                u=u_hat, v=v_hat, c=c, lambda_ij=float(s[i]),
                round_idx=round_idx))

        # Estimate residual after applying these corrections
        if frames:
            correction = sum(f.c * np.outer(f.u, f.v) for f in frames)
            new_residual_norm = np.linalg.norm(residual - correction) / (np.linalg.norm(self.data) + 1e-8)
        else:
            new_residual_norm = np.linalg.norm(residual) / (np.linalg.norm(self.data) + 1e-8)

        return RefinementPacket(
            round_idx=round_idx,
            frames=frames,
            phi_after=float(new_residual_norm),
            is_final=new_residual_norm < self.config.epsilon
        )

    def generate_refinements(self, initial_belief: BeliefState):
        """
        Yield RefinementPackets one per round.
        Uses the SHARED belief state — sender tracks receiver's covariance.
        """
        belief = initial_belief
        for round_idx in range(self.config.max_rounds):
            err = belief.residual_norm(self.data)
            if err < self.config.epsilon:
                break
            packet = self._make_packet(belief, round_idx)
            yield packet
            # Simulate receiver applying the packet (shared belief update)
            belief.apply_correction(packet.frames)
            belief.update_covariance(packet.frames)

    def retransmit(self, current_belief: BeliefState, round_idx: int) -> RefinementPacket:
        """
        Fresh correction against receiver's CURRENT belief (not a copy).
        Norm guaranteed ≤ original because residual can only shrink.
        """
        return self._make_packet(current_belief, round_idx)
