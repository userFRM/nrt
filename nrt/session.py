import numpy as np
import random
from nrt import NRTConfig, SessionResult
from nrt.belief import BeliefState
from nrt.sender import NRTSender
from nrt.receiver import NRTReceiver
from nrt.models import zero_prior, spectral_prior

def _packet_bytes(packet) -> int:
    total = 0
    for f in packet.frames:
        total += f.u.nbytes + f.v.nbytes + 4 + 4 + 4  # u, v, c (float), lambda (float), round (int)
    return total

class NRTSession:
    def run(self, data: np.ndarray, prior: BeliefState = None,
            loss_rate: float = 0.0, config: NRTConfig = None) -> SessionResult:
        if config is None:
            config = NRTConfig()
        if prior is None:
            prior = zero_prior(data.shape)

        # Sender and receiver share the same initial belief structure
        # but maintain independent copies (simulate network separation)
        import copy
        sender_belief = copy.deepcopy(prior)   # sender tracks its model of receiver
        receiver_belief = copy.deepcopy(prior)  # receiver's actual state

        sender = NRTSender(data, config)
        receiver = NRTReceiver(config, receiver_belief)

        errors, bytes_per_round, retransmit_norms = [], [], []
        total_bytes = 0

        for round_idx in range(config.max_rounds):
            if receiver.belief.residual_norm(data) < config.epsilon:
                break

            packet = sender._make_packet(sender_belief, round_idx)
            nb = _packet_bytes(packet)

            if random.random() < loss_rate:
                # Packet lost — retransmit against current receiver belief
                retransmit = sender.retransmit(copy.deepcopy(receiver.belief), round_idx)
                orig_norm = sum(abs(f.c) for f in packet.frames)
                retx_norm = sum(abs(f.c) for f in retransmit.frames)
                retransmit_norms.append((orig_norm, retx_norm))
                # Deliver retransmit
                packet = retransmit
                nb = _packet_bytes(packet)

            # Apply to receiver
            err = receiver.apply(packet, data)
            errors.append(err)
            bytes_per_round.append(nb)
            total_bytes += nb

            # Sender updates its model of receiver state
            sender_belief.apply_correction(packet.frames)
            sender_belief.update_covariance(packet.frames)

        return SessionResult(
            error_per_round=errors,
            bytes_per_round=bytes_per_round,
            total_bytes=total_bytes,
            rounds=len(errors),
            converged=receiver.belief.residual_norm(data) < config.epsilon,
            retransmit_norms=retransmit_norms,
            data_size_bytes=data.nbytes,
        )
