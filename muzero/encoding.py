import pyspiel
import torch
import random

""" 
Gives implementations for encode_board, encode_turn, encode_observations, and
encode_action. This takes open_spiel API and converts to torch tensors in MuZero
format for use in future models.
"""

def encode_board(state):
    """Return float32 (2, 9, 9) stone planes: Black, then White.
    Each channel is 1 where that color has a stone and 0 elsewhere."""

    planes = torch.tensor(state.observation_tensor(0), dtype=torch.float32)
    return planes.reshape(4, 9, 9)[:2]

def encode_turn(state):
    """Return float32 (1, 9, 9) turn plane.
    Plane is all 0s for Black to move or all 1s for White to move."""

    planes = torch.tensor(state.observation_tensor(0), dtype=torch.float32)
    return planes.reshape(4, 9, 9)[3:4]

def encode_observations(history, t, obs_length=8):
    """Returns float32 observation planes of shape (obs_lengths * 2 + 1, 9, 9)
    Ex. for obs_length=8, returns (17, 9, 9)

    history: list of states
    t: index of starting state
    obs_length: how many observations

    Planes are the concatenation of: encoded prior obs_length-1 states,
    encoded starting state, and encoded starting state turn
    """

    if t < 0 or t >= len(history):
        raise IndexError("Index t is out of bounds")

    if obs_length < 1:
        raise ValueError("obs_length must be positive")

    boards = [encode_board(s) for s in history[max(0, t - obs_length + 1):t + 1]]
    padding = torch.zeros((obs_length - len(boards)) * 2, 9, 9)
    return torch.cat([padding, *boards, encode_turn(history[t])], dim=0)

def encode_action(action):
    """Return a float32 (1, 9, 9) move plane.
    Singular 1 at move location, all 0s for pass"""

    plane = torch.zeros(1, 9, 9, dtype=torch.float32)
    if action != 81:
        plane[0, action // 9, action % 9] = 1
    return plane


if __name__ == "__main__":
    game = pyspiel.load_game("go", {"board_size": 9, "komi": 7.5})
    state = game.new_initial_state()
    history = [state.clone()]

    while not state.is_terminal():
        action = random.choice(state.legal_actions())
        state.apply_action(action)

        history.append(state.clone())

    # Verify padding, history order, current-state inclusion, and turn encoding.
    for obs_length in (1, 8):
        for t in range(len(history)):
            observations = encode_observations(history, t, obs_length)
            assert observations.shape == (obs_length * 2 + 1, 9, 9)
            assert observations.dtype == torch.float32
            assert torch.all((observations == 0) | (observations == 1))

            start = max(0, t - obs_length + 1)
            pad = obs_length - (t - start + 1)
            assert torch.count_nonzero(observations[:2 * pad]) == 0
            for i, snapshot in enumerate(history[start:t + 1], start=pad):
                assert torch.equal(observations[2 * i:2 * i + 2], encode_board(snapshot))

            turn = history[t].observation_tensor(0)[3 * 81:]
            assert torch.equal(observations[-1], torch.tensor(turn).reshape(9, 9))

    for t, obs_length, error in ((-1, 8, IndexError), (len(history), 8, IndexError),
                                (0, 0, ValueError)):
        try:
            encode_observations(history, t, obs_length)
        except error:
            pass
        else:
            raise AssertionError(f"Expected {error.__name__}")

    print("encode_observations: all checks passed")