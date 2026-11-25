"""MuZero-style MCTS with uniform priors and random rollouts (no networks).

Install: python -m pip install open_spiel
Example: python mcts/mcts.py --board-size 5 --simulations 100
"""

import argparse
import math
import random
from dataclasses import dataclass, field

import pyspiel


@dataclass
class Node:
    # Values are from the perspective of the player who chose this move.
    player: int
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)

    def expanded(self):
        return bool(self.children)

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def ucb_score(parent, child, pb_c_init=1.25, pb_c_base=19652):
    """MuZero's PUCT formula, using known return bounds [-1, 1]."""
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    value_score = (child.value() + 1) / 2 if child.visit_count else 0.0
    return value_score + pb_c * child.prior


def select_child(node):
    return max(node.children.items(), key=lambda item: (ucb_score(node, item[1]), item[0]))


def add_exploration_noise(root, rng, alpha=0.03, fraction=0.25):
    # Normalized Gamma samples give MuZero's Dirichlet root noise.
    noise = [rng.gammavariate(alpha, 1) for _ in root.children]
    total = sum(noise)
    for child, sample in zip(root.children.values(), noise):
        child.prior = (1 - fraction) * child.prior + fraction * sample / total


def select_action(root, temperature, rng):
    if temperature == 0:
        return max(root.children, key=lambda action: root.children[action].visit_count)
    actions = list(root.children)
    maximum = max(child.visit_count for child in root.children.values())
    weights = [(root.children[a].visit_count / maximum) ** (1 / temperature)
               for a in actions]
    return rng.choices(actions, weights=weights)[0]


def selection(root, state, on_selection=None):
    """Follow PUCT scores to a leaf, using OpenSpiel for exact transitions."""
    node = root
    path = [node]
    while node.expanded():
        action, child = select_child(node)
        if on_selection is not None:
            on_selection(node, action, child)
        node = child
        state.apply_action(action)
        path.append(node)
    return path


def expansion(node, state, rng):
    """Create all legal children with uniform priors instead of a policy net."""
    actions = list(state.legal_actions())
    rng.shuffle(actions)
    for action in actions:
        node.children[action] = Node(state.current_player(), 1 / len(actions))


def simulation(state, rng):
    """Replace the network's leaf value with a random rollout's final returns."""
    state = state.clone()
    while not state.is_terminal():
        state.apply_action(rng.choice(state.legal_actions()))
    return state.returns()


def backpropagation(path, returns):
    """Back up terminal returns; player indexing handles opponent perspective."""
    for node in reversed(path):
        node.value_sum += returns[node.player]
        node.visit_count += 1


def run_mcts(root, state, simulations, rng, on_iteration=None, on_selection=None):
    for iteration in range(1, simulations + 1):
        position = state.clone()
        path = selection(root, position, on_selection)
        if not position.is_terminal():
            expansion(path[-1], position, rng)
        returns = simulation(position, rng)
        backpropagation(path, returns)
        if on_iteration is not None:
            on_iteration(iteration, root, position, path, returns)


def mcts(state, simulations=100, rng=None, on_iteration=None, temperature=None,
         root_noise=True, on_selection=None, on_initialization=None):
    """Search finite deterministic, sequential, perfect-information games.

    Assumes terminal returns in [-1, 1] (as in Go), with no discounting.
    Returns (action, {legal_action: visits / total_visits}, root_value).
    The visit distribution is untempered; root_value is for the root player.
    """
    if state.is_terminal():
        raise ValueError("Cannot search a terminal state")
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if temperature is None:
        temperature = 1.0 if len(state.history()) < 30 else 0.0
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    rng = rng if rng is not None else random.Random()
    root = Node(state.current_player())
    expansion(root, state, rng)
    # MuZero also evaluates and backs up the root before running simulations.
    initial_returns = simulation(state, rng)
    backpropagation([root], initial_returns)
    if root_noise:
        add_exploration_noise(root, rng)
    if on_initialization is not None:
        on_initialization(root, initial_returns)
    run_mcts(root, state, simulations, rng, on_iteration, on_selection)
    total = sum(child.visit_count for child in root.children.values())
    visits = {action: child.visit_count / total for action, child in root.children.items()}
    return select_action(root, temperature, rng), visits, root.value()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.board_size < 2 or args.board_size > 19:
        parser.error("board-size must be between 2 and 19")
    if args.simulations < 1:
        parser.error("simulations must be positive")

    game = pyspiel.load_game("go", {"board_size": args.board_size})
    state = game.new_initial_state()
    rng = random.Random(args.seed)
    print(state)
    while not state.is_terminal():
        action, _, _ = mcts(state, simulations=args.simulations, rng=rng)
        print(state.action_to_string(state.current_player(), action))
        state.apply_action(action)
        print(state)
    print("Returns (Black, White):", state.returns())


if __name__ == "__main__":
    main()
