"""Run mcts.py and save search iterations and played moves as a single SVG image.

python mcts_visualize.py --moves 3 --simulations 800 --iteration 4
Requires only open_spiel; SVG images open in any browser.
"""

import argparse
from html import escape
from pathlib import Path
import random
import textwrap

import pyspiel
from mcts.mcts import Node, mcts, ucb_score


def board_svg(position, board_size):
    parts = []

    def text(x, y, value, size=13, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" '
                     f'text-anchor="{anchor}">{escape(str(value))}</text>')

    # Observation planes are Black, White, empty, and side to play.
    cells = list(position.observation_tensor(0))
    count = board_size * board_size
    spacing = 210 / (board_size - 1)
    parts.append('<rect x="80" y="72" width="250" height="250" '
                 'rx="8" fill="#e7c995"/>')
    for row in range(board_size):
        v = 100 + row * spacing
        parts.append(f'<path d="M100 {v}H310 M{v} 100V310" '
                     'stroke="#846b45" fill="none"/>')
        text(v, 87, "ABCDEFGHJKLMNOPQRST"[row], size=11, anchor="middle")
        text(67, v + 4, board_size - row, size=11, anchor="middle")
    for index in range(count):
        if cells[index] or cells[count + index]:
            color = "#17212e" if cells[index] else "#ffffff"
            x = 100 + (index % board_size) * spacing
            # OpenSpiel indexes from A1 at the bottom left.
            y = 310 - (index // board_size) * spacing
            parts.append(f'<circle cx="{x}" cy="{y}" r="{spacing * .38}" '
                         f'fill="{color}" stroke="#344054"/>')
    if position.history() and position.history()[-1] < count:
        last = position.history()[-1]
        x = 100 + (last % board_size) * spacing
        y = 310 - (last // board_size) * spacing
        parts.append(f'<circle cx="{x}" cy="{y}" r="{spacing * .14}" '
                     'fill="none" stroke="#d97706" stroke-width="2"/>')
    return "".join(parts)


def visualize(board_size=5, iterations=4, seed=0, output="mcts_visualize.svg", moves=1,
              simulations=800):
    if simulations < 1 or iterations < 1:
        raise ValueError("simulations and iterations must be positive")
    shown = min(iterations, simulations)
    state = pyspiel.load_game("go", {"board_size": board_size}).new_initial_state()
    rows = []
    rng = random.Random(seed)
    panel_width = max(440, (shown + 1) * 100)

    selection_scores = {}

    def record_selection(parent, action, child):
        if final_stats.get("iterations", 0) >= shown:
            return
        selection_scores.update((id(c), ucb_score(parent, c))
                                for c in parent.children.values())

    def score(parent, child):
        return selection_scores.get(id(child), ucb_score(parent, child))

    def initialization(root, returns):
        parts = [board_svg(state, board_size)]

        def text(y, value, size=13, x=20, anchor="start"):
            parts.append(f'<text x="{x}" y="{y}" font-size="{size}" '
                         f'text-anchor="{anchor}">'
                         f'{escape(str(value))}</text>')

        text(30, f"Move {move + 1} / Initialization", 22)
        text(55, "Before iteration 1: evaluate the root")
        text(355, f"Random rollout: Black {returns[0]:+g}, White {returns[1]:+g}")
        x = panel_width / 2
        for cx in (x - 105, x + 105):
            parts.append(f'<path d="M{x} 430L{cx} 565" stroke="#c5cbd3"/>')
        parts.append(f'<circle cx="{x}" cy="430" r="32" fill="white" '
                     'stroke="#98a2b3" stroke-width="2"/>')
        text(387, "Root", 11, x, "middle")
        for offset, label in zip((-17, -5, 7, 19),
                                 (f"N {root.visit_count}", f"W {root.value_sum:+g}",
                                  f"Q {root.value():+.1f}", "PUCT n/a")):
            text(430 + offset, label, 8, x, "middle")
        prior = 1 / len(root.children)
        before = ucb_score(root, Node(root.player, prior=prior))
        priors = [child.prior for child in root.children.values()]
        scores = [ucb_score(root, child) for child in root.children.values()]
        groups = ((x - 105, "Before noise", [prior], [before], "#98a2b3"),
                  (x + 105, "After noise", priors, scores, "#d97706"))
        for cx, label, preferences, values, color in groups:
            parts.append(f'<circle cx="{cx}" cy="565" r="48" fill="white" '
                         f'stroke="{color}" stroke-width="2" stroke-dasharray="3 3"/>')
            text(504, label, 11, cx, "middle")
            for offset, value in zip((-29, -15, -2, 12, 25),
                                     (f"+{len(root.children)}", "prior",
                                      f"{min(preferences):.3f}..{max(preferences):.3f}",
                                      "PUCT", f"{min(values):.3f}..{max(values):.3f}")):
                text(565 + offset, value, 9, cx, "middle")
        panels.append("".join(parts))
        heights.append(640)

    def snapshot(iteration, root, position, path, returns):
        final_stats.update(root=root, iterations=iteration)
        if iteration > shown:
            return
        parts = []

        def text(x, y, value, size=13, color="#243247", anchor="start"):
            parts.append(f'<text x="{x}" y="{y}" font-size="{size}" '
                         f'fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')

        text(20, 30, f"Move {move + 1} / Iteration {iteration}", 22)
        text(20, 55, "1. Select a leaf; 2. Expand its legal moves")
        parts.append(board_svg(position, board_size))

        labels = []
        replay = state.clone()
        for parent, child in zip(path, path[1:]):
            action = next(a for a, c in parent.children.items() if c is child)
            labels.append(replay.action_to_string(replay.current_player(), action))
            replay.apply_action(action)
        text(20, 350, "Selected path:")
        for line, label in enumerate(labels):
            text(35, 372 + line * 19, label)
        y = 385 + len(labels) * 19
        text(20, y, f"3. Random rollout: Black {returns[0]:+g}, White {returns[1]:+g}")
        text(20, y + 24, "4. Back up results along the selected path")
        y += 55
        text(20, y, "Tree after this iteration", 15)
        y += 16

        # Lay out the visible tree by assigning equal space to each leaf.
        nodes, edges = [], []
        leaf = 0
        tree_top = y + 45

        def tree(node, board, depth=0, label="Root", puct="n/a"):
            nonlocal leaf
            children = []
            for action, child in node.children.items():
                if child.visit_count:
                    next_board = board.clone()
                    next_board.apply_action(action)
                    name = board.action_to_string(board.current_player(), action)
                    children.append(tree(child, next_board, depth + 1, name, f"{score(node, child):.3f}"))
            unvisited = sum(c.visit_count == 0 for c in node.children.values())
            if unvisited:
                scores = [score(node, c) for c in node.children.values() if not c.visit_count]
                group = (leaf, depth + 1, None, str(unvisited), node is path[-1],
                         f"{min(scores):.3f}..{max(scores):.3f}")
                leaf += 1
                nodes.append(group)
                children.append(group)
            x = (children[0][0] + children[-1][0]) / 2 if children else leaf
            if not children:
                leaf += 1
            item = (x, depth, node, label, False, puct)
            nodes.append(item)
            edges.extend((item, child) for child in children)
            return item

        tree(root, state)

        def xy(item):
            return (50 + item[0] * (panel_width - 100) / max(1, leaf - 1),
                    tree_top + item[1] * 125)

        selected_ids = {id(node) for node in path}
        for parent, child in edges:
            x1, y1 = xy(parent)
            x2, y2 = xy(child)
            selected = child[2] is not None and id(child[2]) in selected_ids
            color = "#007d79" if selected else "#c5cbd3"
            parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color}" '
                         f'stroke-width="{3 if selected else 1}"/>')
        for item in nodes:
            x, cy = xy(item)
            _, _, node, label, new_children, puct = item
            selected = node is not None and id(node) in selected_ids
            new = node is path[-1] and node.visit_count == 1
            color = "#d97706" if new or new_children else "#007d79" if selected else "#98a2b3"
            dash = ' stroke-dasharray="3 3"' if node is None else ""
            parts.append(f'<circle cx="{x}" cy="{cy}" r="32" fill="white" '
                         f'stroke="{color}" stroke-width="2"{dash}/>')
            if node is None:
                text(x, cy - 12, f"+{label}", size=12, anchor="middle")
                text(x, cy + 1, "PUCT", size=9, anchor="middle")
                text(x, cy + 14, puct, size=8, anchor="middle")
                text(x, cy + 49, "new replies" if new_children else "unvisited",
                     size=10, color=color, anchor="middle")
            else:
                text(x, cy - 43, label, size=11, color=color, anchor="middle")
                for offset, value in zip((-17, -5, 7, 19),
                                         (f"N {node.visit_count}", f"W {node.value_sum:+g}",
                                          f"Q {node.value():+.1f}", f"PUCT {puct}")):
                    text(x, cy + offset, value, size=8, anchor="middle")
                if new:
                    text(x, cy + 49, "NEW", size=10, color=color, anchor="middle")
        y = tree_top + max(item[1] for item in nodes) * 125 + 65
        selection_scores.clear()
        panels.append("".join(parts))
        heights.append(y + 20)

    for move in range(moves):
        if state.is_terminal():
            break
        panels, heights = [], []
        final_stats = {}
        temperature = 1.0 if len(state.history()) < 30 else 0.0
        action, visit_distribution, root_value = mcts(state, simulations=simulations, rng=rng,
                      on_iteration=snapshot, temperature=temperature,
                      on_selection=record_selection, on_initialization=initialization)
        chosen = state.action_to_string(state.current_player(), action)
        state.apply_action(action)
        # The final column is the real board after applying the sampled action.
        caption = f"Selected: {chosen} | {final_stats['iterations']} iterations"
        result = (f'<text x="20" y="30" font-size="22">Move {move + 1}: played</text>'
                  f'<text x="20" y="55" font-size="14">{escape(caption)}</text>'
                  + board_svg(state, board_size))
        parts = [result]

        def result_text(y, value, size=13, color="#243247"):
            parts.append(f'<text x="20" y="{y}" font-size="{size}" '
                         f'fill="{color}">{escape(str(value))}</text>')

        result_text(355, f"Final move probabilities (T={temperature:g})", 15)
        result_text(379, "P(move) = visits / total visits" if temperature
                    else "P(most visited) = 100%")
        root = final_stats["root"]
        best = max(root.children, key=lambda a: root.children[a].visit_count)
        y = 405
        for candidate, child in root.children.items():
            if child.visit_count:
                probability = (visit_distribution[candidate] if temperature
                               else float(candidate == best))
                label = state.action_to_string(root.player, candidate)
                selected = candidate == action
                color = "#d97706" if selected else "#007d79"
                marker = " *" if selected else ""
                result_text(y, f"{label}: N={child.visit_count}, {probability:.1%}{marker}",
                            color=color)
                parts.append(f'<rect x="310" y="{y - 11}" width="{100 * probability}" '
                             f'height="12" fill="{color}" rx="2"/>')
                y += 22
        result_text(y, "All unvisited moves: 0%", color="#667085")
        result_text(y + 22, f"Root value: {root_value:+.3f}")
        result_text(y + 44, "* Selected action", color="#d97706")
        if state.is_terminal():
            y += 24
            result_text(y + 44, f"Game over: {state.returns()}")
        panels.append("".join(parts))
        rows.append((panels, max(heights + [y + 72])))

    width = panel_width * (shown + 2)
    title = (f"{board_size} x {board_size} Go | {len(rows)} played moves | "
             f"{simulations} simulations per move; first {shown} shown")
    header = [title, "",
              "How to read this image",
              "Read each row from left to right: starting evaluation, practice searches, then the move actually played.",
              "Search boards show possible future positions. A rollout plays random legal moves to the end to estimate a result.",
              "Teal marks the selected path. Orange marks a newly explored move or new replies; a board ring marks the last move.",
              "Dashed circles group moves that have not been tried. Their PUCT label shows the lowest and highest scores.", "",
              "What the numbers mean",
              "N = number of visits. W = total result (+1 for a win, -1 for a loss). Q = average result, W / N.",
              "Each move's results are counted for the player who made it. The root starts with one visit from its first rollout.",
              "PUCT scores explain selection before results are added; N, W and Q show the updated totals afterward.",
              "New replies show their current scores. The root has no PUCT score because it has no parent.", "",
              "How the search chooses moves",
              "Higher PUCT scores balance good past results with trying less-explored moves.",
              "A prior is a move's starting preference. All legal moves start equally; noise randomly adjusts root preferences.",
              "Noise is added once per played move and stays fixed throughout its search: prior = 0.75 * uniform + 0.25 * noise.",
              "The played board and final probabilities use every simulation, including those not shown.", "",
              "PUCT formula",
              "PUCT = V + exploration; V = (Q + 1) / 2 for tried moves, otherwise 0.",
              "exploration = [ln((Np + 19653) / 19652) + 1.25] * sqrt(Np) / (Nc + 1) * prior",
              "Np = parent visits; Nc = child visits. Exploration favors higher-prior, less-visited moves."]
    header = [part for line in header
              for part in (textwrap.wrap(line, width=min(120, max(30, (width - 40) // 9))) or [""])]
    top = len(header) * 21 + 25
    height = sum(height for _, height in rows) + top
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="monospace">',
           '<rect width="100%" height="100%" fill="#f5f7fa"/>']
    for i, line in enumerate(header):
        svg.append(f'<text x="20" y="{24 + i * 21}" font-size="14">{escape(line)}</text>')
    for panels, row_height in rows:
        for i, panel in enumerate(panels):
            svg.append(f'<g transform="translate({i * panel_width},{top})">{panel}</g>')
        top += row_height
    Path(output).write_text("".join(svg) + "</svg>", encoding="utf-8")
    print(f"Saved {Path(output).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--iteration", "--iterations", dest="iterations", type=int, default=4,
                        help="show the first N search iterations per played move")
    parser.add_argument("--simulations", type=int, default=800,
                        help="total MCTS simulations per played move (default: 800)")
    parser.add_argument("--moves", type=int, default=1, help="number of actual moves to play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="mcts_visualize.svg")
    args = parser.parse_args()
    if not 2 <= args.board_size <= 19 or not 1 <= args.iterations <= 12:
        parser.error("board-size must be 2..19 and iterations must be 1..12")
    if args.moves < 1:
        parser.error("moves must be positive")
    if args.simulations < 1:
        parser.error("simulations must be positive")
    visualize(args.board_size, args.iterations, args.seed, args.output, args.moves,
              args.simulations)
