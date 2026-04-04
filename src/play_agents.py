from engine.game import Game
from agents import RandomAgent, HeuristicAgent


def apply_action(game, action):
    if action[0] == "attack":
        game.attack(action[1])
    elif action[0] == "throw_in":
        game.throw_in(action[1])
    elif action[0] == "defend":
        game.defend(action[1], action[2])
    elif action[0] == "stop":
        game.stop()
    elif action[0] == "pick_up":
        game.pick_up()
    else:
        raise ValueError(f"Unknown action: {action}")


def play_game(seed=None, verbose=False):
    game = Game(num_players=2, seed=seed)
    agent1 = HeuristicAgent()
    agent2 = RandomAgent()

    step = 0

    while game.state.winner is None:
        step += 1

        current = (
            game.state.attacker_id
            if game.state.phase.name == "ATTACK"
            else game.state.defender_id
        )

        agent = agent1 if current == 0 else agent2
        action = agent.select_action(game)

        if verbose:
            print(f"\nStep {step}")
            print(f"Player {current + 1} action: {action}")

        apply_action(game, action)

    if verbose:
        print(f"\nWinner: Player {game.state.winner + 1}")

    return game.state.winner


def run_matches(num_games=100):
    wins = {0: 0, 1: 0}

    for i in range(num_games):
        winner = play_game(seed=i)
        wins[winner] += 1

    print("\nResults after", num_games, "games:")
    print(f"HeuristicAgent (Player 1): {wins[0]} wins")
    print(f"RandomAgent    (Player 2): {wins[1]} wins")


if __name__ == "__main__":
    print("Running single game (verbose)...")
    play_game(seed=42, verbose=True)

    print("\nRunning batch simulation...")
    run_matches(50)
