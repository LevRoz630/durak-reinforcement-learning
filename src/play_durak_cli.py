from engine.game import Game
from engine.game_state import Phase


def card_label(card) -> str:
    return f"{card.rank.name.title()} of {card.suit.name.title()}"


def print_state(game: Game) -> None:
    state = game.state
    print("\n" + "=" * 60)
    print(f"Trump suit: {state.trump_suit.name.title()}")
    print(f"Deck cards remaining: {len(state.deck.cards)}")
    print(f"Attacker: Player {state.attacker_id + 1}")
    print(f"Defender: Player {state.defender_id + 1}")
    print(f"Phase: {state.phase.name}")
    print("-" * 60)

    if not state.table:
        print("Table: empty")
    else:
        print("Table:")
        for i, (atk, dfn) in enumerate(state.table, start=1):
            if dfn is None:
                print(f"  {i}. Attack: {card_label(atk)} | Defense: [uncovered]")
            else:
                print(f"  {i}. Attack: {card_label(atk)} | Defense: {card_label(dfn)}")
    print("=" * 60)


def print_hand(game: Game, player_id: int) -> None:
    hand = game.state.hands[player_id]
    print(f"\nPlayer {player_id + 1} hand:")
    for i, card in enumerate(hand, start=1):
        print(f"  {i}. {card_label(card)}")


def pause_for_hidden_hand() -> None:
    input("\nPress Enter when the next player is ready to look at the screen...")


def choose_from_hand(game: Game, player_id: int, allowed_action_names: set[str]):
    legal = game.get_legal_actions()
    allowed_cards = {}

    if game.state.phase == Phase.ATTACK:
        for action in legal:
            if action[0] in allowed_action_names and len(action) >= 2:
                allowed_cards[action[1]] = action[0]
    else:
        for action in legal:
            if action[0] in allowed_action_names and len(action) >= 3:
                # For defend actions, map defense card to full action tuple.
                allowed_cards[action[2]] = action

    hand = game.state.hands[player_id]

    while True:
        print_hand(game, player_id)
        choice = input("\nEnter the card number: ").strip()

        if not choice.isdigit():
            print("Please enter a valid card number.")
            continue

        idx = int(choice) - 1
        if idx < 0 or idx >= len(hand):
            print("That card number is out of range.")
            continue

        selected = hand[idx]

        if game.state.phase == Phase.ATTACK:
            if selected not in allowed_cards:
                print("That card is not a legal move right now.")
                continue
            return allowed_cards[selected], selected
        else:
            if selected not in allowed_cards:
                print("That card cannot defend any uncovered attack card.")
                continue
            return allowed_cards[selected]


def attacker_turn(game: Game) -> None:
    attacker = game.state.attacker_id

    while True:
        print_state(game)
        legal = game.get_legal_actions()
        action_names = {a[0] for a in legal}

        if not game.state.table:
            print(f"\nPlayer {attacker + 1}, you must attack.")
            action_name, card = choose_from_hand(game, attacker, {"attack"})
            game.attack(card)
            return

        print(f"\nPlayer {attacker + 1}, choose an action:")
        print("  1. Throw in a card")
        print("  2. Stop the bout")

        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            if "throw_in" not in action_names:
                print("You do not have any legal throw-in cards right now.")
                continue
            action_name, card = choose_from_hand(game, attacker, {"throw_in"})
            game.throw_in(card)
            return
        elif choice == "2":
            if "stop" not in action_names:
                print("You cannot stop right now.")
                continue
            game.stop()
            print("\nBout ended. Successful defense.")
            return
        else:
            print("Invalid choice. Please enter 1 or 2.")


def defender_turn(game: Game) -> None:
    defender = game.state.defender_id

    while True:
        print_state(game)
        legal = game.get_legal_actions()
        action_names = {a[0] for a in legal}

        print(f"\nPlayer {defender + 1}, choose an action:")
        if "defend" in action_names:
            print("  1. Defend with a card")
        print("  2. Pick up the table")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            if "defend" not in action_names:
                print("You do not have any legal defense cards.")
                continue
            defend_action = choose_from_hand(game, defender, {"defend"})
            _, attack_card, defense_card = defend_action
            game.defend(attack_card, defense_card)
            return
        elif choice == "2":
            game.pick_up()
            print("\nDefender picked up the cards.")
            return
        else:
            print("Invalid choice.")


def print_winner(game: Game) -> None:
    print_state(game)
    if game.state.winner is not None:
        print(f"\nGame over. Player {game.state.winner + 1} wins!")
    else:
        print("\nGame ended without a winner.")


def main() -> None:
    print("Welcome to 2-player Durak")
    print("This uses your existing engine logic.")

    seed_text = input("Enter a seed for a repeatable shuffle, or press Enter for random: ").strip()
    seed = int(seed_text) if seed_text else None

    game = Game(num_players=2, seed=seed)

    while game.state.winner is None:
        current_player = game.state.attacker_id if game.state.phase == Phase.ATTACK else game.state.defender_id
        print(f"\nPass the screen to Player {current_player + 1}.")
        pause_for_hidden_hand()

        if game.state.phase == Phase.ATTACK:
            attacker_turn(game)
        else:
            defender_turn(game)

    print_winner(game)


if __name__ == "__main__":
    main()
