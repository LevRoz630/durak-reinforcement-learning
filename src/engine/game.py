from engine.card import Card
from engine.deck import HAND_SIZE
from engine.game_state import GameState, Phase


class Game:
    def __init__(self, num_players: int = 2, seed: int | None = None):
        self.state = GameState(num_players, seed)

    def get_legal_actions(self) -> list:
        """Return all legal actions for the current player.

        During ATTACK phase:
        - Table empty: must play a card (any card in hand)
        - Table not empty: can throw in (matching rank) or stop
        - Respect throw_in_limit

        During DEFEND phase:
        - For each uncovered card: any card in hand that beats it
        - Can always pick up instead

        TODO(human): Implement legal action generation.
        """
        raise NotImplementedError

    def attack(self, card: Card) -> None:
        """Attacker plays a card onto the table.

        - Remove card from attacker's hand
        - Add (card, None) to table
        - Switch phase to DEFEND
        - Check for win (empty hand)

        TODO(human): Implement attack logic.
        """
        raise NotImplementedError

    def defend(self, attack_card: Card, defense_card: Card) -> None:
        """Defender covers a specific attack card.

        - Validate defense_card beats attack_card
        - Remove defense_card from defender's hand
        - Update the table pair: replace None with defense_card
        - If no uncovered cards remain, switch phase to ATTACK
        - Check for win (empty hand)

        TODO(human): Implement defend logic.
        """
        raise NotImplementedError

    def throw_in(self, card: Card) -> None:
        """Attacker throws in an additional card.

        - Card rank must match a rank already on the table
        - Must not exceed throw_in_limit
        - Remove card from attacker's hand
        - Add (card, None) to table
        - Switch phase to DEFEND
        - Check for win (empty hand)

        TODO(human): Implement throw-in logic.
        """
        raise NotImplementedError

    def pick_up(self) -> None:
        """Defender picks up all cards from the table.

        - All table cards (both attack and defense) go into defender's hand
        - Clear the table
        - End the bout (attacker stays the same)
        - Draw phase

        TODO(human): Implement pick-up logic.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Attacker ends the bout (successful defense).

        - Clear the table (cards go to discard)
        - Set first_defense_completed = True
        - Draw phase
        - Swap roles

        TODO(human): Implement stop logic.
        """
        raise NotImplementedError

    def _draw_phase(self) -> None:
        """Both players draw up to HAND_SIZE. Attacker draws first.

        TODO(human): Implement draw phase.
        """
        raise NotImplementedError

    def _check_win(self) -> None:
        """Check if any player has an empty hand (instant win).

        Only relevant when the deck is also empty.

        TODO(human): Implement win detection.
        """
        raise NotImplementedError
