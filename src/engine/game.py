from engine.card import Card
from engine.deck import HAND_SIZE
from engine.game_state import GameState, Phase


class Game:
    def __init__(self, num_players: int = 2, seed: int | None = None):
        self.state = GameState(num_players, seed)

    def get_legal_actions(self) -> list:
        """Return all legal actions for the current player."""
        if self.state.phase == Phase.ATTACK:
            if not self.state.table:
                # Must play a card
                return [("attack", card) for card in self.state.hands[self.state.attacker_id]]
            else:
                # Can throw in or stop
                legal_actions = []
                for card in self.state.hands[self.state.attacker_id]:
                    if card.rank in self.state.table_ranks and len(self.state.table) < self.state.throw_in_limit:
                        legal_actions.append(("throw_in", card))
                legal_actions.append(("stop", None))
                return legal_actions
        else:  # DEFEND phase
            legal_actions = []
            for atk_card, dfn_card in self.state.table:
                if dfn_card is None:  # Uncovered attack card
                    for card in self.state.hands[self.state.defender_id]:
                        if self._can_defend(atk_card, card):
                            legal_actions.append(("defend", atk_card, card))
            legal_actions.append(("pick_up", None))
            return legal_actions

    def attack(self, card: Card) -> None:
        """Attacker plays a card onto the table."""
        if self.state.phase != Phase.ATTACK:
            raise ValueError("Cannot attack during DEFEND phase.")

        self.state.hands[self.state.attacker_id].remove(card)
        self.state.table.append((card, None))
        self.state.phase = Phase.DEFEND
        self._check_win()

    def defend(self, attack_card: Card, defense_card: Card) -> None:
        """Defender covers a specific attack card."""


        if defense_card.beats(attack_card, self.state.trump_suit):
            self.state.hands[self.state.defender_id].remove(defense_card)
            idx = self.state.table.index((attack_card, None))
            self.state.table[idx] = (attack_card, defense_card)
        else:
            raise ValueError(f"Defense card does not beat that attack card {defense_card}, {attack_card}")
        self._check_win()

        self.state.phase = Phase.ATTACK
    def throw_in(self, card: Card) -> None:
        """Throw in an additional card matching a rank on the table.

        - Card rank must match a rank already on the table
        - Must not exceed throw_in_limit
        - Remove card from attacker's hand
        - Add (card, None) to table
        - Switch phase to DEFEND
        - Check for win (empty hand)

        TODO(human): Implement throw-in logic.
        """
        

    def pick_up(self) -> None:
        """Defender picks up all cards from the table.

        - All table cards (both attack and defense) go into defender's hand
        - Clear the table
        - End the bout (attacker stays the same)
        - Draw phase

        TODO(human): Implement pick-up logic.
        """

    def stop(self) -> None:
        """Attacker ends the bout (successful defense).

        - Clear the table (cards go to discard)
        - Set first_defense_completed = True
        - Draw phase
        - Swap roles

        TODO(human): Implement stop logic.
        """

    def _draw_phase(self) -> None:
        """Both players draw up to HAND_SIZE. Attacker draws first."""
        for player in [self.state.attacker_id, self.state.defender_id]:
            while len(self.state.hands[player]) < HAND_SIZE and self.state.deck.cards:
                self.state.hands[player].append(self.state.deck.cards.pop())

    def _check_win(self) -> None:
        """Check if any player has an empty hand when the deck is empty."""
        for i, hand in enumerate(self.state.hands):
            if not hand and not self.state.deck.cards:
                # This player wins
                self.state.winner = i
                break

