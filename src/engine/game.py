from engine.card import Card
from engine.deck import HAND_SIZE
from engine.game_state import GameState, Phase


class Game:
    def __init__(self, num_players: int = 2, seed: int | None = None):
        self.state = GameState(num_players, seed)

    def get_legal_actions(self) -> list:
        """Return all legal actions for the current player.

        ATTACK phase (empty table): must play a card — returns [("attack", card), ...]
        ATTACK phase (cards on table): can throw in matching ranks or stop (only if all
            attack cards are covered) — returns [("throw_in", card), ..., ("stop", None)]
        DEFEND phase: can cover any uncovered attack card, or pick up everything —
            returns [("defend", atk_card, def_card), ..., ("pick_up", None)]
        """
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
                if not self.state.uncovered_cards:
                    legal_actions.append(("stop", None))
                return legal_actions
        else:  # DEFEND phase
            legal_actions = []
            for atk_card, dfn_card in self.state.table:
                if dfn_card is None:  # Uncovered attack card
                    for card in self.state.hands[self.state.defender_id]:
                        if card.beats(atk_card, self.state.trump_suit):
                            legal_actions.append(("defend", atk_card, card))
            legal_actions.append(("pick_up", None))
            return legal_actions

    def attack(self, card: Card) -> None:
        """Play a card from the attacker's hand onto the table to start or continue a bout."""
        if self.state.phase != Phase.ATTACK:
            raise ValueError("Cannot attack during DEFEND phase.")

        self.state.hands[self.state.attacker_id].remove(card)
        self.state.table.append((card, None))
        self.state.phase = Phase.DEFEND
        self._check_win()

    def defend(self, attack_card: Card, defense_card: Card) -> None:
        """Cover a specific attack card with a defense card.

        defense_card must beat attack_card (higher rank same suit, or any trump vs non-trump).
        After covering, phase returns to ATTACK so the attacker can throw in or stop.
        """
        if defense_card.beats(attack_card, self.state.trump_suit):
            self.state.hands[self.state.defender_id].remove(defense_card)
            idx = self.state.table.index((attack_card, None))
            self.state.table[idx] = (attack_card, defense_card)
        else:
            raise ValueError(f"Defense card does not beat that attack card {defense_card}, {attack_card}")
        self._check_win()

        self.state.phase = Phase.ATTACK

    def throw_in(self, card: Card) -> None:
        """Throw an additional card onto the table after the defender has covered.

        The card's rank must match a rank already on the table (attack or defense side).
        Cannot exceed throw_in_limit (5 on the first bout, defender's hand size thereafter).
        """
        if self.state.phase != Phase.ATTACK:
            raise ValueError("Cannot throw in during DEFEND phase.")

        if card.rank not in self.state.table_ranks:
            raise ValueError("Card rank must match a rank on the table.")

        if len(self.state.table) >= self.state.throw_in_limit:
            raise ValueError("Throw-in limit reached.")
        if card not in self.state.hands[self.state.attacker_id]:
            raise ValueError("Card not in hand.")

        self.state.hands[self.state.attacker_id].remove(card)
        self.state.table.append((card, None))
        self.state.phase = Phase.DEFEND
        self._check_win()

    def pick_up(self) -> None:
        """Defender concedes and picks up all cards from the table.

        All table cards (attack and defense) go into the defender's hand.
        The defender who picked up is skipped — the next player after them becomes
        the new attacker, and the player after that becomes the new defender.
        """
        # Defender takes all cards
        for atk_card, dfn_card in self.state.table:
            self.state.hands[self.state.defender_id].append(atk_card)
            if dfn_card is not None:
                self.state.hands[self.state.defender_id].append(dfn_card)

        # Clear table
        self.state.table.clear()

        self.state.phase = Phase.ATTACK

        # Draw cards
        self._draw_phase()

        self.state.attacker_id = (self.state.defender_id + 1) % self.state.num_players
        self.state.defender_id = (self.state.defender_id + 2) % self.state.num_players

    def stop(self) -> None:
        """Attacker ends the bout after a successful defense.

        All table cards go to the discard pile. The defender becomes the new attacker,
        and the next player in the ring becomes the new defender.
        """
        for atk_card, dfn_card in self.state.table:
            self.state.discard_pile.append(atk_card)
            self.state.discard_pile.append(dfn_card)

        self.state.table.clear()

        self.state.first_defense_completed = True

        self._draw_phase()
        self.state.phase = Phase.ATTACK

        self.state.attacker_id = self.state.defender_id
        self.state.defender_id = (self.state.defender_id + 1) % self.state.num_players

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
