import random
from engine.game import Game

Action = tuple[any,...]

class RandomAgent:
    """Chooses uniformly from the current legal actions."""
    def __init__(self,seed: int| None = None):
          self.rng = random.Random(seed)
    def select_action(self, game: Game) -> Action:
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions available.")
        return self.rng.choice(legal_actions)
    

class HeuristicAgent:
    """Simple rule-based Durak agent.

    Strategy:
    - Attack with the lowest non-trump card if possible
    - Throw in with the lowest non-trump card if possible
    - Defend with the cheapest card that works
    - Prefer picking up over spending a high trump to defend
    - Stop instead of throwing in if no sensible throw-in exists
    """

    def select_action(self, game: Game) -> Action:
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions available.")
        
        phase = game.state.phase.name

        if phase == 'ATTACK':
            attack_actions = [a for a in legal_actions if a[0] == "attack"]
            if attack_actions:
                return self._choose_attack(game, attack_actions)
            
            throw_in_actions = [a for a in legal_actions if a[0] == "throw_in"]
            if throw_in_actions:
                chosen = self._choose_throw_in(game, throw_in_actions)
                if chosen is not None:
                    return chosen
                
            stop_actions = [a for a in legal_actions if a[0] == "stop"]
            if stop_actions:
                return stop_actions[0]

            return legal_actions[0]
        
        defend_actions = [a for a in legal_actions if a[0] == "defend"]
        pickup_actions = [a for a in legal_actions if a[0] == "pick_up"]

        if defend_actions:
            chosen_defense = self._choose_defense(game, defend_actions)
            if chosen_defense is not None:
                return chosen_defense
            
        if pickup_actions:
            return pickup_actions[0]

        return legal_actions[0]
    
    def _choose_attack(self, game: Game, actions: list[Action]) -> Action:
        trump = game.state.trump_suit

        non_trumps = [a for a in actions if a[1].suit != trump]
        if non_trumps:
            return min(non_trumps, key=lambda a: (a[1].rank, a[1].suit))

        return min(actions, key=lambda a: (a[1].rank, a[1].suit))
    
    def _choose_throw_in(self, game: Game, actions: list[Action]) -> Action | None:
        trump = game.state.trump_suit

        non_trumps = [a for a in actions if a[1].suit != trump]
        if non_trumps:
            return min(non_trumps, key=lambda a: (a[1].rank, a[1].suit))

        lowest_trump_throw = min(actions, key=lambda a: (a[1].rank, a[1].suit))
        if lowest_trump_throw[1].rank >= 11:
            stop_actions = [a for a in game.get_legal_actions() if a[0] == "stop"]
            if stop_actions:
                return None

        return lowest_trump_throw
    
    def _choose_defense(self, game: Game, actions: list[Action]) -> Action | None:
        trump = game.state.trump_suit

        def defense_cost(action: Action) -> tuple[int, int, int]:
            _, attack_card, defense_card = action

            uses_trump = 1 if defense_card.suit == trump else 0
            waste_trump = 1 if defense_card.suit == trump and attack_card.suit != trump else 0

            return (uses_trump, waste_trump, defense_card.rank)

        best = min(actions, key=defense_cost)
        _, attack_card, defense_card = best

        if (
            defense_card.suit == trump
            and attack_card.suit != trump
            and defense_card.rank >= 12
            and any(a[0] == "pick_up" for a in game.get_legal_actions())
        ):
            return None

        return best        
