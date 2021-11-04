from hanabi.objects import Card, Color

class Colorfield:
    def __init__(self, color:Color):

        self.color = color
        self.cards = []

    def is_able_to_add(self, card: Card):
        if self.color != card.color:
            return False

        if len(self.cards) + 1 != card.number:
            return False
        return True

    def add_card(self, card: Card):
        if not self.is_able_to_add():
            raise RuntimeError(
                "card type is mistaken"
                            )
        self.cards.append(card)



