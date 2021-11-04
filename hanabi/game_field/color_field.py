from hanabi.objects import Card, Color


class ColorField:
    def __init__(self, color: Color):
        self.color = color
        self.cards = []

    def is_able_to_add(self, card: Card) -> bool:
        if self.color != card.color:
            return False

        top_card_number = self.cards[-1].number
        if top_card_number.value + 1 != card.number.value:
            return False
        return True

    def add_card(self, card: Card):
        if not self.is_able_to_add():
            raise RuntimeError("card type is mistaken")
        self.cards.append(card)
