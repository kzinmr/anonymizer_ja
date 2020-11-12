from mecab_matcher import MecabMatcher


class PersonMatcher(MecabMatcher):
    def __init__(self):
        super().__init__("名詞-固有名詞-人名")

    # def parse(self, text: str):
    #     return self.parse_tokens(text)
