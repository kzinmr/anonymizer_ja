from mecab_matcher import MecabMatcher


class CompanyMatcher(MecabMatcher):
    def __init__(self):
        super().__init__("名詞-固有名詞-組織")
