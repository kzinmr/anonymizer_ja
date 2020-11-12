import regex as re
from mecab_matcher import MecabMatcher


class CompanyMatcherV0(MecabMatcher):
    def __init__(self):
        super().__init__("名詞-固有名詞-組織")


re_banks = r"(バンク|信用組合|農業協同組合|組合連合会|労働金庫|信用金庫|中央金庫|損害保険|(株式会社.{1,30}(銀行|保険|証券|證券))|((銀行|保険|証券|證券)((株式|相互)会社)+))"


class CompanyMatcher(MecabMatcher):
    """前株後株など企業名らしいパターンを持つ名詞句を抽出
    NOTE: 一般の企業名抽出を解いてるわけではないので、応用依存の文脈による後処理を要する
    NOTE: "名詞,固有名詞,組織" の品詞列を頼りに抽出するのはリコール不足なので不採用
    """

    # dots = {"・", "•", "&"}

    def __init__(self):
        super().__init__("名詞|接頭詞|記号")

    def parse(self, text):
        np_list = super().parse(text)
        print(np_list)
        candidates_list = []
        for nounphrase in np_list:
            # 応用依存の前処理
            # nounphrase = re.sub(party_type, "", nounphrase)
            if re.search(
                r"((株式|特例有限|持分|合同|合資|合名|特定目的|特殊|相互|準備)会社|[\p{Han}\p{Latin}]{2,5}法人)",
                nounphrase,
            ):
                candidates_list.append(nounphrase)
            elif re.search(
                r"(リミテッド|組合連合会|(中央|信用)金庫|(株式|特例有限|持分|合同|合資|合名|特定目的|特殊|相互|準備)会社|[\p{Han}\p{Latin}]{2,5}法人)$",
                nounphrase,
            ):
                candidates_list.append(nounphrase)
        print(candidates_list)
        # 応用依存の後処理
        candidates_list_postprocessed = []
        for nounphrase in candidates_list:
            # 後ろの円
            nounphrase = re.sub(r"[\p{Numeric_Type=Numeric}\p{N}]*円?$", "", nounphrase)
            nounphrase = re.sub(
                r"((?P<year>\d{4})[-/年](?P<ysuf>[^0-9\-/年月日\s]+)?)?((?P<month>\d{1,2})[-/月](?P<msuf>[^0-9\-/年月日\s]+)?)?((?P<date>\d{1,2})[日]?(?P<dsuf>[^0-9\-/年月日\s]+)?)?",
                "",
                nounphrase,
            )
            # nounphrase = re.sub(suffixes, "", nounphrase)
            # nounphrase = re.sub(section, "", nounphrase)
            # FIXME: 株式会社名が連続する場合に前株など仮定しないと分割できない
            candidates_list_postprocessed.append(nounphrase)
        return candidates_list_postprocessed