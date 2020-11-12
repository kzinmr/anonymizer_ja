from typing import Dict, Generator, List, Tuple

import regex as re
import pandas as pd


class AddressMatcher:
    def __init__(self):
        self.pref_city_addr_matcher = self.geolonia_dataset()

    @staticmethod
    def geolonia_dataset() -> Dict:
        url = "https://raw.githubusercontent.com/geolonia/japanese-addresses/master/data/latest.csv"
        address_df = pd.read_csv(url)
        pref_city_addr = address_df[["都道府県名", "市区町村名", "大字町丁目名"]]
        pref_city_addr_matcher = {
            pref: {
                city: set(addr["大字町丁目名"])
                for city, addr in city_addr.set_index("市区町村名").groupby(level=0)
            }
            for pref, city_addr in pref_city_addr.set_index("都道府県名").groupby(level=0)
        }
        return pref_city_addr_matcher

    @staticmethod
    def ksuji(match):
        tt_ksuji = str.maketrans("123456789", "一二三四五六七八九")
        if match is not None:
            ks = match.group(1)
            return f"{ks.translate(tt_ksuji)}丁目"
        else:
            return None

    def match_address(
        self, text: str, partial: bool = True
    ) -> Generator[Tuple[int, int], None, None]:
        text = re.sub(
            "([1-9])丁目", self.ksuji, text
        )  # unicodedata.normalize('NFKC', text)
        for pref in self.pref_city_addr_matcher:
            for m in re.finditer(pref, text):
                if m is not None:
                    begin, end = m.span()
                    pref_text = text[end:]
                    for city in self.pref_city_addr_matcher[pref]:
                        if pref_text.startswith(city):
                            city_text = pref_text[len(city) :]
                            for addr in self.pref_city_addr_matcher[pref][city]:
                                if city_text.startswith(addr):
                                    addr_span = (begin, end + len(city) + len(addr))
                                    yield addr_span
                                    break
                            else:
                                if partial:
                                    addr_partial = {
                                        re.sub("[一二三四五六七八九]丁目", "", addr)
                                        for addr in self.pref_city_addr_matcher[pref][
                                            city
                                        ]
                                    }
                                    for addr in addr_partial:
                                        if city_text.startswith(addr):
                                            addr_span = (
                                                begin,
                                                end + len(city) + len(addr),
                                            )
                                            yield addr_span

    @staticmethod
    def adjunct_address(
        text: str, spans: Tuple[int, int], margin: int = 10
    ) -> Generator[Tuple[int, int], None, None]:
        address_numbers = re.compile(
            r"^\p{N}+-\p{N}+-\p{N}+|"
            "^\p{N}+-\p{N}+|"
            "^\p{N}+丁目\p{N}+番地?\p{N}+号?|"
            "^\p{N}+丁目\p{N}+|"
            "^\p{N}+番地?\p{N}+号?|"
            "^\p{N}+番地?|"
            "^(\p{N}+)[^\-]"
        )
        for b, e in spans:
            text_fragment = text[b : e + margin]
            m = address_numbers.search(text_fragment[e - b :])
            if m:
                match = m.group() if m.group(1) is None else m.group(1)
                yield (b, e + len(match))
            else:
                yield (b, e)

    def match_spans(self, text: str) -> List[Tuple[int, int]]:
        return list(self.adjunct_address(text, list(self.match_address(text))))

    def parse(self, text: str) -> List[List[str]]:
        # TODO: 名寄せ機能
        return [[text[s:e].strip()] for s, e in self.match_spans(text)]
