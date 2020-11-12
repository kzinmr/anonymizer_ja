import regex as re
from typing import List
import MeCab

from pos_dictionary import pos2id


class MecabMatcher:
    def __init__(self, pattern: str = "名詞-固有名詞-人名"):
        self.tagger = MeCab.Tagger(
            "-r /dev/null -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"
        )
        self.pattern = pattern

    @staticmethod
    def __split(t: str) -> List[str]:
        if len(t.split("\t")) == 2:
            surface, features = t.split("\t")
            if len(features.split(",")) > 1:
                features = features.split(",")
            elif len(features.split("|")) > 1:
                features = features.split("|")
            else:
                features = [features]
            return [surface] + features
        else:
            print("ERROR: invalid tagline")
            raise

    def convert_string(self, pos_raw_list: List[str]) -> str:
        pos_list = ["-".join(t[1:5]) for t in pos_raw_list]
        return "".join([pos2id[p] for p in pos_list])

    def convert_pattern(self, string):
        for i, j in pos2id.items():
            string = string.replace(i, "({})".format(j))
        return string

    @staticmethod
    def fuzzy_match(x: str, text: str) -> List[str]:
        fuzzy_pattern = re.compile("(?:%s){i<=3:\s}" % (re.escape(x)))
        text_spans = [m.span() for m in fuzzy_pattern.finditer(text)]
        if len(text_spans) > 0:
            return [text[s:e] for s, e in text_spans]
        else:
            return []

    def parse(self, text: str) -> List[List[str]]:
        pos_raw = [t for t in self.tagger.parse(text).split("\n") if "\t" in t]
        pos_raw_list = [self.__split(t) for t in pos_raw]
        tokens = [t[0] for t in pos_raw_list]

        # convert string into id sequence
        pos_seq = self.convert_string(pos_raw_list)

        # convert pattern into id sequence
        reg_pattern = self.convert_pattern(self.pattern)

        output = []
        for m in re.finditer(reg_pattern, pos_seq):
            start = int(m.start() / 4)
            end = int(start + len(m.group()) / 4)
            if not len(m.group()):
                continue
            # token列をjoinしてfuzzyマッチで原文復元
            x = "".join(tokens[start:end])
            source_texts = self.fuzzy_match(x, text)
            if source_texts:
                output.append([s.strip() for s in source_texts])

        return output
