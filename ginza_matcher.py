# Convert CoNLL2003-like column data to chunks and spans
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import json
import re
import MeCab
import pandas as pd
import requests
import spacy
import tokenizations
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU
from spacy.gold import biluo_tags_from_offsets, iob_to_biluo
from ginza.ene_ontonotes_mapper import ENE_ONTONOTES_MAPPING


@dataclass(frozen=True)
class Token:
    text: str
    label: str


@dataclass(frozen=True)
class Chunk:
    tokens: List[Token]
    label: str
    span: Tuple[int, int]

    def __iter__(self):
        for token in self.tokens:
            yield token


@dataclass(frozen=True)
class Sentence(Iterable[Token]):
    text: str
    tokens: List[Token]
    chunks: List[Chunk]

    def __init__(self, tokens):
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "text", "".join([token.text for token in self.tokens]))
        object.__setattr__(self, "chunks", self.__build_chunks(self.tokens))
        self.assert_spans()

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __build_chunks(self, tokens: List[Token]) -> List[Chunk]:
        chunks = self.__chunk_tokens(tokens)
        chunk_spans = self.__chunk_span(tokens)
        return [
            Chunk(
                tokens=chunk_tokens,
                label=chunk_tokens[0].label.split("-")[1],
                span=chunk_span,
            )
            for chunk_tokens, chunk_span in zip(chunks, chunk_spans)
        ]

    @staticmethod
    def __chunk_tokens(tokens: List[Token]) -> List[List[Token]]:
        chunks = []
        chunk = []
        for token in tokens:
            if token.label.startswith("B"):
                if chunk:
                    chunks.append(chunk)
                    chunk = []
                chunk = [token]
            elif token.label.startswith("I"):
                chunk.append(token)
            elif chunk:
                chunks.append(chunk)
                chunk = []
        return chunks

    @staticmethod
    def __chunk_span(tokens: List[Token]) -> List[Tuple[int, int]]:
        pos = 0
        spans = []
        chunk_spans = []
        for token in tokens:

            token_len = len(token.text)
            span = (pos, pos + token_len)
            pos += token_len

            if token.label.startswith("B"):
                # I->B
                if len(spans) > 0:
                    chunk_spans.append((spans[0][0], spans[-1][1]))
                    spans = []
                spans.append(span)
            elif token.label.startswith("I"):
                spans.append(span)
            elif len(spans) > 0:
                # B|I -> O
                chunk_spans.append((spans[0][0], spans[-1][1]))
                spans = []

        return chunk_spans

    def assert_spans(self):
        for chunk in self.chunks:
            assert self.text[chunk.span[0] : chunk.span[1]] == "".join(
                [t.text for t in chunk]
            )


class ConllConverter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    @staticmethod
    def get_superspan(
        query_span: Tuple[int, int], superspans: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """return superspan for given query span from set of superspans if any"""
        for superspan in superspans:
            if query_span[0] >= superspan[0] and query_span[1] <= superspan[1]:
                return superspan
        return None

    @classmethod
    def get_token_labels(
        cls,
        token_spans: List[Tuple[int, int]],
        chunk_spans: List[Tuple[int, int]],
        chunk_labels: List[str],
    ) -> List[str]:
        """chunk単位のNE-typeラベルから、token単位のBIOラベルを構成"""

        chunkspan2tagtype = dict(zip(chunk_spans, chunk_labels))

        # token_spansに含まれるchunk(span)を決定し、chunkのtagtypeも同時に記録
        target_token_spans = []
        tagtypes = []
        for token_span in token_spans:
            chunk_span = cls.get_superspan(token_span, chunk_spans)
            if chunk_span is not None and chunk_span in chunkspan2tagtype:
                target_token_spans.append(token_span)
                tagtypes.append(chunkspan2tagtype[chunk_span])
        tokenspan2tagtype = dict(zip(target_token_spans, tagtypes))

        # token に対応する label をchunkのtagtypeを基に構成
        label = "O"
        token_labels = []
        for token_span in token_spans:
            if token_span in tokenspan2tagtype:
                tagtype = tokenspan2tagtype[token_span]
                if label == "O":
                    label = f"B-{tagtype}"
                else:
                    label = f"I-{tagtype}"
            else:
                label = "O"

            token_labels.append(label)
        return token_labels

    def tokenize_and_align_spans(
        self, text: str, chunk_spans: List[Tuple[int, int]], chunk_labels: List[str]
    ) -> List[Tuple[str, str]]:

        # text -> tokens
        tokens = self.tokenize(text)

        # 各tokenがtextのどこにあるか(token_spans)を計算
        token_spans = tokenizations.get_original_spans(tokens, text)
        # assertion
        spannedtokens = [text[span[0] : span[1]] for span in token_spans]
        assert spannedtokens == tokens

        # 各tokenに対応するchunk NE-typeを同定(token-span vs chunk-span の包含関係計算)
        token_labels = self.get_token_labels(token_spans, chunk_spans, chunk_labels)

        # CoNLL2003-likeなtoken行単位の列データを返す
        return [(token, label) for token, label in zip(tokens, token_labels)]


class MecabTokenizer:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")

    def tokenize(self, text: str) -> List[str]:
        return self.tagger.parse(text).split()


class SpacyTokenizer:
    def __init__(self, nlp=None):
        if nlp is None:
            self.nlp = spacy.load("ja_ginza")
        else:
            self.nlp = nlp

    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [token.text for token in doc]


def make_sentences_from_conll(conll_filepath: str) -> List[Sentence]:
    """conll (token-label columns) -> sentences (token-span-label containers)"""
    with open(conll_filepath) as fp:
        sentences = fp.read().split("\n\n")
        sentences = [
            Sentence(
                [
                    Token(*token.split("\t"))
                    for token in sentence.split("\n")
                    if len(token.split("\t")) == 2
                ]
            )
            for sentence in sentences
        ]
        return [s for s in sentences if s.text]


def retokenize_sentences(sentences: List[Sentence], tokenizer=None) -> str:
    """sentences -> re-tokenization -> conll"""
    if tokenizer is None:
        tokenizer = MecabTokenizer()
    conll = ConllConverter(tokenizer)
    sentence_columns = []
    for sentence in sentences:
        text, chunks = sentence.text, sentence.chunks
        chunk_spans = [chunk.span for chunk in chunks]
        chunk_tagtypes = [chunk.label for chunk in chunks]
        # print(text, chunk_spans, chunk_tagtypes)

        sentence_column: List[Tuple[str, str]] = conll.tokenize_and_align_spans(
            text, chunk_spans, chunk_tagtypes
        )
        sentence_column_str = "\n".join(
            [f"{token}\t{label}" for token, label in sentence_column]
        )
        sentence_columns.append(sentence_column_str + "\n")
    return "\n".join(sentence_columns)


def evaluate_ginza(
    filepath: str, as_ontonote: bool = False, retokenize: bool = False
) -> Tuple[List[str], List[str]]:
    """filepath: conll2003-like token and bio columns dataset
    If different tokenization is used, pass retokenize=True.
    If you want to evaluate the result as OntoNote5 scheme, pass as_ontonote=True.
    Especially, UD Japanese GSD dataset uses OntoNote5, while GiNZA outputs ENE scheme.
    """
    nlp = spacy.load("ja_ginza")

    def parse(sentence: str) -> dict:
        doc = nlp(sentence)
        tokens = [token.text for token in doc]
        labels = [[ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]
        tags = biluo_tags_from_offsets(doc, labels)
        return {
            "text": sentence,
            "tokens": tokens,
            "labels": labels,
            "tags": tags,
        }

    def map2ontonote(tag: str) -> str:
        if tag == "O":
            return tag
        elif len(tag.split("-")) == 2:
            prefix, tagtype = tag.split("-")
            if tagtype in ENE_ONTONOTES_MAPPING:
                tagtype = ENE_ONTONOTES_MAPPING[tagtype]
            return f"{prefix}-{tagtype}"
        else:
            print("invalid tag")
            return "O"

    if retokenize:
        tokenizer = SpacyTokenizer(nlp)
        sentences: List[Sentence] = make_sentences_from_conll(filepath)
        new_conll = retokenize_sentences(sentences, tokenizer)
        filepath = filepath + ".retokenize"
        with open(filepath, "w") as fp:
            fp.write(new_conll)

    sentences = make_sentences_from_conll(filepath)
    results = [parse(s.text) for s in sentences]
    pred = [r["tags"] for r in results]
    gold = [iob_to_biluo([t.label for t in s]) for s in sentences]
    if as_ontonote:
        pred = [list(map(map2ontonote, tags)) for tags in pred]
        gold = [list(map(map2ontonote, tags)) for tags in gold]
    print(classification_report(gold, pred, mode="strict", scheme=BILOU))
    return gold, pred


def calc_metric(
    gold: List[str], pred: List[str], sort_by_metric: str = "netype"
) -> pd.DataFrame:
    metrics_str = classification_report(gold, pred, mode="strict", scheme=BILOU)
    metrics_str = "netype" + metrics_str
    metrics_str = re.sub("[ ]+", " ", metrics_str)
    tpls = [
        l.strip().split(" ")
        for l in metrics_str.split("\n")
        if len(l.strip().split(" ")) == 5
    ]
    header, data = tpls[0], tpls[1:]
    df = pd.DataFrame(data, columns=header)
    return df.sort_values(by=[sort_by_metric], ascending=True)


def make_jsonl(conll_filepath="data/test.bio"):
    sentences_test = make_sentences_from_conll(conll_filepath)
    with open(conll_filepath + ".jsonl", "w") as fp:
        for sentence in sentences_test[1:]:
            text = sentence.text
            jl = [
                text,
                {
                    "entities": [
                        [c.span[0], c.span[1], c.label] for c in sentence.chunks
                    ]
                },
            ]
            fp.write(json.dumps(jl, ensure_ascii=False))
            fp.write("\n")


def download_conll_data(filepath: str = "test.bio"):
    """conllフォーマットデータのダウンロード"""
    filename = Path(filepath).name
    url = f"https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/{filename}"
    response = requests.get(url)
    if response.ok:
        with open(filepath, "w") as fp:
            fp.write(response.content.decode("utf8"))
        return filepath


if __name__ == "__main__":
    filepath = "./test.bio"
    if download_conll_data(filepath):
        # CoNLL2003 -> List[Sentence]
        gold, pred = evaluate_ginza(filepath, as_ontonote=True, retokenize=True)
        df = calc_metric(gold, pred)
        print(df)
