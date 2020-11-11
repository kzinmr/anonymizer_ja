import MeCab

tagger = MeCab.Tagger(
    "-r /dev/null -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"
)

if __name__ == "__main__":
    with open("/app/data/text.txt") as fp:
        inputs = filter(None, fp.read().splitlines())
    outputs = [tagger.parse(text) for text in inputs]
    with open("/app/data/predict.txt", mode="w") as fp:
        fp.write("\n".join(outputs))
    print(outputs)