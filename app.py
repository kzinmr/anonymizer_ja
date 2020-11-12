import json
import jsonlines
from address_matcher import AddressMatcher
from company_matcher import CompanyMatcher
from person_matcher import PersonMatcher

am = AddressMatcher()
cm = CompanyMatcher()
pm = PersonMatcher()


def jsonl_ja(js):
    return json.dumps(js, ensure_ascii=False)


def convert_spans(jsons):
    jsons_new = []
    for j in jsons:
        text = j["text"]
        labels_all = []
        for ent in j["persons"]:
            if len(ent) > 2:
                pattern = re.compile("(?:%s){i<=3:\s}" % (re.escape(ent)))
                spans = [m.span() for m in pattern.finditer(text)]
                labels = [[s, e, "PERSON"] for s, e in spans]
                labels_all.extend(labels)
        for ent in j["companies"]:
            if len(ent) > 2:
                pattern = re.compile("(?:%s){i<=3:\s}" % (re.escape(ent)))
                spans = [m.span() for m in pattern.finditer(text)]
                labels = [[s, e, "COMPANY"] for s, e in spans]
                labels_all.extend(labels)
        for ent in j["addresses"]:
            if len(ent) > 2:
                pattern = re.compile("(?:%s){i<=3:\s}" % (re.escape(ent)))
                spans = [m.span() for m in pattern.finditer(text)]
                labels = [[s, e, "ADDRESS"] for s, e in spans]
                labels_all.extend(labels)
        meta = {"doc_id": j["doc_id"], "annotations": j["gold"]}
        jsons_new.append({"text": text, "labels": labels_all, "meta": meta})
    return jsons_new


if __name__ == "__main__":
    with jsonlines.open("/app/data/text.jsonl") as fp:
        jsons = [jsonstr for jsonstr in fp.iter()]
    outputs = []
    for jd in jsons:
        doc_id = jd["document_id"]
        gold = jd["gold"]
        input_text = jd["text"]
        addresses = am.parse(input_text)
        companies = cm.parse(input_text)
        persons = pm.parse(input_text)
        output = {
            "doc_id": doc_id,
            "gold": gold,
            "text": input_text,
            "addresses": addresses,
            "companies": companies,
            "persons": persons,
        }
        outputs.append(output)
    with jsonlines.open("/app/data/predict.jsonl", "w", dumps=jsonl_ja) as fp:
        fp.write_all(outputs)
    span_jsons = convert_spans(outputs)
    with jsonlines.open("/app/data/predict_spans.jsonl", "w", dumps=jsonl_ja) as fp:
        fp.write_all(span_jsons)
    print(output[-1])
    print(span_jsons[-1])
