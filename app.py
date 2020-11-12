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
    print(output)
