import json
from address_matcher import AddressMatcher
from company_matcher import CompanyMatcher
from person_matcher import PersonMatcher

am = AddressMatcher()
cm = CompanyMatcher()
pm = PersonMatcher()

if __name__ == "__main__":
    with open("/app/data/text.txt") as fp:
        input_text = fp.read()
    addresses = am.parse(input_text)
    companies = cm.parse(input_text)
    persons = pm.parse(input_text)
    output = {"addresses": addresses, "companies": companies, "persons": persons}
    with open("/app/data/predict.json", mode="w") as fp:
        fp.write(json.dumps(output, ensure_ascii=False))
    print(output)
