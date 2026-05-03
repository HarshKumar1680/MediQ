import requests

def test_pubmed():
    term = "diabetes"
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&retmax=2&retmode=json"
    res = requests.get(search_url).json()
    id_list = res["esearchresult"]["idlist"]
    print("IDs:", id_list)
    
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(id_list)}&retmode=xml"
    fetch_res = requests.get(fetch_url)
    print("Fetch output length:", len(fetch_res.text))
    print(fetch_res.text[:500])

if __name__ == "__main__":
    test_pubmed()
