from whoosh.analysis import StemmingAnalyzer
from rank_bm25 import BM25Okapi
from utils.documents_retrieval import GetTopics
import json


BASE_DATA_DIR = './retrieved_documents/'
OUTPUT_FILE_NAME = './run.txt'
TAG = 'BERT_and_bm25_ranking'

def GetTokenizedDocuments(analyzer, documents):
    tokenized_docs = []
    for doc in documents:
        with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
            args_list = f.readlines()
        arg_text = ''.join(args_list)
        tokenized_docs.append([token.text for token in analyzer(arg_text)])
    return tokenized_docs

def ranking(bm25_instance, analyzer, query, documents, size=15):
    tokenized_query = [token.text for token in analyzer(query)]
    doc_scores = bm25_instance.get_scores(tokenized_query)
    result = [{'trec_id':doc['trec_id'], 'score':doc_scores[index]} for index, doc in enumerate(documents)]
    result = sorted(result, key=lambda x: x['score'], reverse=True)
    return result[:size]

def main():
    output_lines = []
    analyzer = StemmingAnalyzer()
    topics = GetTopics()
    documents = json.load(open('retrieval_results_with_args.json', encoding='utf-8'))

    for topic_docs in documents:
        tokenized_docs = GetTokenizedDocuments(analyzer, topic_docs['documents'])
        print(len(tokenized_docs))
        bm25 = BM25Okapi(tokenized_docs)
        query = topic_docs['topic']['title']
        ranked_docs = ranking(bm25, analyzer, query, topic_docs['documents'])
        for rank, item in enumerate(ranked_docs, start=1):
            # output format
            #  qid Q0 doc rank score tag
            output_line = '{} Q0 {} {} {} {}\n'.format(topic_docs['topic']['id'], item['trec_id'], rank, item['score'], TAG)
            output_lines.append(output_line)

    # Save the run output file
    with open(OUTPUT_FILE_NAME, 'w') as f:
        for line in output_lines:
            f.write(line)

if __name__ == '__main__':
    main()