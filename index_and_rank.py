from utils.config import PROJECT_ROOT_DIR

from whoosh.analysis import StemmingAnalyzer
from rank_bm25 import BM25Okapi
from utils.documents_retrieval import GetTopics

# for document embedding calculation 
from sentence_transformers import SentenceTransformer, util

import json


DOCUMENTS_PER_TOPIC = 20
BASE_DATA_DIR = PROJECT_ROOT_DIR + '/retrieved_documents/'
OUTPUT_FILE_NAME = PROJECT_ROOT_DIR + '/run.txt'
TAG = 'BERT_and_bm25_ranking'

analyzer = StemmingAnalyzer()
embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def GetTokenizedDocuments(analyzer, documents):
    tokenized_docs = []
    for doc in documents:
        try:
            with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
                args_list = f.readlines()
        except:
            args_list = []
        arg_text = ''.join(args_list)
        tokenized_docs.append([token.text for token in analyzer(arg_text)])
    return tokenized_docs

def arg_tfidf_ranking(query, documents):
    tokenized_query = [token.text for token in analyzer(query)]
    tokenized_docs = GetTokenizedDocuments(analyzer, documents)
    bm25 = BM25Okapi(tokenized_docs)
    doc_scores = bm25.get_scores(tokenized_query)
    for index, doc in enumerate(documents):
        doc['tfidf_score'] = doc_scores[index]

def arg_support_ranking(query, documents):
    for doc in documents:
        try:
            with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
                args_list = f.readlines()
        except:
            args_list = []

        with open(doc['content_file_path'], 'r', encoding='utf-8') as f:
            all_sents_list = f.readlines()

        doc['arg_support_score_sum'] = len(args_list)
        doc['arg_support_score_norm'] = len(args_list) / len(all_sents_list)
        print('absolut arg support {} {}: {}'.format(query, doc['trec_id'], doc['arg_support_score_norm']), end='\r')

def arg_similarity_ranking(query, documents):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    for doc in documents:
        try:
            with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
                args_list = f.readlines()

            arg_embeddings = embedding_model.encode(args_list, convert_to_tensor=True)
            sentences_similarities = util.pytorch_cos_sim(query_embedding, arg_embeddings).flatten().tolist()
            doc['similarity_score_sum'] = sum(sentences_similarities)
            doc['similarity_score_avg'] = sum(sentences_similarities) / len(sentences_similarities)
        except:
            args_list = []
            doc['similarity_score_avg'] = 0.0
            doc['similarity_score_sum'] = 0.0

        print('similarity {} {}: {}'.format(query, doc['trec_id'], doc['similarity_score_avg']), end='\r')

def normalize_and_compute(documents):
    max_tfidf_score     = max([ doc['tfidf_score'] for doc in documents])
    max_chatnoir_score  = max([ doc['score'] for doc in documents])
    max_page_rank_score = max([ doc['page_rank'] for doc in documents])
    for doc in documents:
        doc['tfidf_score_norm']     = float(doc['tfidf_score']) / max_tfidf_score
        doc['chatnoir_score_norm']  = float(doc['score']) / max_chatnoir_score
        doc['page_rank_norm']       = float(doc['page_rank']) / max_page_rank_score

    chatnoir_w      = 15
    tfidf_w         = 25
    arg_support_w   = 25
    similarity_w    = 15
    page_rank_w     = 20

    for doc in documents:
        doc['final_score'] = chatnoir_w     * doc['chatnoir_score_norm']    \
                            + tfidf_w       * doc['chatnoir_score_norm']    \
                            + arg_support_w * doc['arg_support_score_norm'] \
                            + similarity_w  * doc['similarity_score_avg']   \
                            + page_rank_w   * doc['page_rank_norm']



def rank_documents(query, documents):
    print('Ranking {} documents for topic: {}'.format(len(documents), query))

    print('Perform tfidf ranking...', end='\r')
    arg_tfidf_ranking(query, documents)

    print('Perform argument support ranking...', end='\r')
    arg_support_ranking(query, documents)
    
    print('Perform documents similarity ranking...', end='\r')
    arg_similarity_ranking(query, documents)

    print('Normalization and overall score calculation ...', end='\r')
    normalize_and_compute(documents)

def main():
    topics = GetTopics()
    documents = json.load(open(PROJECT_ROOT_DIR + 'retrieval_results_with_args.json', encoding='utf-8'))

    output_lines = []
    for topic_docs in documents:
        rank_documents(topic_docs['topic']['title'], topic_docs['documents'])
        ranked_docs = sorted(topic_docs['documents'], key=lambda x: x['final_score'], reverse=True)
        for rank, doc in enumerate(ranked_docs[:DOCUMENTS_PER_TOPIC], start=1):
            # output format
            #  qid Q0 doc rank score tag
            output_line = '{} Q0 {} {} {} {}\n'.format(topic_docs['topic']['id'], doc['trec_id'], rank, doc['tfidf_score'], TAG)
            output_lines.append(output_line)

    # Save the run output file
    with open(OUTPUT_FILE_NAME, 'w') as f:
        for line in output_lines:
            f.write(line)


if __name__ == '__main__':
    main()