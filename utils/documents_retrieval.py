from utils.config import *
import trafilatura
import requests
import xml.etree.ElementTree as ET
from boilerpy3 import extractors
import concurrent.futures


from rake_nltk import Rake

rake = Rake()

RETIEVED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + '/retrieved_documents/row-data/'
EXTRACTED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + '/retrieved_documents/extracted-data/'
TOPICS_FILE = PROJECT_ROOT_DIR + '/topics.xml'

baseUrl = 'https://www.chatnoir.eu'
api_key = 'e47fe59e-2d2f-475e-a424-afcdb94ba17b'
extractor = extractors.ArticleExtractor()


def GetTrimmedKeyword(keyword):
    stop_words = ['better', 'difference', 'best']
    trimmed_word = ' '.join([token.lemma_ for token in nlp(keyword) if token.text not in stop_words])
    return trimmed_word

def GetKeywords(text):
    keywords = []
    rake.extract_keywords_from_text(text)
    keywords_rake = rake.get_ranked_phrases()
    for word in keywords_rake:
        trimmed_word = GetTrimmedKeyword(word)
        if trimmed_word: keywords.append(trimmed_word)
    return keywords

def GetDocumentIntersection(set_1, set_2):
    intersection_set = []
    for doc_1 in set_1:
        for doc_2 in set_2:
            if doc_1['uuid'] == doc_2['uuid']:
                doc_1['score_2'] = doc_2['score']
                doc_1['page_rank_2'] = doc_2['page_rank']
                doc_1['spam_rank_2'] = doc_2['spam_rank']
                intersection_set.append(doc_1)
                break
    print('intersection set: ', len(intersection_set))
    return intersection_set

def CreatTopicDirs(topic):
    if not os.path.exists(RETIEVED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id'])):
            os.mkdir(RETIEVED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id']))
    if not os.path.exists(EXTRACTED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id'])):
        os.mkdir(EXTRACTED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id']))

def GetTopics(fileLocarion=TOPICS_FILE):
    topics = []
    tree = ET.parse(fileLocarion)
    root = tree.getroot()
    for element in root:
        topic = {
                'id': element[0].text.strip()
                ,'title': element[1].text.strip()
                }
        topics.append(topic)

    return topics

def GetDocumentsFromChatNoir(query, size=10):
    data = {
            'apikey': api_key
            ,'query': query
            ,'index': ['cw12']
            ,'pretty': True
            ,'size':size
            }
    p = requests.post(baseUrl + '/api/v1/_search', data=data)
    results = json.loads(p.text)

    return results['results']

def GetDocContent(topic_id, uuid, index='cw12'):
    url = baseUrl + '/cache?uuid={}&index={}&raw&plain'.format(uuid, index)
    # g = requests.get(url) 5e733d53-43e8-58f0-abfe-fa7fc2538733
    source_file = trafilatura.fetch_url(url) # g.text

    if not source_file:
        print('Cannot retrieve document {}'.format(uuid))
        time.sleep(0.5)
        return ' ', ' '
        # return GetDocContent(topic_id, uuid, index)

    print('Document has been retrieved succesfully {}'.format(uuid))

    # Extract content using boilerpy3 and trafilatura, then combine results
    data_1 = trafilatura.extract(source_file)
    if data_1:
        data_1 = TAG_RE.sub('', data_1)
        doc_1 = nlp(data_1)
        sents_1 = [sent.text.strip().lower().replace('\n', ' ') for sent in doc_1.sents if len(sent.text) > 20]
    else:
        sents_1 = []

    data_2 = extractor.get_content(source_file)
    if data_2:
        data_2 = TAG_RE.sub('', data_2)
        doc_2 = nlp(data_2)
        sents_2 = [sent.text.strip().lower().replace('\n', ' ') for sent in doc_2.sents if len(sent.text) > 20]
    else:
        sents_2 = []

    final_data = list(set(sents_1) | set(sents_2))
    main_content = '\n'.join(final_data)

    return source_file, main_content

def GetDocumentsTopic(topic, size=60):
    result_obj = {'topic' : topic, 'documents' : []}
    CreatTopicDirs(topic)

    keywords = GetKeywords(topic['title'])
    print('\n{}\n# Retrieving documents for topic {}: {}\n{}\n'.format('#'*30, topic['id'], topic['title'], '#'*30))

    query_1 = topic['title']
    query_2 = ' AND '.join(keywords)
    documents_set_1 = GetDocumentsFromChatNoir(query_1, size=size)
    documents_set_2 = GetDocumentsFromChatNoir(query_2, size=size)

    documentsFromChatNoir = documents_set_1 #GetDocumentIntersection(documents_set_1, documents_set_2) # documents_set_1

    for doc in documentsFromChatNoir:

        source_file_path = RETIEVED_DOCUMENTS_DIR + 'topic-{}/{}.html'.format(topic['id'], doc['trec_id'])
        content_file_path = EXTRACTED_DOCUMENTS_DIR + 'topic-{}/{}.txt'.format(topic['id'], doc['trec_id'])

        doc['source_file_path'] = os.path.abspath(source_file_path)
        doc['content_file_path'] = os.path.abspath(content_file_path)
        result_obj['documents'].append(doc)

        # No need to retrieve the file if it is already retrieved
        if os.path.exists(source_file_path) and os.path.exists(content_file_path):
            print('Document {} has been already retrieved'.format(doc['uuid']))
            continue

        source_file, main_content = GetDocContent(topic['id'], doc['uuid'])
        with open(source_file_path, 'w', encoding='utf-8') as f:
            f.write(source_file)
        with open(content_file_path, 'w', encoding='utf-8') as f:
            f.write(main_content)

    return result_obj

def DownloadRelatedDocuments():
    retrieval_results = []
    topics = GetTopics()

    for topic in topics:
        '''
        result_obj = {
                    'topic' : topic
                    ,'documents' : []
                    }

        CreatTopicDirs(topic)

        print('\n{}\n# Retrieving documents for topic {}: {}\n{}\n'.format('#'*30, topic['id'], topic['title']), '#'*30)
        documentsFromChatNoir = GetDocumentsFromChatNoir(topic['title'], size=30)

        for doc in documentsFromChatNoir:

            source_file_path = RETIEVED_DOCUMENTS_DIR + 'topic-{}/{}.html'.format(topic['id'], doc['trec_id'])
            content_file_path = EXTRACTED_DOCUMENTS_DIR + 'topic-{}/{}.txt'.format(topic['id'], doc['trec_id'])

            doc['source_file_path'] = os.path.abspath(source_file_path)
            doc['content_file_path'] = os.path.abspath(content_file_path)
            result_obj['documents'].append(doc)

            # No need to retrieve the file if it is already retrieved
            if os.path.exists(source_file_path) and os.path.exists(content_file_path):
                print('Document {} has been already retrieved'.format(doc['uuid']))
                continue

            time.sleep(0.5)
            source_file, main_content = GetDocContent(topic['id'], doc['uuid'])
            # Save local copy of the source file and the content file
            with open(source_file_path, 'w', encoding='utf-8') as f:
                f.write(source_file)
            with open(content_file_path, 'w', encoding='utf-8') as f:
                f.write(main_content)

        '''
        topic_docs = GetDocumentsTopic(topic)
        retrieval_results.append(topic_docs)

        print('{} documents has been collected.'.format(len(topic_docs['documents'])))
    with open(PROJECT_ROOT_DIR + '/retrieval_results.json', 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f)

def GetRetrievalResults():
    with open(PROJECT_ROOT_DIR + '/retrieval_results.json', 'r', encoding='utf-8') as f:
        retrieval_results = json.load(f)
    return retrieval_results
