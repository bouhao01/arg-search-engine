from utils.config import *
import trafilatura
import requests
import xml.etree.ElementTree as ET
from boilerpy3 import extractors




RETIEVED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + '/retrieved_documents/row-data/'
EXTRACTED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + '/retrieved_documents/extracted-data/'
TOPICS_FILE = PROJECT_ROOT_DIR + '/topics_2020.xml'

baseUrl = 'https://www.chatnoir.eu'
api_key = 'e47fe59e-2d2f-475e-a424-afcdb94ba17b'
extractor = extractors.ArticleExtractor()


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
    url = baseUrl + '/cache?uuid={}&index={}&raw'.format(uuid, index)
    g = requests.get(url)
    source_file = g.text # trafilatura.fetch_url(url)

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

def DownloadRelatedDocuments():
    retrieval_results = []
    topics = GetTopics()

    for topic in topics:
        result_obj = {
                    'topic' : topic
                    ,'documents' : []
                    }

        time.sleep(0.2)
        if not os.path.exists(RETIEVED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id'])):
            os.mkdir(RETIEVED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id']))
        if not os.path.exists(EXTRACTED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id'])):
            os.mkdir(EXTRACTED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id']))

        print('\n', '#'*30)
        print('# Retrieving documents for topic {}: {}'.format(topic['id'], topic['title']))
        print('#'*30)
        documentsFromChatNoir = GetDocumentsFromChatNoir(topic['title'], 20)

        for doc in documentsFromChatNoir:
            time.sleep(0.2)
            source_file, main_content = GetDocContent(topic['id'], doc['uuid'])
            source_file_path = RETIEVED_DOCUMENTS_DIR + 'topic-{}/{}.html'.format(topic['id'], doc['trec_id'])
            content_file_path = EXTRACTED_DOCUMENTS_DIR + 'topic-{}/{}.txt'.format(topic['id'], doc['trec_id'])

            # Save local copy of the source file and the content file
            with open(source_file_path, 'w', encoding='utf-8') as f:
                f.write(source_file)
            with open(content_file_path, 'w', encoding='utf-8') as f:
                f.write(main_content)

            doc['source_file_path'] = os.path.abspath(source_file_path)
            doc['content_file_path'] = os.path.abspath(content_file_path)
            result_obj['documents'].append(doc)

        retrieval_results.append(result_obj)

        print('{} documents has been collected.'.format(len(documentsFromChatNoir)))
    with open(PROJECT_ROOT_DIR + '/retrieval_results.json', 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f)

def GetRetrievalResults():
    with open(PROJECT_ROOT_DIR + '/retrieval_results.json', 'r', encoding='utf-8') as f:
        retrieval_results = json.load(f)
    return retrieval_results
