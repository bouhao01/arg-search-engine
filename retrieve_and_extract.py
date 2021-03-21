from utils.config import *
from utils.model_handler import LoadBertModel, Predict, TextToSentences
from utils.documents_retrieval import DownloadRelatedDocuments, GetRetrievalResults




def init():
    global model

    bert_model_path = PROJECT_ROOT_DIR + '/ArgumentExtraction/models/distilroberta-base_essays_a_n_epoch_0_weighted.pt'
    model = LoadBertModel(bert_model_path)

def main(args):
    # Retrieve docs from chatnoir
    #####################################
    # DownloadRelatedDocuments() # Run only once
    retrieval_info = ExtractDocsMainCentent()

    for ret_obj in retrieval_info:
        for doc in ret_obj['documents']:
            with open(doc['content_file_path'], 'r', encoding='utf-8') as f:
                sentences = f.readlines()

            if not sentences:
                print('no sentences in {}'.format(doc_content['trec_id']))
                continue

            # Extract argument from documents
            #####################################
            predicted = Predict(model, sentences)
            args = [sentences[idx] for idx, y in enumerate(predicted) if 0 != int(y)]

            # Save extracted arguments
            # and update doc info
            ####################################
            args_file_path = EXTRACTED_DOCUMENTS_DIR + 'topic-{}/{}.args'.format(ret_obj['topic']['id'], doc['trec_id'])
            with open(args_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(args))
            doc['args_file_path'] = os.path.abspath(args_file_path)

    # Save args info file
    ####################################
    with open(PROJECT_ROOT_DIR + '/retrieval_results_with_args.json', 'w', encoding='utf-8') as f:
        json.dump(retrieval_info, f)

if __name__ == '__main__':
    init()
    main(sys.argv[1:])