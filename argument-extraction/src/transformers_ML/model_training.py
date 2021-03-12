from transformers_ML.config import *
from transformers_ML.data_loader import DataLoadHandler
from transformers_ML.utils import *



# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2,
                                                        output_attentions = False, output_hidden_states = False)

model = model.to(device)

def train_model(epochs=3):
    data_loader = DataLoadHandler(data_path='combined_sentence.json')
    train_dataloader, test_dataloader = data_loader.GetDataLoaders()
    class_weights = data_loader.GetClassWeights()

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    for epoch_i in range(epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        train_loss = TrainBertSeqCl(model, train_dataloader, optimizer, scheduler, class_weights)
        loss_values.append(train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print('validation with training data: ')
        EvaluateBertSeqCl(model, train_dataloader)

        # ========================================
        #             Model Saving
        # ========================================
        torch.save(model.state_dict(), f'./models/essay-a-n_trained_model_epoch_{epoch_i}_weighted.pt')

