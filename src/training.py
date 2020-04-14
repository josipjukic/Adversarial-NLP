import torch
import time
from data_utils import json_dump


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': float('inf'),
            'learning_rate': args.learning_rate,
            'json_path': args.train_state_file,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'model_path': args.model_save_file}


def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_path'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_path'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def dump_train_state_to_json(train_state, path):
    obj = dict(epochs=train_state['epoch_index'],
               train_loss=train_state['train_loss'],
               train_acc=train_state['train_acc'],
               val_loss=train_state['val_loss'],
               val_acc=train_state['val_acc'],
               test_loss=train_state['test_loss'],
               test_acc=train_state['test_acc'])
    json_dump(obj, path)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(y_pred, y_gold):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(y_pred))
    correct = (rounded_preds == y_gold).float() # convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion, train_state):
    
    print('Entering training mode...')

    running_loss = 0.
    running_acc = 0.
    num_batches = len(iterator)
    
    model.train()
    
    for batch_index, batch in enumerate(iterator, 1):
        # 5 step training routine

        # --------------------------------------
        # 1) zero the gradients
        optimizer.zero_grad()
        
        # 2) compute the output
        x_int, lengths = batch.text
        y_pred = model(x_in, lengths).squeeze()

        # 3) compute the loss
        loss = criterion(y_pred, batch.label)
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / batch_index
        
        # 4) use loss to produce gradients
        loss.backward()

        # 5) use optimizer to take gradient step
        optimizer.step()
        # -----------------------------------------

        # compute the accuracy
        acc_t = binary_accuracy(y_pred, batch.label)
        running_acc += (acc_t - running_acc) / batch_index
        print(f'batch index: {batch_index}/{num_batches} | '
              f'train_loss = {running_loss}; train_acc = {running_acc}')
                
    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    return running_loss, running_acc


def evaluate(model, iterator, criterion, train_state, mode='valid'):
    
    print(f'Entering {mode} mode...')

    running_loss = 0.
    running_acc = 0.
    num_batches = len(iterator)
    
    model.eval()
    
    with torch.no_grad():
    
        for batch_index, batch in enumerate(iterator, batch):
            x_int, lengths = batch.text
            y_pred = model(x_in, lengths).squeeze()

            loss = criterion(y_pred, batch.label)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / batch_index
            
            acc_t = binary_accuracy(y_pred, batch.label)
            running_acc += (acc_t - running_acc) / batch_index
            print(f'batch index: {batch_index}/{num_batches} | '
                  f'{mode}_loss = {running_loss}; {mode}_acc = {running_acc}')
    
    train_state[f'{mode}_loss'].append(running_loss)
    train_state[f'{mode}_acc'].append(running_acc)
        
    return running_loss, running_acc


def run_experiment(args, model, iterator, optimizer, criterion):
    
    train_state = make_train_state(args)

    for epoch in range(args.num_epochs):

        start_time = time.time()
        
        train_loss, train_acc = train(model, iterator['train'], optimizer, criterion, train_state)
        valid_loss, valid_acc = evaluate(model, iterator['valid'], criterion, train_state)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_state = update_train_state(args=args, model=model,
                                         train_state=train_state)

        if train_state['stop_early']:
            break
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')



    test_loss, test_acc = evaluate(model, iterator['test'], criterion, train_state, mode='test')
    print(f'test_loss = {test_loss}; test_acc = {test_acc}')
    dump_train_state_to_json(train_state, args.train_state_file)