import torch
import time
from data_utils import (json_dump, generate_batches)


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}


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
        torch.save(model.state_dict(), train_state['model_filename'])
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
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def dump_train_state_to_json(train_state, path):
    obj = dict(epochs=train_state['epoch_index']+1,
               train_loss=train_state['train_loss'],
               train_acc=train_state['train_acc'],
               val_loss=train_state['val_loss'],
               val_acc=train_state['val_acc'],
               test_loss=train_state['test_loss'],
               test_acc=train_state['test_acc'])
    json_dump(obj, path)


def compute_accuracy_binary(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def run_experiment(args, classifier, loss_func, optimizer, dataset, logger):

    train_state = make_train_state(args)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on

            dataset.set_split('train')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # 5 step training routine

                # --------------------------------------
                # 1) zero the gradients
                optimizer.zero_grad()

                # 2) compute the output
                y_pred = classifier(batch_dict['x_data']).squeeze()

                # 3) compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # 4) use loss to produce gradients
                loss.backward()

                # 5) use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------

                # compute the accuracy
                acc_t = compute_accuracy_binary(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                logger.info(f'Epoch {epoch_index+1}/{args.num_epochs} | '
                            f'batch index: {batch_index+1}/{dataset.get_num_batches(args.batch_size)} | '
                            f'train_loss = {running_loss}; train_acc = {running_acc}\n')

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)
            logger.info(f'Epoch {epoch_index+1}/{args.num_epochs} | '
                        f'train_loss = {running_loss}; train_acc = {running_acc}\n')

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss=0.
            running_acc=0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred = classifier(batch_dict['x_data']).squeeze()

                # compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy_binary(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
            logger.info(f'Epoch {epoch_index+1}/{args.num_epochs} | '\
                        f'val_loss = {running_loss}; val_acc = {running_acc}\n')

            train_state = update_train_state(args=args, model=classifier,
                                             train_state=train_state)


            if train_state['stop_early']:
                break

    except KeyboardInterrupt:
        print("Exiting loop...")


    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(batch_dict['x_data']).squeeze()

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t=compute_accuracy_binary(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss']=running_loss
    train_state['test_acc']=running_acc
    logger.info(f'test_loss = {running_loss}; test_acc = {running_acc}\n')

    dump_train_state_to_json(train_state, args.train_state_file)


def run(args, model, loss_func, optimizer, iterator):

    train_state = make_train_state(args)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index


            running_loss = 0.0
            running_acc = 0.0
            model.train()

            for batch_index, batch in enumerate(iterator['train']):
                # 5 step training routine

                # --------------------------------------
                # 1) zero the gradients
                optimizer.zero_grad()

                # 2) compute the output
                x_in, lengths = batch.text
                y_pred = model(x_in, lengths).squeeze()

                # 3) compute the loss
                loss = loss_func(y_pred, batch.label)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # 4) use loss to produce gradients
                loss.backward()

                # 5) use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------

                # compute the accuracy
                acc_t = compute_accuracy_binary(y_pred, batch.label)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                print(f'Epoch {epoch_index+1}/{args.num_epochs} | '
                            f'batch index: {batch_index+1} | '
                            f'train_loss = {running_loss}; train_acc = {running_acc}\n')

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)
            print(f'Epoch {epoch_index+1}/{args.num_epochs} | '
                        f'train_loss = {running_loss}; train_acc = {running_acc}\n')

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            running_loss=0.
            running_acc=0.
            model.eval()

            for batch_index, batch_dict in enumerate(iterator['valid']):

                # compute the output
                x_in, lengths = batch.text
                y_pred = model(x_in, lengths).squeeze()

                # compute the loss
                loss = loss_func(y_pred, batch.label)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy_binary(y_pred, batch.label)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
            print(f'Epoch {epoch_index+1}/{args.num_epochs} | '\
                        f'val_loss = {running_loss}; val_acc = {running_acc}\n')

            train_state = update_train_state(args=args, model=model,
                                             train_state=train_state)

            if train_state['stop_early']:
                break

    except KeyboardInterrupt:
        print("Exiting loop...")


    # dataset.set_split('test')
    # batch_generator = generate_batches(dataset,
    #                                    batch_size=args.batch_size,
    #                                    device=args.device)
    # running_loss = 0.
    # running_acc = 0.
    # classifier.eval()

    # for batch_index, batch_dict in enumerate(batch_generator):
    #     # compute the output
    #     y_pred = classifier(batch_dict['x_data']).squeeze()

    #     # compute the loss
    #     loss = loss_func(y_pred, batch_dict['y_target'].float())
    #     loss_t = loss.item()
    #     running_loss += (loss_t - running_loss) / (batch_index + 1)

    #     # compute the accuracy
    #     acc_t=compute_accuracy_binary(y_pred, batch_dict['y_target'])
    #     running_acc += (acc_t - running_acc) / (batch_index + 1)

    # train_state['test_loss']=running_loss
    # train_state['test_acc']=running_acc
    # logger.info(f'test_loss = {running_loss}; test_acc = {running_acc}\n')

    # dump_train_state_to_json(train_state, args.train_state_file)


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        print('NEW BATCH')
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def run_exp(model, train_iterator, valid_iterator, optimizer, criterion):
    N_EPOCHS = 20
    best_valid_loss = float('inf')
    print('START')
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')