import torch
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


def run_experiment(args, classifier, loss_func, optimizer, scheduler, dataset, logger):

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

            scheduler.step(train_state['val_loss'][-1])

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