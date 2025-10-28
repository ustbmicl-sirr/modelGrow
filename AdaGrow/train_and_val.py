import torch
import os
from copy import deepcopy
import time
from functools import wraps

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"■ training for this epoch used : {elapsed_time:.4f} seconds.")
        return result
    return wrapper

best_acc = 0

@timing
def train(args, logger, trainloader, device, epoch, net, optimizer, criterion, lr_schedule_values, wd_schedule_values, num_training_steps_per_epoch, param_ema):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if lr_schedule_values is not None or wd_schedule_values is not None:
            it = epoch * num_training_steps_per_epoch + batch_idx
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        total_cost = loss
        total_cost.backward()
        optimizer.step()

        params_data_dict = {}
        for n, p in net.named_parameters():
            params_data_dict[n] = p.data
        param_ema.push(params_data_dict)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if 0 == batch_idx % 100 or batch_idx == len(trainloader) - 1:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                lr_schedule_value_curr = lr_schedule_values[it]
                it_curr = it
                wd_schedule_value_curr = wd_schedule_values[it]
            else:
                lr_schedule_value_curr = -1
                it_curr = -1
                wd_schedule_value_curr = -1
            logger.info(
                'Epoch: %d | (%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d) | Lr: %.7f | WD: %.7f | it: %d' % (
                    epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 
                    100.*correct/total, correct, total, lr_schedule_value_curr, wd_schedule_value_curr, it_curr
                )
            )
    if lr_schedule_values is not None or wd_schedule_values is not None:
        return train_loss/len(trainloader), 100.*correct/total, lr_schedule_values[epoch*num_training_steps_per_epoch], wd_schedule_values[epoch*num_training_steps_per_epoch]
    else:
        return train_loss/len(trainloader), 100.*correct/total


@timing
def train(args, logger, trainloader, device, epoch, net, optimizer, criterion, lr_schedule_values, wd_schedule_values, num_training_steps_per_epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if lr_schedule_values is not None or wd_schedule_values is not None:
            it = epoch * num_training_steps_per_epoch + batch_idx
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        total_cost = loss
        total_cost.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if 0 == batch_idx % 100 or batch_idx == len(trainloader) - 1:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                lr_schedule_value_curr = lr_schedule_values[it]
                it_curr = it
                wd_schedule_value_curr = wd_schedule_values[it]
            else:
                lr_schedule_value_curr = -1
                it_curr = -1
                wd_schedule_value_curr = -1
            logger.info(
                'Epoch: %d | (%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d) | Lr: %.7f | WD: %.7f | it: %d' % (
                    epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 
                    100.*correct/total, correct, total, lr_schedule_value_curr, wd_schedule_value_curr, it_curr
                )
            )
    if lr_schedule_values is not None or wd_schedule_values is not None:
        return train_loss/len(trainloader), 100.*correct/total, lr_schedule_values[epoch*num_training_steps_per_epoch], wd_schedule_values[epoch*num_training_steps_per_epoch]
    else:
        return train_loss/len(trainloader), 100.*correct/total


def test(args, logger, testloader, device, epoch, net, criterion, save_path, save=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if 0 == batch_idx % 100 or batch_idx == len(testloader) - 1:
                logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and save:
        logger.info('Saving best %.3f @ %d ...' %(acc, epoch))
        deploy_net = deepcopy(net.module)
        try: 
            deploy_net.switch_to_deploy()
        except:
            pass
        state = {
            'net': net.module,
            "deploy": deploy_net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(save_path, 'best_ckpt.pth'))
    best_acc = acc if acc > best_acc else best_acc
    logger.info('======> Best %.3f ===== ' %(best_acc))

    return test_loss/len(testloader), acc

