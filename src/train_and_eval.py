import copy
import torch
import numpy as np

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training for teacher GNNs
def train(model, g, feats, labels, criterion, optimizer, idx):

    model.train()

    logits = model(g, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx], labels[idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Testing for teacher GNNs
def evaluate(model, g, feats):

    model.eval()

    with torch.no_grad():
        logits = model(g, feats)
        out = logits.log_softmax(dim=1)

    return logits, out


# Training for student MLPs
def train_mini_batch(model, edge_idx, g, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx, edge_weight, param):

    model.train()

    logits = model(g, feats)
    out = logits.log_softmax(dim=1)
    loss_l = criterion_l(out[idx], labels[idx])
    if param['model_mode'] == 0:
        s_logits = (logits/param['tau']).log_softmax(dim=1)[edge_idx[0]]
        t_logits = (out_t_all/param['tau']).log_softmax(dim=1)[edge_idx[1]]
    elif param['model_mode'] == 1:
        edge_weight = edge_weight * torch.tensor(np.random.beta(param['beta'], param['beta'], edge_weight.shape[0])).view(-1, 1).to(device)
        s_logits = (logits/param['tau']).log_softmax(dim=1)[edge_idx[0]]
        log_target = (out_t_all/param['tau']).softmax(dim=1)[edge_idx[0]] * (1 - edge_weight) + (out_t_all/param['tau']).softmax(dim=1)[edge_idx[1]] * edge_weight
        if param['dataset'] != 'cora' and param['dataset'] != 'citeseer' and param['dataset'] != 'pubmed':
            t_logits = ((log_target + 1e-6) / (log_target + 1e-6).sum(dim=1, keepdim=True)).log()
        else:
            t_logits = log_target.log()    
    elif param['model_mode'] == 2:
        g = dgl.graph((edge_idx[0], edge_idx[1]), num_nodes=feats.shape[0]).to(device)
        weighted_coefficient = model.cal_weighted_coefficient(g, out_t_all.softmax(dim=1), edge_weight, edge_idx[0])
        s_logits = (logits/param['tau']).log_softmax(dim=1)[edge_idx[0]]
        t_logits = (out_t_all/param['tau']).log_softmax(dim=1)[edge_idx[1]]

    if param['model_mode'] == 2:
        loss_t = (criterion_t(s_logits, t_logits.detach()).sum(dim=-1) * weighted_coefficient).mean()
    else:
        loss_t = criterion_t(s_logits, t_logits.detach()).sum(dim=-1).mean()
    loss = loss_l * param['lambda'] + loss_t * (1 - param['lambda'])

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss_l.item() * param['lambda'], loss_t.item() * (1-param['lambda'])


# Testing for student MLPs
def evaluate_mini_batch(model, g, feats):

    model.eval()

    with torch.no_grad():
        logits = model(g, feats)
        out = logits.log_softmax(dim=1)

    return logits, out


def train_teacher(param, model, g, feats, labels, indices, criterion, evaluator, optimizer):

    idx_train, idx_val, idx_test = indices
    g = g.to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):
        
        train_loss = train(model, g, feats, labels, criterion, optimizer, idx_train)
        _, out = evaluate(model, g, feats)
        train_acc = evaluator(out[idx_train], labels[idx_train])
        val_acc = evaluator(out[idx_val], labels[idx_val])
        test_acc = evaluator(out[idx_test], labels[idx_test])

        if epoch % 10 == 0:
            print("\033[0;30;46m [{}] CLA: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                                        epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_val, test_best))

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break


    model.load_state_dict(state)
    model.eval()
    out, _ = evaluate(model, g, feats)

    return out, test_acc, test_val, test_best, state


def train_student(param, model, g, feats, labels, out_t_all, indices, criterion_l, criterion_t, evaluator, optimizer):

    idx_train, idx_val, idx_test = indices
    edge_idx_list = extract_indices(g)
    g = g.to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):

        if epoch <= 1:
            edge_idx = edge_idx_list[0]   
            edge_weight = torch.ones((edge_idx.shape[1])).view(-1, 1).to(device)  

        else:
            alpha = param['alpha'] / (100 ** ((epoch-1) / (param["max_epoch"] - 1)))
            edge_prob = subgraph_extractor(out_t_all, logits_s, edge_idx_list[1], alpha)
            sampling_mask = torch.bernoulli(torch.tensor(edge_prob)).bool() 
            edge_idx = torch.masked_select(edge_idx_list[1], sampling_mask).view(2, -1).detach().cpu().numpy().swapaxes(1, 0)

            edge_idx = edge_idx.tolist()
            for i in range(feats.shape[0]):
                edge_idx.append([i, i])
            edge_idx = np.array(edge_idx).swapaxes(1, 0)  
            edge_weight = torch.cat([torch.tensor(edge_prob)[sampling_mask], torch.ones((feats.shape[0]))]).view(-1, 1).to(device)       

        loss_l, loss_t = train_mini_batch(model, edge_idx, g, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx_train, edge_weight, param)
        logits_s, out = evaluate_mini_batch(model, g, feats)
        train_acc = evaluator(out[idx_train], labels[idx_train])
        val_acc = evaluator(out[idx_val], labels[idx_val])
        test_acc = evaluator(out[idx_test], labels[idx_test])
            
        if epoch % 10 == 0:
            print("\033[0;30;43m [{}] CLA: {:.5f}, KD: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}  | #Edge: {:}\033[0m".format(
                                        epoch, loss_l, loss_t, loss_l + loss_t, train_acc, val_acc, test_acc, val_best, test_val, test_best, edge_idx.shape[1]))

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break


    model.load_state_dict(state)
    model.eval()
    out, _ = evaluate_mini_batch(model, g, feats)

    return out, test_acc, test_val, test_best, state