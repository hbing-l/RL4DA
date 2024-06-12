import logging
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
from model_cl_multitask import SDG, Predictor
from data import RotNISTDataLoader
import ot
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

# Assuming these classes are already defined:
# SDG, Predictor, FeatureExtractor, Classifiercc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_distribution_discrepancy(representations1, representations2, device):
    m = ot.dist(representations1, representations2, metric='euclidean')
    m = m / m.max()
    n1 = representations1.shape[0]
    n2 = representations2.shape[0]
    a, b = torch.ones((n1,)) / n1, torch.ones((n2,)) / n2
    m, a, b = m.to(device), a.to(device), b.to(device)
    w_dist = ot.sinkhorn2(a, b, m, 1)
    return w_dist

# Helper function to compute reward based on discrepancy measurement
def compute_reward(source_feature_, selected_domain, target_feature_, device, gamma=0.9):
    # Placeholder for actual discrepancy computation
    source_feature = source_feature_.view(source_feature_.shape[0], -1).clone()
    target_feature = target_feature_.view(target_feature_.shape[0], -1).clone()
    if len(selected_domain) == 0:
        reward = gamma * compute_distribution_discrepancy(source_feature, source_feature, device)
        logging.info(f"selected domain list=0, reward:{reward}")
    elif len(selected_domain) == 1:
        inter_feature = selected_domain[-1].clone()
        inter_feature = inter_feature.view(inter_feature.shape[0], -1)
        reward = gamma * compute_distribution_discrepancy(inter_feature, source_feature, device) - compute_distribution_discrepancy(source_feature, source_feature, device) \
                - (gamma * compute_distribution_discrepancy(inter_feature, target_feature, device) - compute_distribution_discrepancy(source_feature, target_feature, device))
        logging.info(f"selected domain list=1, reward:{reward}")
    else:
        current_inter_feature = selected_domain[-1].clone()
        current_inter_feature = current_inter_feature.view(current_inter_feature.shape[0], -1)
        prev_inter_feature = selected_domain[-2].clone()
        prev_inter_feature = prev_inter_feature.view(prev_inter_feature.shape[0], -1)
        reward = gamma * compute_distribution_discrepancy(current_inter_feature, source_feature, device) - compute_distribution_discrepancy(prev_inter_feature, source_feature, device) \
                - (gamma * compute_distribution_discrepancy(current_inter_feature, target_feature, device) - compute_distribution_discrepancy(prev_inter_feature, target_feature, device))
        logging.info(f"selected domain list>1, reward:{reward}")
    return reward

def calculate_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def eval(sdg, predictor, dataloader, intermediate_loaders, device):
    sdg.eval()
    predictor.eval()
    selected_domain = []
    selected_domain_index = []

    with torch.no_grad():
        data, label = [x.to(device) for x in dataloader]
        feature1, feature2, out = predictor(data)
        true_label = label.argmax(dim=1)
        pred = out.argmax(dim=1)
        correct = pred.eq(true_label).sum().float().item()
        acc = correct / label.shape[0]

        for loader_idx, loader in enumerate(intermediate_loaders):
            inter_data, inter_label = [x.to(device) for x in loader]
            inter_feature1, inter_feature2, inter_out = predictor(inter_data)
            inter_feature1_ = inter_feature1.detach().view(inter_feature1.shape[0], -1).clone()
            selection_probs = sdg(inter_feature1_)
            if selection_probs.mean() > 0.5:
                selected_domain.append(inter_feature1_.clone())
                selected_domain_index.append(loader_idx)
    
    return acc, selected_domain, selected_domain_index
        

def train(sdg, predictor, device, source_loader, intermediate_loaders, target_loader, test_intermediate_loaders, test_target_loader, epochs, n_j):
    optimizer_sdg = torch.optim.Adam(sdg.parameters(), lr=1e-3)
    optimizer_predictor = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    tick = 0
    tick1 = 0
    best_acc = 0

    for epoch in range(epochs):
        policy_losses = []  # For accumulating policy gradients
        value_losses = []   # For predictor loss
        selected_d = []
        all_rewards = []

        # Loop through each 'bag' (domain) multiple times
        for idx in range(n_j):
            # Shuffle the intermediate loaders (domains)
            setup_seed((epoch+1)*(idx+1))
            intermediate_idx = torch.randperm(len(intermediate_loaders)).tolist()
            selected_domain = []
            selected_domain_index = []
            selected_domain_probs = []
            episode_rewards = []
            log_probs = []
            
            for i, loader_idx in enumerate(intermediate_idx):
                intermediate_loader = intermediate_loaders[loader_idx]
                inter_data, inter_label = [x.to(device) for x in intermediate_loader]
                source_data, source_label = [x.to(device) for x in source_loader]
                target_data, target_label = [x.to(device) for x in target_loader]

                # inter_data, inter_label = intermediate_loader
                # source_data, source_label = source_loader
                # target_data, target_label = target_loader

                inter_feature1, inter_feature2, inter_out = predictor(inter_data)
                source_feature1, source_feature2, source_out = predictor(source_data)
                target_feature1, target_feature2, target_out = predictor(target_data)

                # Train the predictor on selected features using the classification loss
                # Assuming binary classification for simplicity
                source_feature2_ = source_feature2.view(source_feature2.shape[0], -1).clone()
                inter_feature2_ = inter_feature2.view(inter_feature2.shape[0], -1).clone()
                target_feature2_ = target_feature2.view(target_feature2.shape[0], -1).clone()
                distance1 = compute_distribution_discrepancy(source_feature2_, inter_feature2_, device)
                distance2 = compute_distribution_discrepancy(source_feature2_, target_feature2_, device)
                prediction_loss = criterion(source_out.float(), source_label.float())
                value_loss = distance1 + distance2 + prediction_loss
                # value_loss = nn.BCELoss()(predictions, torch.ones_like(predictions))
                value_losses.append(value_loss)
                writer.add_scalar("predictor loss", value_loss, tick)

                # Update the predictor
                optimizer_predictor.zero_grad()
                value_loss.backward(retain_graph=True)
                optimizer_predictor.step()

                del value_loss, prediction_loss

                # Get action and value from SDG and Predictor
                inter_feature1_ = inter_feature1.detach().view(inter_feature1.shape[0], -1).clone()
                selection_probs = sdg(inter_feature1_)
                if selection_probs.mean() > 0.7:
                    selected_domain.append(inter_feature1_.clone())
                    selected_domain_index.append(loader_idx)
                selected_domain_probs.append(selection_probs.mean().item())
                
                log_probs.append(torch.log(selection_probs))
                
                # Compute the Wasserstein distance as the reward
                # distance = compute_distribution_discrepancy(source_feature1, selected_domain, target_feature1)
                reward = compute_reward(source_feature1, selected_domain, target_feature1, device)  # We don't want to backprop through the reward
                reward = reward.detach()
                episode_rewards.append(reward)
                writer.add_scalar("reward after one selection within all domains", reward, tick)
                
                tick += 1

            logging.info(f"selection probs: {selected_domain_probs}")
            selected_d.append(selected_domain_index)
            returns = calculate_returns(episode_rewards, gamma=0.99)
            writer.add_scalar("reward after all selection within nj", torch.stack(returns).sum().item(), tick1)

            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-torch.sum(log_prob * R))
            
            writer.add_scalar("policy loss after all selection within nj", torch.stack(policy_loss).mean().item(), tick1)

            policy_losses.extend(policy_loss)
            all_rewards.extend(episode_rewards)

            tick1 += 1

        # Update the SDG policy
        optimizer_sdg.zero_grad()
        policy_loss = torch.stack(policy_losses).sum() / n_j
        policy_loss.backward()
        optimizer_sdg.step()

        writer.add_scalar("reward per epoch", torch.stack(all_rewards).mean().item(), epoch)
        writer.add_scalar("policy loss per epoch", policy_loss.item(), epoch)

        # Print training progress
        logging.info(f"Epoch {epoch + 1}/{epochs} - Policy Loss: {policy_loss.item()}, Predictor Loss: {torch.stack(value_losses).mean().item()}, Reward: {torch.stack(all_rewards).mean().item()}")
        logging.info(f"Selected domain: {selected_d}")

        acc, selected_domain, selected_domain_index = eval(sdg, predictor, test_target_loader, test_intermediate_loaders, device)
        writer.add_scalar("acc on test set per epoch", acc, epoch)
        logging.info(f"Accuracy on test target domain: {acc}")
        logging.info(f"Selected test domain index: {selected_domain_index}")

        if epoch % 10 == 0:
            torch.save({"state_dict":predictor.state_dict(), "epoch":epoch}, f'/home/hanbingliu/RL4DA/ckpt2/predictor_epoch{epoch}.bin')
            torch.save({"state_dict":sdg.state_dict(), "epoch":epoch}, f'/home/hanbingliu/RL4DA/ckpt2/sdg_epoch{epoch}.bin')

        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict":predictor.state_dict(), "epoch":epoch}, '/home/hanbingliu/RL4DA/ckpt2/predictor_best_model.bin')
            torch.save({"state_dict":sdg.state_dict(), "epoch":epoch}, '/home/hanbingliu/RL4DA/ckpt2/sdg_best_model.bin')
            logging.info(f"saving best testing model of acc {best_acc}")


localtime = time.localtime(time.time())
time1 = time.strftime("%m%d_%H%M",time.localtime(time.time()))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler('/home/hanbingliu/RL4DA/log/{}.log'.format(time1), mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 获取根日志器，并添加上面定义的两个处理器
logging.getLogger().addHandler(console_handler)
logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(logging.INFO)
writer = SummaryWriter('/home/hanbingliu/RL4DA/runs/{}/'.format(time1))

if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")

# Initialize the SDG and Predictor
sdg = SDG(input_dim=64*7*7, hidden_dim=256, output_dim=1).to(device)
predictor = Predictor().to(device)

RotNISTLoader = RotNISTDataLoader('/home/hanbingliu/angle', 0, [18, 36, 54, 72], 90, True, 'train', 256)
source_loader, intermediate_loaders, target_loader = RotNISTLoader.get_loaders()

RotNISTLoader = RotNISTDataLoader('/home/hanbingliu/angle', 0, [18, 36, 54, 72], 90, True, 'test', 256)
test_source_loader, test_intermediate_loaders, test_target_loader = RotNISTLoader.get_loaders()

RotNISTLoader = RotNISTDataLoader('/home/hanbingliu/angle', 0, [18, 36, 54, 72], 90, True, 'valid', 256)
valid_source_loader, valid_intermediate_loaders, valid_target_loader = RotNISTLoader.get_loaders()

# Train the models
train(sdg, predictor, device, source_loader, intermediate_loaders, target_loader, test_intermediate_loaders, test_target_loader, 100, 3)

# Eval the models
valid_sdg = SDG(input_dim=64*7*7, hidden_dim=256, output_dim=1).to(device)
valid_predictor = Predictor().to(device)
logging.info("Loading checkpoint ... ...")
saved_dict = torch.load("/home/hanbingliu/RL4DA/ckpt2/predictor_best_model.bin")
predictor_epoch = saved_dict['epoch']
predictor_state_dict = saved_dict['state_dict']
valid_predictor.load_state_dict(predictor_state_dict)

saved_dict = torch.load("/home/hanbingliu/RL4DA/ckpt2/sdg_best_model.bin")
sdg_epoch = saved_dict['epoch']
sdg_state_dict = saved_dict['state_dict']
valid_sdg.load_state_dict(sdg_state_dict)
logging.info(f"best model on sdg and precitor are trained for {sdg_epoch} and {predictor_epoch} epochs")

acc, selected_domain, selected_domain_index = eval(valid_sdg, valid_predictor, valid_target_loader, valid_intermediate_loaders, device)
logging.info(f"Accuracy on vaclid target domain: {acc}")
logging.info(f"Selected valid domain index: {selected_domain_index}")