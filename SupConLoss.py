import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        features = torch.nn.functional.normalize(features, dim=1)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss

import torch.nn.functional as F


def prompt_centloss(prompts, labels, similarity_metric='cosine'):

    batch_size, num_prompts, prompt_dim = prompts.shape

    # [batch_size, prompt_dim_total]
    prompts_flat = prompts.view(batch_size, -1)

    if similarity_metric == 'cosine':

        normed_prompts = F.normalize(prompts_flat, p=2, dim=-1)
        similarity_matrix = torch.matmul(normed_prompts, normed_prompts.T)  # [batch_size, batch_size]
    elif similarity_metric == 'euclidean':

        squared_norm = torch.sum(prompts_flat ** 2, dim=-1, keepdim=True)  # [batch_size, 1]
        similarity_matrix = squared_norm + squared_norm.T - 2 * torch.matmul(prompts_flat, prompts_flat.T)
        similarity_matrix = torch.sqrt(F.relu(similarity_matrix))


    labels_expand = labels.unsqueeze(1)  # [batch_size, 1]
    same_class_mask = (labels_expand == labels_expand.T).float()  # [batch_size, batch_size]


    if similarity_metric == 'cosine':

        similarity_loss = (1 - similarity_matrix) * same_class_mask
    elif similarity_metric == 'euclidean':

        similarity_loss = similarity_matrix * same_class_mask


    similarity_loss = similarity_loss.sum() / batch_size

    return similarity_loss


