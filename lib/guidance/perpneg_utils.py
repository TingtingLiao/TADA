import torch
import math

# Please refer to the https://perp-neg.github.io/ for details about the paper and algorithm
def get_perpendicular_component(x, y):
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum())/max(torch.norm(y)**2, 1e-6)) * y


def batch_get_perpendicular_component(x, y):
    assert x.shape == y.shape
    result = []
    for i in range(x.shape[0]):
        result.append(get_perpendicular_component(x[i], y[i]))
    return torch.stack(result)


def weighted_perpendicular_aggregator(delta_noise_preds, weights, batch_size):
    """ 
    Notes: 
     - weights: an array with the weights for combining the noise predictions
     - delta_noise_preds: [B x K, 4, 64, 64], K = max_prompts_per_dir
    """
    delta_noise_preds = delta_noise_preds.split(batch_size, dim=0) # K x [B, 4, 64, 64]
    weights = weights.split(batch_size, dim=0) # K x [B]
    # print(f"{weights[0].shape = } {weights = }")

    assert torch.all(weights[0] == 1.0)

    main_positive = delta_noise_preds[0] # [B, 4, 64, 64]

    accumulated_output = torch.zeros_like(main_positive)
    for i, complementary_noise_pred in enumerate(delta_noise_preds[1:], start=1):
        # print(f"\n{i = }, {weights[i] = }, {weights[i].shape = }\n")

        idx_non_zero = torch.abs(weights[i]) > 1e-4
        
        # print(f"{idx_non_zero.shape = }, {idx_non_zero = }")
        # print(f"{weights[i][idx_non_zero].shape = }, {weights[i][idx_non_zero] = }")
        # print(f"{complementary_noise_pred.shape = }, {complementary_noise_pred[idx_non_zero].shape = }")
        # print(f"{main_positive.shape = }, {main_positive[idx_non_zero].shape = }")
        if sum(idx_non_zero) == 0:
            continue
        accumulated_output[idx_non_zero] += weights[i][idx_non_zero].reshape(-1, 1, 1, 1) * batch_get_perpendicular_component(complementary_noise_pred[idx_non_zero], main_positive[idx_non_zero])
    
    assert accumulated_output.shape == main_positive.shape, f"{accumulated_output.shape = }, {main_positive.shape = }"


    return accumulated_output + main_positive


def adjust_text_embeddings(embeddings, azimuth, opt):
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b])
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)  # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)  # [B * K]
    return text_embeddings, weights


def get_pos_neg_text_embeddings(embeddings, azimuth_val, front_decay_factor=2, side_decay_factor=10, negative_w=-2):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * front_decay_factor) * negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1 - r) * side_decay_factor) * negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = negative_w
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * side_decay_factor) * negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)