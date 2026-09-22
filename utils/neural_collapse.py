import logging
import numpy as np
import torch
import torch.nn.functional as F


class NeuralCollapseMetrics:
    """
    SEC-Prompt 등 FSCIL 모델에 붙여서 세션별 NC 측정
    """

    @staticmethod
    def compute_etf(K: int, d: int, device='cuda') -> torch.Tensor:
        """
        K개 클래스의 이상적 ETF 행렬 생성 (closed-form)
        Returns: (K, K) ETF 행렬
        """
        ETF = torch.eye(K, device=device) - (1.0 / K) * torch.ones(K, K, device=device)
        ETF = ETF * np.sqrt(K / (K - 1))
        return ETF  # shape: (K, K)

    @staticmethod
    def compute_nc1(features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        NC1: Within-class variability collapse
        = trace(Σ_W) / trace(Σ_B)
        """
        classes = labels.unique()
        K = len(classes)
        d = features.shape[1]

        mu_G = features.mean(dim=0)

        Sigma_W = torch.zeros(d, d, device=features.device)
        Sigma_B = torch.zeros(d, d, device=features.device)

        for c in classes:
            mask = (labels == c)
            f_c = features[mask]
            n_c = f_c.shape[0]
            mu_c = f_c.mean(dim=0)

            diff_W = f_c - mu_c
            Sigma_W += (diff_W.T @ diff_W) / features.shape[0]

            diff_B = (mu_c - mu_G).unsqueeze(1)
            Sigma_B += (n_c / features.shape[0]) * (diff_B @ diff_B.T)

        nc1 = torch.trace(Sigma_W) / (torch.trace(Sigma_B) + 1e-8)
        return nc1.item()

    @staticmethod
    def compute_nc2(classifier_weights: torch.Tensor) -> float:
        """
        NC2 (Version B): ETF deviation in cosine space.

        Let W be the classifier weight matrix whose rows correspond to classes.
        We first L2-normalize each row:

          W_hat[i] = W[i] / ||W[i]||_2

        and then build the cosine-similarity Gram matrix:

          G = W_hat W_hat^T

        The reported metric is:

          ||G / ||G||_F - ETF / ||ETF||_F||_F
        """
        K = classifier_weights.shape[0]

        W_norm = F.normalize(classifier_weights, dim=1)  # (K, d)
        WWT = W_norm @ W_norm.T                           # (K, K)

        ETF = NeuralCollapseMetrics.compute_etf(K, classifier_weights.shape[1],
                                                device=classifier_weights.device)
        nc2 = torch.norm(
            WWT / WWT.norm(p='fro') - ETF / ETF.norm(p='fro'),
            p='fro',
        )
        return nc2.item()

    @staticmethod
    def compute_nc3(features: torch.Tensor,
                    labels: torch.Tensor,
                    classifier_weights: torch.Tensor) -> float:
        """
        NC3: Self-duality — class mean과 classifier weight의 정렬
        = ||W̃ − M̃||_F

        Class means are computed from the provided ``features``/``labels`` pair.
        Classifier rows are matched using the actual label ids present in
        ``labels``.
        """
        classes = labels.unique(sorted=True)
        K = len(classes)
        d = features.shape[1]

        class_means = torch.zeros(K, d, device=features.device)
        for i, c in enumerate(classes):
            class_means[i] = features[labels == c].mean(dim=0)

        M_norm = F.normalize(class_means, dim=1)
        W_norm = F.normalize(classifier_weights[classes.long()], dim=1)

        nc3 = torch.norm(W_norm - M_norm, p='fro')
        return nc3.item()

    @staticmethod
    def compute_nc3_loss(features: torch.Tensor,
                         targets: torch.Tensor,
                         classifier_weights: torch.Tensor) -> torch.Tensor:
        """
        Differentiable, scale-normalized NC3 loss.

        For the classes present in the batch, compute normalized class means and
        classifier directions, then minimize their average cosine distance:

            L_NC3 = mean_c [ 1 - cos(mu_c, w_c) ]

        This stays in [0, 2], making it more scale-compatible with the
        normalized NC2 loss when combined under a single lambda.
        """
        classes = targets.unique(sorted=True)
        if classes.numel() == 0:
            return features.new_zeros(())

        class_means = []
        class_weights = []
        for c in classes:
            mask = targets == c
            f_c = features[mask]
            if f_c.numel() == 0:
                continue
            class_means.append(f_c.mean(dim=0))
            class_weights.append(classifier_weights[int(c.item())])

        if not class_means:
            return features.new_zeros(())

        means = torch.stack(class_means, dim=0)
        weights = torch.stack(class_weights, dim=0)
        means = F.normalize(means, dim=1)
        weights = F.normalize(weights, dim=1)
        cosine = (means * weights).sum(dim=1)
        return (1.0 - cosine).mean()

    @staticmethod
    def compute_nc4(features: torch.Tensor,
                    labels: torch.Tensor,
                    classifier_weights: torch.Tensor) -> float:
        """
        NC4: Decision rule agreement
        Version B로 고정:
        nearest-W 분류 == nearest-μ cosine 분류 일치율

        Both ``pred_W`` and ``pred_mu`` are computed in the class order induced
        by ``labels.unique(sorted=True)``.
        """
        classes = labels.unique(sorted=True)
        K = len(classes)
        d = features.shape[1]

        class_means = torch.zeros(K, d, device=features.device)
        for i, c in enumerate(classes):
            class_means[i] = features[labels == c].mean(dim=0)

        W_norm = F.normalize(classifier_weights[classes.long()], dim=1)
        f_norm = F.normalize(features, dim=1)
        pred_W = (f_norm @ W_norm.T).argmax(dim=1)

        mu_norm = F.normalize(class_means, dim=1)
        pred_mu = (f_norm @ mu_norm.T).argmax(dim=1)

        nc4 = (pred_W == pred_mu).float().mean()
        return nc4.item()

    @staticmethod
    def compute_nc1_loss(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Differentiable within-class compactness loss for training (no .item()).
        Features are L2-normalized first so the loss is bounded in [0, 4],
        making it scale-compatible with CE loss across different lambda values.
        L = mean over classes of mean squared distance to normalized class mean.
        """
        f_norm = F.normalize(features, dim=1)
        classes = targets.unique()
        loss = features.new_zeros(1).squeeze()
        for c in classes:
            mask = (targets == c)
            f_c = f_norm[mask]
            mu_c = f_c.mean(dim=0)
            loss = loss + ((f_c - mu_c) ** 2).sum(dim=1).mean()
        return loss / len(classes)

    @staticmethod
    def compute_supcon_loss(features: torch.Tensor, targets: torch.Tensor,
                            temperature: float = 0.07) -> torch.Tensor:
        """
        Supervised Contrastive Loss (Khosla et al. 2020).
        Pulls same-class features together, pushes different-class apart.
        Features are L2-normalized internally. Loss scale ~ [0, log(N)].
        """
        f_norm = F.normalize(features, dim=1)
        N = f_norm.shape[0]
        sim = torch.matmul(f_norm, f_norm.T) / temperature  # (N, N)

        # mask[i,j]=1 if i,j same class and i≠j
        labels = targets.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        # exclude self from denominator
        self_mask = torch.eye(N, device=features.device)
        exp_sim = torch.exp(sim) * (1 - self_mask)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()

    @staticmethod
    def build_etf_vectors(K: int, d: int, device, seed: int = 42) -> torch.Tensor:
        """
        K개 ETF 단위벡터를 d-dim 공간에 생성 (closed-form).
        cos(v_i, v_j) = -1/(K-1) for all i != j
        Returns: (K, d)
        """
        if K <= 1:
            return torch.zeros(K, d, device=device)

        G = (K / (K - 1)) * (
            torch.eye(K, device=device)
            - (1.0 / K) * torch.ones(K, K, device=device)
        )
        eigvals, eigvecs = torch.linalg.eigh(G)   # ascending order
        eigvals_r = eigvals[1:].clamp(min=0)       # (K-1,) non-zero eigenvalues
        eigvecs_r = eigvecs[:, 1:]                 # (K, K-1)
        W_r = eigvecs_r * eigvals_r.sqrt().unsqueeze(0)  # (K, K-1)

        gen = torch.Generator()
        gen.manual_seed(seed)
        rand = torch.randn(d, K - 1, generator=gen)
        Q, _ = torch.linalg.qr(rand)              # (d, K-1) orthonormal columns
        Q = Q.to(device)

        return W_r @ Q.T  # (K, d), ||row_i|| ≈ sqrt(K/(K-1)) ≈ 1

    @staticmethod
    def compute_nc2_from_centroids(features: torch.Tensor, labels: torch.Tensor) -> float:
        """NC2 computed from class mean centroids of features (not classifier weights)."""
        classes = labels.unique(sorted=True)
        K = len(classes)
        centroids = torch.zeros(K, features.shape[1], device=features.device)
        for i, c in enumerate(classes):
            centroids[i] = features[labels == c].mean(dim=0)
        return NeuralCollapseMetrics.compute_nc2(centroids)

    @staticmethod
    def compute_angular_nonunif(vectors: torch.Tensor) -> float:
        """
        Angular non-uniformity: std of off-diagonal pairwise cosine similarities.
        Lower = more uniform angular distribution (closer to ETF ideal).
        """
        V = F.normalize(vectors, dim=1)
        G = V @ V.T
        K = G.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=vectors.device)
        return G[mask].std().item()

    @staticmethod
    def compute_topk_proximal_cosine(vectors: torch.Tensor, k: int = 5) -> float:
        """
        Top-K proximal pair mean cosine: for each class, mean cosine to its K nearest neighbors.
        Higher = nearby classes are too similar (less separation).
        """
        V = F.normalize(vectors, dim=1)
        G = V @ V.T
        G.fill_diagonal_(-2.0)
        k = min(k, G.shape[0] - 1)
        return G.topk(k, dim=1).values.mean().item()

    @staticmethod
    def measure_init(features: torch.Tensor, labels: torch.Tensor,
                     classifier_weights: torch.Tensor, k: int = 5) -> dict:
        """
        Full init-stage measurement: NC2 (weight), NC2 (centroid), angular_nonunif, topk_proximal.
        Runs on both train-no-aug and test feature sets (call once per split).
        """
        classes = labels.unique(sorted=True)
        K = len(classes)
        centroids = torch.zeros(K, features.shape[1], device=features.device)
        for i, c in enumerate(classes):
            centroids[i] = features[labels == c].mean(dim=0)

        nc2_w  = NeuralCollapseMetrics.compute_nc2(classifier_weights)
        nc2_c  = NeuralCollapseMetrics.compute_nc2(centroids)
        ang    = NeuralCollapseMetrics.compute_angular_nonunif(centroids)
        topk   = NeuralCollapseMetrics.compute_topk_proximal_cosine(centroids, k=k)
        return {'NC2_weight': nc2_w, 'NC2_centroid': nc2_c,
                'angular_nonunif': ang, f'top{k}_proximal': topk}

    @staticmethod
    def compute_nc2_loss(classifier_weights: torch.Tensor) -> torch.Tensor:
        """
        Differentiable NC2 loss for training (no .item()).
        """
        K = classifier_weights.shape[0]
        W_norm = F.normalize(classifier_weights, dim=1)
        WWT = W_norm @ W_norm.T
        ETF = NeuralCollapseMetrics.compute_etf(K, classifier_weights.shape[1],
                                                device=classifier_weights.device)
        return torch.norm(
            WWT / WWT.norm(p='fro') - ETF / ETF.norm(p='fro'),
            p='fro',
        )

    @staticmethod
    def decompose_nc2(classifier_weights: torch.Tensor, init_cls: int) -> dict:
        """
        NC2 붕괴를 base/new 관점에서 3가지로 분해.

        전체 K_total × K_total Gram matrix G를 블록으로 나눠
        각 블록이 ideal ETF_K의 대응 블록에서 얼마나 벗어났는지 측정.

          G = [G_bb  G_bn]    ETF_K = [E_bb  E_bn]
              [G_nb  G_nn]             [E_nb  E_nn]

        Returns:
            nc2_bb  : base-base 블록 편차  (frozen weights → 거의 상수 예상)
            nc2_nn  : new-new 블록 편차    (5-shot prototype → 높을 것으로 예상)
            nc2_bn  : cross 블록 편차      (base-new 통합 품질)
            nc2_new_standalone : new 클래스 weight만 K_new 크기 ETF와 독립 비교
        """
        K_total = classifier_weights.shape[0]
        K_base = int(init_cls)
        K_new = K_total - K_base

        if K_new <= 0:
            return {}

        W = F.normalize(classifier_weights, dim=1)   # (K_total, d)
        G = W @ W.T                                   # (K_total, K_total)

        ETF_full = NeuralCollapseMetrics.compute_etf(
            K_total, classifier_weights.shape[1], device=classifier_weights.device
        )

        G_bb = G[:K_base, :K_base]
        G_nn = G[K_base:, K_base:]
        G_bn = G[:K_base, K_base:]

        E_bb = ETF_full[:K_base, :K_base]
        E_nn = ETF_full[K_base:, K_base:]
        E_bn = ETF_full[:K_base, K_base:]

        def _block_dev(A: torch.Tensor, B: torch.Tensor) -> float:
            a_f = A.norm(p='fro')
            b_f = B.norm(p='fro')
            if a_f < 1e-8 or b_f < 1e-8:
                return float('nan')
            return torch.norm(A / a_f - B / b_f, p='fro').item()

        nc2_bb = _block_dev(G_bb, E_bb)
        nc2_nn = _block_dev(G_nn, E_nn)
        nc2_bn = _block_dev(G_bn, E_bn)

        # new 클래스 weight만 뽑아 K_new 크기 ETF와 독립 비교
        if K_new >= 2:
            W_new = F.normalize(classifier_weights[K_base:], dim=1)
            G_new = W_new @ W_new.T
            ETF_new = NeuralCollapseMetrics.compute_etf(
                K_new, classifier_weights.shape[1], device=classifier_weights.device
            )
            nc2_new_standalone = torch.norm(
                G_new / G_new.norm(p='fro') - ETF_new / ETF_new.norm(p='fro'),
                p='fro',
            ).item()
        else:
            nc2_new_standalone = float('nan')

        return {
            'nc2_bb': nc2_bb,
            'nc2_nn': nc2_nn,
            'nc2_bn': nc2_bn,
            'nc2_new_standalone': nc2_new_standalone,
        }

    @classmethod
    def measure_all(cls,
                    features: torch.Tensor,
                    labels: torch.Tensor,
                    classifier_weights: torch.Tensor,
                    session: int = 0) -> dict:
        """
        NC1~NC4 전체 측정.

        ``classifier_weights`` must contain exactly one row per class present in
        ``labels``.
        """
        classes = labels.unique(sorted=True)
        num_classes = int(classes.numel())
        assert classifier_weights.shape[0] == num_classes, (
            f"NC metric K mismatch: weights={classifier_weights.shape[0]}, "
            f"labels={num_classes}"
        )
        if num_classes > 0:
            assert int(classes.max().item()) < classifier_weights.shape[0], (
                f"Label id {int(classes.max().item())} is out of range for "
                f"{classifier_weights.shape[0]} classifier rows"
            )
        nc1 = cls.compute_nc1(features, labels)
        nc2 = cls.compute_nc2(classifier_weights)
        nc3 = cls.compute_nc3(features, labels, classifier_weights)
        nc4 = cls.compute_nc4(features, labels, classifier_weights)

        result = {
            'session': session,
            'NC1': nc1,
            'NC2': nc2,
            'NC3': nc3,
            'NC4': nc4,
        }
        return result


def pearson_corr(xs, ys):
    """
    Safe Pearson correlation helper.
    """
    if xs is None or ys is None:
        return None
    if len(xs) < 3 or len(ys) < 3:
        logging.debug("Pearson correlation skipped because fewer than 3 points are available.")
        return None
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])
