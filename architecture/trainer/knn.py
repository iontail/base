import torch

class KNN:
    def __init__(self, k=1):
        self.k = k
        self.x = None
        self.y = None


    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x: torch.Tensor, metric: str = 'euclidean'):
        # compute knn metric by loop

        if x.ndim == 1:
            x = x.unsqueeze(0)

        N = x.size(0)
        pred = []
        for i in range(N):

            feature_i = x[i]
            if metric == 'euclidean':
                distances = torch.pow(feature_i - self.x, 2).sum(dim=1)
            elif metric == 'cosine':
                # cosine sim's input should be 2D tensor (1, D)
                distances = 1 - torch.nn.functional.cosine_similarity(feature_i.unsqueeze(0), self.x, dim=1)

            nn_id = torch.argmin(distances)

            predicted_label = self.y[nn_id]

            pred.append(predicted_label)

        return torch.stack(pred, dim=0)