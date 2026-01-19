import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticProbe(nn.Module):
    """
    Simple linear probe (Logistic Regression) as used in the paper.
    Maps representation dim -> 1 (logit).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class RINE:
    """
    Redundant Information Neural Estimator (RINE).
    """
    def __init__(self, dim_base, dim_unlearned, lr=1e-3, device='cuda'):
        self.f1 = LogisticProbe(dim_base).to(device)      # Decoder for Base
        self.f2 = LogisticProbe(dim_unlearned).to(device) # Decoder for Unlearned
        self.optimizer = optim.Adam(list(self.f1.parameters()) + list(self.f2.parameters()), lr=lr)
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def train_estimator(self, Z_base, Z_unlearned, Y, beta=5.0, epochs=1000, verbose=True):
        self.f1.train()
        self.f2.train()
        
        # Convert inputs to tensors
        Z_b = torch.tensor(Z_base, dtype=torch.float32).to(self.device)
        Z_u = torch.tensor(Z_unlearned, dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            logits_1 = self.f1(Z_b)
            logits_2 = self.f2(Z_u)
            
            # 1. Prediction Loss (Cross Entropy)
            loss_1 = self.bce(logits_1, Y_t)
            loss_2 = self.bce(logits_2, Y_t)
            
            # 2. Consistency Constraint (L1 distance of probabilities)
            probs_1 = torch.sigmoid(logits_1)
            probs_2 = torch.sigmoid(logits_2)
            dist_loss = torch.mean(torch.abs(probs_1 - probs_2))
            
            # 3. Lagrangian Loss (Eq 12 in paper)
            total_loss = 0.5 * (loss_1 + loss_2) + beta * dist_loss
            
            total_loss.backward()
            self.optimizer.step()
            
        return self.evaluate(Z_b, Z_u, Y_t)

    def evaluate(self, Z_b, Z_u, Y):
        self.f1.eval()
        self.f2.eval()
        
        with torch.no_grad():
            logits_1 = self.f1(Z_b)
            logits_2 = self.f2(Z_u)
            
            ce_1 = self.bce(logits_1, Y).item()
            ce_2 = self.bce(logits_2, Y).item()
            
            # Average prediction entropy under constraint
            L_cap = 0.5 * (ce_1 + ce_2)
            
        # H(Y): Entropy of labels (in nats)
        p = Y.mean().item()
        if 0 < p < 1:
            H_Y = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        else:
            H_Y = 0

        # Convert nats to bits
        log2 = np.log(2)
        
        I_cap = (H_Y - L_cap) / log2       # Residual Knowledge
        I_base = (H_Y - ce_1) / log2       # Total info in Base
        I_uniq_B = I_base - I_cap          # Unlearned Knowledge
        
        return {
            "Residual Knowledge": max(0, I_cap),
            "Unlearned Knowledge": max(0, I_uniq_B),
            "Total Base Info": I_base,
            "logits_base": logits_1.cpu().numpy(),
            "logits_unlearned": logits_2.cpu().numpy()
        }