import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np


class PostEmbeddingDataset(Dataset):
    def __init__(self, posts, sbert_model):
        self.sbert = sbert_model
        # Create embeddings for all posts
        self.embeddings = self.sbert.encode(posts, convert_to_tensor=True)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, latent_dim=256):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add noise during training
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),  # Help keep the output in a similar range as SBERT embeddings
        )

    def forward(self, x):
        # Add noise during training
        if self.training:
            x = x + torch.randn_like(x) * 0.1

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def train_autoencoder(
    posts, sbert_model, num_epochs=10, batch_size=32, learning_rate=1e-4
):
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = DenoisingAutoencoder().to(device)

    # Create dataset and dataloader
    dataset = PostEmbeddingDataset(posts, sbert_model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            reconstructed, encoded = autoencoder(batch)
            loss = criterion(reconstructed, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    return autoencoder


class EnhancedEmbedder:
    def __init__(self, sbert_model, autoencoder):
        self.sbert = sbert_model
        self.autoencoder = autoencoder
        self.device = next(autoencoder.parameters()).device

    def encode(self, texts):
        # Get SBERT embeddings
        with torch.no_grad():
            embeddings = self.sbert.encode(texts, convert_to_tensor=True).to(
                self.device
            )
            # Get enhanced embeddings through the encoder part only
            _, enhanced = self.autoencoder(embeddings)
        return enhanced
