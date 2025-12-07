import torch
from torch import nn
from transformers import BertModel


class BertRNNRecommender(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_movies: int,
        rnn_vocab_size: int,
        rnn_emb_dim: int = 256,
        rnn_hidden_dim: int = 128,
        mlp_hidden_dim: int = 512,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.num_movies = num_movies

        # BERT
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_hidden = self.bert.config.hidden_size  # 768

        # RNN para sequência de filmes
        self.movie_emb = nn.Embedding(
            num_embeddings=rnn_vocab_size,
            embedding_dim=rnn_emb_dim,
            padding_idx=0,  # 0 = PAD
        )
        self.gru = nn.GRU(
            input_size=rnn_emb_dim,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
        )

        # MLP final (BERT[CLS] + RNN -> logits por filme)
        self.fc1 = nn.Linear(bert_hidden + rnn_hidden_dim, mlp_hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(mlp_hidden_dim, num_movies)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        rnn_input,
        rnn_mask,
    ):
        # ----- ramo BERT -----
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = bert_out.last_hidden_state[:, 0, :]  # (B, 768)

        # ----- ramo RNN -----
        # rnn_input: (B, T) com índices já shiftados (0 = PAD)
        emb = self.movie_emb(rnn_input)           # (B, T, rnn_emb_dim)

        lengths = rnn_mask.sum(dim=1)             # (B,) número de filmes válidos
        lengths = lengths.cpu()

        # Empacota para lidar com comprimentos variáveis
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)                 # h_n: (1, B, rnn_hidden_dim)
        rnn_repr = h_n[-1]                        # (B, rnn_hidden_dim)

        # ----- combinação + MLP -----
        x = torch.cat([cls, rnn_repr], dim=1)     # (B, 768 + rnn_hidden_dim)
        x = self.fc1(x)
        x = self.act(x)
        logits = self.fc2(x)                      # (B, num_movies)

        # NÃO aplica sigmoid aqui; deixa para a loss (BCEWithLogits)
        return logits
