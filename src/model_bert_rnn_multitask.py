# model_bert_rnn_multitask.py

import torch
from torch import nn
from transformers import BertModel


class BertRNNMultiTaskRecommender(nn.Module):
    """
    Multi-task:
      - Tarefa 1 (ReDial): dado usuário (texto + sequência de filmes),
        prever distribuição multi-label sobre filmes (num_movies).
      - Tarefa 2 (MovieLens): dado filme, prever tags multi-label (num_tags).

    Compartilha um embedding de filmes entre as duas tarefas.
    """

    def __init__(
        self,
        model_name: str,
        num_movies: int,
        num_tags: int,
        movie_emb_dim: int = 128,
        rnn_hidden_dim: int = 128,
        tag_hidden_dim: int = 256,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.num_movies = num_movies
        self.num_tags = num_tags

        # -------- BERT --------
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        bert_hidden = self.bert.config.hidden_size  # 768

        # -------- Movie embeddings compartilhados --------
        # índice 0 = PAD, filmes reais vão de 1..num_movies
        self.movie_emb = nn.Embedding(
            num_embeddings=num_movies + 1,
            embedding_dim=movie_emb_dim,
            padding_idx=0,
        )

        # -------- RNN sobre sequência de filmes --------
        self.gru = nn.GRU(
            input_size=movie_emb_dim,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
        )

        # -------- Projeção usuário → mesmo espaço das embeddings de filme --------
        self.user_proj = nn.Linear(bert_hidden + rnn_hidden_dim, movie_emb_dim)
        self.user_act = nn.ReLU()

        # -------- Cabeça de tags (MovieLens) --------
        self.tag_fc1 = nn.Linear(movie_emb_dim, tag_hidden_dim)
        self.tag_act = nn.ReLU()
        self.tag_fc2 = nn.Linear(tag_hidden_dim, num_tags)

    # ------------------- ReDial (usuário → logits de filmes) -------------------

    def forward_redial(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        rnn_input,
        rnn_mask,
    ):
        """
        input_ids, attention_mask, token_type_ids: BERT
        rnn_input: (B, T) com índices 0..num_movies (0=PAD, 1..num_movies = filmes)
        rnn_mask: (B, T) booleans (True = token de filme válido)
        """
        # BERT
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = bert_out.last_hidden_state[:, 0, :]  # (B, bert_hidden)

        # RNN de filmes
        emb = self.movie_emb(rnn_input)  # (B, T, movie_emb_dim)
        lengths = rnn_mask.sum(dim=1).cpu()  # (B,)

        if lengths.max() == 0:
            # caso degenerado: ninguém tem filme, zera saída da RNN
            batch_size = rnn_input.size(0)
            rnn_repr = torch.zeros(
                batch_size,
                self.gru.hidden_size,
                device=rnn_input.device,
            )
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.gru(packed)  # (1, B, hidden)
            rnn_repr = h_n[-1]         # (B, hidden)

        # Combina BERT + RNN e projeta para espaço de filmes
        user_hidden = torch.cat([cls, rnn_repr], dim=1)  # (B, bert_hidden + rnn_hidden)
        user_vec = self.user_act(self.user_proj(user_hidden))  # (B, movie_emb_dim)

        # Logits de filmes = user_vec · movie_emb^T (ignora índice 0/PAD)
        movie_emb_weight = self.movie_emb.weight[1:]  # (num_movies, movie_emb_dim)
        logits = torch.matmul(user_vec, movie_emb_weight.T)  # (B, num_movies)

        return logits

    # ------------------- MovieLens (filme → logits de tags) -------------------

    def forward_tags(self, movie_indices):
        """
        movie_indices: (B,) com índices internos de filmes (0..num_movies-1).
        """
        # shift +1 para usar a mesma embedding (0 é PAD)
        emb = self.movie_emb(movie_indices + 1)  # (B, movie_emb_dim)

        h = self.tag_act(self.tag_fc1(emb))
        logits_tags = self.tag_fc2(h)  # (B, num_tags)

        return logits_tags

    # Opcional: forward padrão = tarefa de ReDial
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        rnn_input,
        rnn_mask,
    ):
        return self.forward_redial(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            rnn_input=rnn_input,
            rnn_mask=rnn_mask,
        )
