"""
BERT (Bidirectional Encoder Representations from Transformers)
Implementação completa do zero com PyTorch

Estrutura deste arquivo:
  PARTE 1 — Tokenizador e Vocabulário
  PARTE 2 — Componentes da Arquitetura BERT
  PARTE 3 — Pré-treinamento (MLM + NSP)
  PARTE 4 — Fine-tuning para Análise de Sentimento
  PARTE 5 — Exemplo de uso completo
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import random
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PARTE 1 — TOKENIZADOR E VOCABULÁRIO
# ─────────────────────────────────────────────────────────────────────────────
#
# O BERT usa WordPiece tokenization: palavras raras são quebradas em sub-tokens.
# Aqui implementamos uma versão simplificada baseada em caracteres e palavras.
#
# Tokens especiais do BERT:
#   [PAD]   — preenchimento até atingir comprimento fixo
#   [UNK]   — palavra desconhecida (fora do vocabulário)
#   [CLS]   — início de cada sequência; o embedding deste token é usado
#             para tarefas de classificação
#   [SEP]   — separador entre duas sentenças (usada no NSP)
#   [MASK]  — substitui tokens durante o Masked Language Modeling

class Vocabulario:
    """
    Constrói e gerencia o vocabulário do modelo.

    Como funciona:
      1. Conta a frequência de cada palavra no corpus
      2. Mantém apenas as `tamanho_max` mais frequentes
      3. Adiciona os tokens especiais obrigatórios do BERT
    """

    TOKENS_ESPECIAIS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self, tamanho_max: int = 10000):
        self.tamanho_max = tamanho_max
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    def construir(self, textos: list[str]):
        # Conta frequência de todas as palavras no corpus
        contador = Counter()
        for texto in textos:
            tokens = self._tokenizar_basico(texto)
            contador.update(tokens)

        # Monta o vocabulário: especiais primeiro, depois os mais frequentes
        vocab = self.TOKENS_ESPECIAIS + [
            palavra
            for palavra, _ in contador.most_common(self.tamanho_max - len(self.TOKENS_ESPECIAIS))
        ]

        self.token2idx = {tok: idx for idx, tok in enumerate(vocab)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

    def _tokenizar_basico(self, texto: str) -> list[str]:
        # Converte para minúsculas e separa pontuação de palavras
        texto = texto.lower()
        texto = re.sub(r"([^\w\s])", r" \1 ", texto)
        return texto.split()

    def codificar(self, texto: str, max_len: int = 128) -> list[int]:
        tokens = ["[CLS]"] + self._tokenizar_basico(texto) + ["[SEP]"]
        tokens = tokens[:max_len]  # trunca se necessário

        ids = [self.token2idx.get(t, self.token2idx["[UNK]"]) for t in tokens]

        # Preenche com [PAD] até atingir max_len
        ids += [self.token2idx["[PAD]"]] * (max_len - len(ids))
        return ids

    def decodificar(self, ids: list[int]) -> str:
        return " ".join(
            self.idx2token.get(i, "[UNK]")
            for i in ids
            if i != self.token2idx.get("[PAD]", 0)
        )

    def salvar_json(self, caminho: str):
        """
        Persiste o vocabulário em JSON com três seções:
          metadata   — tamanho_max e quantidade real de tokens
          token2idx  — mapeamento token → índice (usado na codificação)
          idx2token  — mapeamento índice → token (usado na decodificação)
        """
        dados = {
            "metadata": {
                "tamanho_max":    self.tamanho_max,
                "tamanho_real":   len(self.token2idx),
                "tokens_especiais": self.TOKENS_ESPECIAIS,
            },
            "token2idx": self.token2idx,
            "idx2token": {str(k): v for k, v in self.idx2token.items()},
        }
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        print(f"     Vocabulário salvo em: {caminho}  ({len(self.token2idx)} tokens)")

    @classmethod
    def carregar_json(cls, caminho: str) -> "Vocabulario":
        """
        Reconstrói um Vocabulario a partir de um arquivo JSON salvo por salvar_json.
        """
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)

        obj = cls(tamanho_max=dados["metadata"]["tamanho_max"])
        obj.token2idx = dados["token2idx"]
        obj.idx2token = {int(k): v for k, v in dados["idx2token"].items()}
        return obj

    def __len__(self):
        return len(self.token2idx)


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 2 — COMPONENTES DA ARQUITETURA BERT
# ─────────────────────────────────────────────────────────────────────────────

# ── 2.1  EMBEDDINGS ──────────────────────────────────────────────────────────
#
# O BERT soma três tipos de embedding para cada token de entrada:
#
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  Entrada final = Token Embedding                                 │
#  │               + Positional Embedding  (posição na sequência)    │
#  │               + Segment Embedding     (sentença A ou B)         │
#  └─────────────────────────────────────────────────────────────────┘
#
# Diferentemente do Transformer original (que usa senos/cossenos),
# o BERT *aprende* os positional embeddings durante o treinamento.

class BERTEmbeddings(nn.Module):
    """
    Token Embedding    — mapeia cada ID de token para um vetor denso
    Positional Embedding — codifica a posição (0, 1, 2, …) de cada token
    Segment Embedding  — indica se o token pertence à sentença A (0) ou B (1)
    """

    def __init__(self, tamanho_vocab: int, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.token_emb    = nn.Embedding(tamanho_vocab, d_model, padding_idx=0)
        self.posicao_emb  = nn.Embedding(max_seq_len, d_model)
        self.segmento_emb = nn.Embedding(2, d_model)  # apenas 2 segmentos: A e B
        self.norm         = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)

        # Cria vetor de posições [0, 1, 2, ..., seq_len-1] para cada item do batch
        posicoes = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        emb = (
            self.token_emb(input_ids)       # (batch, seq, d_model)
            + self.posicao_emb(posicoes)    # (1,     seq, d_model) — broadcast
            + self.segmento_emb(segment_ids) # (batch, seq, d_model)
        )
        return self.dropout(self.norm(emb))


# ── 2.2  MULTI-HEAD SELF-ATTENTION ───────────────────────────────────────────
#
# É o coração do Transformer/BERT.
#
# Intuição:
#   Para cada token, a atenção calcula "quanto devo prestar atenção
#   em cada outro token da sequência" — e usa isso para criar uma
#   representação contextualizada.
#
# Matemática:
#   Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
#
#   Onde Q (Query), K (Key) e V (Value) são projeções lineares da entrada.
#   O fator √d_k evita que os produtos internos fiquem muito grandes.
#
# Multi-Head:
#   Em vez de uma atenção única, usamos `n_heads` atenções em paralelo,
#   cada uma focada em diferentes aspectos semânticos.
#   No final, as saídas são concatenadas e projetadas.
#
#   MultiHead(Q, K, V) = Concat(head_1, …, head_h) · W_O
#   onde head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads  # dimensão por cabeça de atenção

        # Projeções lineares para Q, K e V (uma por cabeça, mas implementadas
        # como uma única matriz grande para eficiência)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # projeção de saída

        self.dropout = nn.Dropout(dropout)

    def _dividir_cabecas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape de (batch, seq, d_model)
              para (batch, n_heads, seq, d_head)

        Isso permite calcular a atenção de cada cabeça em paralelo.
        """
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_head)
        return x.transpose(1, 2)  # (batch, n_heads, seq, d_head)

    def _atencao_escalada(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mascara: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Computa Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

        A máscara impede que o modelo "veja" tokens de padding:
        colocamos -inf nas posições mascaradas para que o softmax
        produza probabilidade ≈ 0 nesses lugares.
        """
        escala = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / escala  # (batch, heads, seq, seq)

        if mascara is not None:
            scores = scores.masked_fill(mascara == 0, float("-inf"))

        pesos = F.softmax(scores, dim=-1)   # distribuição de atenção
        pesos = self.dropout(pesos)
        return torch.matmul(pesos, V)        # (batch, heads, seq, d_head)

    def forward(
        self,
        x: torch.Tensor,
        mascara: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x.size(0)

        # 1) Projetar entrada em Q, K, V
        Q = self._dividir_cabecas(self.W_q(x))
        K = self._dividir_cabecas(self.W_k(x))
        V = self._dividir_cabecas(self.W_v(x))

        # 2) Atenção escalada por cabeça
        atencao = self._atencao_escalada(Q, K, V, mascara)

        # 3) Juntar cabeças: (batch, heads, seq, d_head) → (batch, seq, d_model)
        atencao = atencao.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

        # 4) Projeção final
        return self.W_o(atencao)


# ── 2.3  FEED-FORWARD NETWORK (FFN) ──────────────────────────────────────────
#
# Após a atenção, cada token passa individualmente por uma rede densa
# com uma camada oculta 4× maior que d_model.
#
# FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
#
# O BERT original usa GELU (Gaussian Error Linear Unit) em vez de ReLU,
# o que suaviza a não-linearidade e melhora o treinamento.
#
# Função de ativação GELU:
#   GELU(x) ≈ x · σ(1.702 · x)

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.ativacao = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.ativacao(self.linear1(x))))


# ── 2.4  BLOCO ENCODER DO TRANSFORMER ────────────────────────────────────────
#
# Cada bloco combina atenção e FFN com conexões residuais e LayerNorm.
#
# Fluxo de dados:
#   x → [MultiHeadAttention] → + x → LayerNorm → [FFN] → + x → LayerNorm
#       └───────────────────────┘                └────────────────────────┘
#                 conexão residual                      conexão residual
#
# As conexões residuais (add & norm) são cruciais:
#   • Permitem gradientes fluírem diretamente (evita vanishing gradient)
#   • Facilitam o treinamento de redes muito profundas (12–24 camadas no BERT)
#
# O LayerNorm normaliza os ativações para média 0 e variância 1,
# estabilizando o treinamento.

class BlocoEncoder(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.atencao = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn      = FeedForward(d_model, d_ff, dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor | None = None) -> torch.Tensor:
        # Sublayer 1: Self-Attention com conexão residual
        x = self.norm1(x + self.dropout(self.atencao(x, mascara)))

        # Sublayer 2: Feed-Forward com conexão residual
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x


# ── 2.5  MODELO BERT COMPLETO ─────────────────────────────────────────────────
#
# Empilha N blocos encoder sobre as embeddings.
#
# Configurações do paper original:
#   BERT-base:  d_model=768,  n_heads=12, n_layers=12, d_ff=3072  (~110M params)
#   BERT-large: d_model=1024, n_heads=16, n_layers=24, d_ff=4096  (~340M params)
#
# Aqui usamos uma versão "micro" para que rode em qualquer máquina:
#   d_model=128, n_heads=4, n_layers=4, d_ff=512

class BERT(nn.Module):
    """
    Arquitetura BERT completa.

    Saídas:
      sequencia  — representações contextualizadas de TODOS os tokens
                   shape: (batch, seq_len, d_model)
      cls_output — representação do token [CLS] (usada para classificação)
                   shape: (batch, d_model)
    """

    def __init__(
        self,
        tamanho_vocab: int,
        d_model: int    = 128,
        n_heads: int    = 4,
        n_layers: int   = 4,
        d_ff: int       = 512,
        max_seq_len: int = 128,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.embeddings = BERTEmbeddings(tamanho_vocab, d_model, max_seq_len, dropout)
        self.encoder    = nn.ModuleList([
            BlocoEncoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.pool_dense = nn.Linear(d_model, d_model)
        self.pool_ativ  = nn.Tanh()

    def _criar_mascara_padding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Cria máscara booleana: 1 onde o token é real, 0 onde é [PAD].
        Expandida para o formato (batch, 1, 1, seq) para broadcast
        com o tensor de scores de atenção (batch, heads, seq, seq).
        """
        return (input_ids != 0).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        mascara = self._criar_mascara_padding(input_ids)

        # Embeddings somadas: token + posição + segmento
        x = self.embeddings(input_ids, segment_ids)

        # Passa pelos N blocos encoder em sequência
        for bloco in self.encoder:
            x = bloco(x, mascara)

        # Representação de toda a sequência (todos os tokens)
        sequencia = x

        # Representação [CLS]: primeiro token, passado por uma camada densa
        # Esta é a "sentença embedding" usada para classificação
        cls_output = self.pool_ativ(self.pool_dense(x[:, 0, :]))

        return sequencia, cls_output


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 3 — PRÉ-TREINAMENTO (MLM + NSP)
# ─────────────────────────────────────────────────────────────────────────────
#
# O BERT é pré-treinado com DOIS objetivos simultâneos:
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  OBJETIVO 1: Masked Language Modeling (MLM)                              │
# │                                                                          │
# │  15% dos tokens são selecionados aleatoriamente:                         │
# │    - 80% são substituídos por [MASK]                                     │
# │    - 10% são substituídos por um token aleatório                         │
# │    - 10% permanecem iguais                                               │
# │                                                                          │
# │  O modelo deve prever os tokens originais nessas posições.               │
# │  Isso força aprendizado bidirecional — o modelo olha para TODOS          │
# │  os tokens ao redor para adivinhar o que foi mascarado.                  │
# │                                                                          │
# │  OBJETIVO 2: Next Sentence Prediction (NSP)                              │
# │                                                                          │
# │  Dado par (sentença A, sentença B):                                      │
# │    - 50% das vezes B é a sentença real que segue A  → rótulo "IsNext"    │
# │    - 50% das vezes B é uma sentença aleatória       → rótulo "NotNext"   │
# │                                                                          │
# │  O modelo deve classificar se B segue A.                                 │
# │  Isso ensina relações inter-sentença (útil para QA, inferência).         │
# └──────────────────────────────────────────────────────────────────────────┘

class CabecaMLM(nn.Module):
    """
    Cabeça de Masked Language Modeling.

    Pega as representações de saída do BERT e projeta para o tamanho
    do vocabulário, produzindo logits para cada posição mascarada.
    """

    def __init__(self, d_model: int, tamanho_vocab: int):
        super().__init__()
        self.dense  = nn.Linear(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.ativacao = nn.GELU()
        self.projecao = nn.Linear(d_model, tamanho_vocab)

    def forward(self, sequencia: torch.Tensor) -> torch.Tensor:
        x = self.ativacao(self.dense(sequencia))
        x = self.norm(x)
        return self.projecao(x)  # (batch, seq, vocab_size)


class CabecaNSP(nn.Module):
    """
    Cabeça de Next Sentence Prediction.

    Usa a representação [CLS] (que captura o contexto da sequência inteira)
    para classificar se as duas sentenças são consecutivas.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.classificador = nn.Linear(d_model, 2)  # 2 classes: IsNext / NotNext

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        return self.classificador(cls_output)  # (batch, 2)


class BERTPreTreinamento(nn.Module):
    """
    Modelo BERT completo com as duas cabeças de pré-treinamento.
    """

    def __init__(self, bert: BERT, tamanho_vocab: int):
        super().__init__()
        self.bert   = bert
        self.mlm    = CabecaMLM(bert.embeddings.token_emb.embedding_dim, tamanho_vocab)
        self.nsp    = CabecaNSP(bert.embeddings.token_emb.embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequencia, cls_output = self.bert(input_ids, segment_ids)
        logits_mlm = self.mlm(sequencia)   # previsão de tokens mascarados
        logits_nsp = self.nsp(cls_output)  # previsão de próxima sentença
        return logits_mlm, logits_nsp


def aplicar_mascaramento(
    input_ids: torch.Tensor,
    vocab: Vocabulario,
    prob_mascara: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementa o mascaramento do MLM.

    Retorna:
      input_mascarado — sequência com tokens substituídos
      rotulos_mlm     — IDs originais (para calcular o loss)
      posicoes_mask   — máscara booleana das posições modificadas
    """
    input_mascarado = input_ids.clone()
    rotulos_mlm     = torch.full_like(input_ids, -100)  # -100 = ignorado no CrossEntropy
    posicoes_mask   = torch.zeros_like(input_ids, dtype=torch.bool)

    id_pad  = vocab.token2idx.get("[PAD]",  0)
    id_cls  = vocab.token2idx.get("[CLS]",  1)
    id_sep  = vocab.token2idx.get("[SEP]",  2)
    id_mask = vocab.token2idx.get("[MASK]", 4)

    for i in range(input_ids.size(0)):        # para cada item no batch
        for j in range(input_ids.size(1)):    # para cada posição
            token = input_ids[i, j].item()

            # Nunca mascara tokens especiais
            if token in (id_pad, id_cls, id_sep):
                continue

            if random.random() < prob_mascara:
                rotulos_mlm[i, j]    = token  # salva o original para calcular loss
                posicoes_mask[i, j]  = True

                r = random.random()
                if r < 0.80:
                    # 80%: substitui por [MASK]
                    input_mascarado[i, j] = id_mask
                elif r < 0.90:
                    # 10%: substitui por token aleatório do vocabulário
                    input_mascarado[i, j] = random.randint(5, len(vocab) - 1)
                # else: 10%: mantém o token original (não faz nada)

    return input_mascarado, rotulos_mlm, posicoes_mask


class DatasetPreTreinamento(Dataset):
    """
    Dataset para pré-treinamento com MLM e NSP.

    Cria pares de sentenças:
      - 50% são pares consecutivos reais (label NSP = 1)
      - 50% são pares aleatórios         (label NSP = 0)
    """

    def __init__(self, textos: list[str], vocab: Vocabulario, max_len: int = 128):
        self.vocab   = vocab
        self.max_len = max_len
        self.sentencas = self._extrair_sentencas(textos)

    def _extrair_sentencas(self, textos: list[str]) -> list[str]:
        sentencas = []
        for texto in textos:
            # Divide em sentenças por ponto final
            partes = re.split(r'(?<=[.!?])\s+', texto.strip())
            sentencas.extend([p for p in partes if len(p) > 10])
        return sentencas

    def __len__(self):
        return len(self.sentencas) - 1

    def __getitem__(self, idx: int):
        sent_a = self.sentencas[idx]

        # Decide se usa sentença consecutiva (NSP positivo) ou aleatória
        if random.random() < 0.5:
            sent_b   = self.sentencas[idx + 1]
            label_nsp = 1  # IsNext
        else:
            rand_idx  = random.randint(0, len(self.sentencas) - 1)
            sent_b    = self.sentencas[rand_idx]
            label_nsp = 0  # NotNext

        # Tokeniza e trunca para caber em max_len com [CLS] e [SEP]
        tokens_a = self.vocab._tokenizar_basico(sent_a)[: self.max_len // 2 - 2]
        tokens_b = self.vocab._tokenizar_basico(sent_b)[: self.max_len // 2 - 1]

        # Monta sequência: [CLS] A [SEP] B [SEP]
        tokens  = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segmentos = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        # Converte para IDs
        ids  = [self.vocab.token2idx.get(t, self.vocab.token2idx["[UNK]"]) for t in tokens]
        segs = segmentos

        # Padding até max_len
        pad_len = self.max_len - len(ids)
        ids  += [self.vocab.token2idx["[PAD]"]] * pad_len
        segs += [0] * pad_len

        return {
            "input_ids":    torch.tensor(ids,  dtype=torch.long),
            "segment_ids":  torch.tensor(segs, dtype=torch.long),
            "label_nsp":    torch.tensor(label_nsp, dtype=torch.long),
        }


def treinar_pretreinamento(
    modelo: BERTPreTreinamento,
    dataloader: DataLoader,
    vocab: Vocabulario,
    n_epocas: int = 3,
    lr: float     = 1e-4,
):
    """
    Loop de pré-treinamento.

    Loss total = Loss_MLM + Loss_NSP

    Ambos usam CrossEntropyLoss.
    Para MLM, posições com rótulo -100 são automaticamente ignoradas.
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)

    criterio_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    criterio_nsp = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print(f"  PRÉ-TREINAMENTO BERT  |  dispositivo: {dispositivo}")
    print(f"{'='*60}")

    for epoca in range(n_epocas):
        modelo.train()
        loss_total = 0.0

        for batch in dataloader:
            input_ids   = batch["input_ids"].to(dispositivo)
            segment_ids = batch["segment_ids"].to(dispositivo)
            label_nsp   = batch["label_nsp"].to(dispositivo)

            # Aplica mascaramento MLM
            input_mask, rotulos_mlm, _ = aplicar_mascaramento(input_ids, vocab)
            rotulos_mlm = rotulos_mlm.to(dispositivo)

            # Forward pass
            logits_mlm, logits_nsp = modelo(input_mask, segment_ids)

            # Calcula losses
            loss_mlm = criterio_mlm(
                logits_mlm.view(-1, len(vocab)),
                rotulos_mlm.view(-1),
            )
            loss_nsp = criterio_nsp(logits_nsp, label_nsp)
            loss     = loss_mlm + loss_nsp

            # Backward pass
            otimizador.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()

            loss_total += loss.item()

        media = loss_total / len(dataloader)
        print(f"  Época {epoca+1}/{n_epocas}  |  Loss: {media:.4f}")

    print(f"{'='*60}\n")
    return modelo


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 4 — FINE-TUNING PARA ANÁLISE DE SENTIMENTO
# ─────────────────────────────────────────────────────────────────────────────
#
# No fine-tuning:
#  1. Pegamos o BERT pré-treinado (com seus pesos já carregados de conhecimento)
#  2. Adicionamos uma cabeça de classificação simples em cima do [CLS] token
#  3. Treinamos TODA a rede (encoder + cabeça) com uma taxa de aprendizado
#     pequena para não "destruir" o conhecimento pré-treinado
#
# Por que o [CLS]?
#  Durante o pré-treinamento (NSP), o [CLS] aprendeu a agregar informações
#  de toda a sequência. É a representação mais adequada para classificação.
#
# Arquitetura final para análise de sentimento:
#
#  Input → BERT Encoder → [CLS] embedding → Dropout → Linear → Softmax
#                                                               ↓
#                                                    [negativo, neutro, positivo]

class BERTAnalisesSentimento(nn.Module):
    """
    BERT + cabeça de classificação para sentimento.

    Apenas uma camada Linear é adicionada ao BERT base.
    O dropout previne overfitting no fine-tuning (dataset menor).
    """

    def __init__(self, bert: BERT, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.bert     = bert
        self.dropout  = nn.Dropout(dropout)
        d_model       = bert.embeddings.token_emb.embedding_dim
        self.linear   = nn.Linear(d_model, n_classes)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Obtém a representação [CLS] do BERT
        _, cls_output = self.bert(input_ids, segment_ids)

        # Dropout + classificação linear
        x = self.dropout(cls_output)
        return self.linear(x)  # logits: (batch, n_classes)


class DatasetSentimento(Dataset):
    """
    Dataset de análise de sentimento.

    Cada exemplo é um (texto, rótulo) onde rótulo é:
      0 → negativo
      1 → neutro
      2 → positivo
    """

    def __init__(self, dados: list[tuple[str, int]], vocab: Vocabulario, max_len: int = 128):
        self.vocab   = vocab
        self.max_len = max_len
        self.dados   = dados

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx: int):
        texto, rotulo = self.dados[idx]
        ids = self.vocab.codificar(texto, self.max_len)
        return {
            "input_ids": torch.tensor(ids,    dtype=torch.long),
            "rotulo":    torch.tensor(rotulo, dtype=torch.long),
        }


def treinar_fine_tuning(
    modelo: BERTAnalisesSentimento,
    dataloader_treino: DataLoader,
    dataloader_val: DataLoader,
    n_epocas: int = 5,
    lr: float     = 2e-5,  # LR menor que no pré-treino — cuidado com catastrofic forgetting
):
    """
    Loop de fine-tuning para análise de sentimento.

    Boas práticas aplicadas aqui:
      • AdamW com weight decay (L2 regularização)
      • Learning rate warmup (evita destruir pesos pré-treinados logo no início)
      • Gradient clipping (estabilidade numérica)
      • Avaliação em conjunto de validação a cada época
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    criterio   = nn.CrossEntropyLoss()

    # Scheduler: warmup linear nas primeiras 10% das etapas, depois decai
    total_steps = len(dataloader_treino) * n_epocas
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(otimizador, lr_lambda)

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING (Análise de Sentimento)  |  dispositivo: {dispositivo}")
    print(f"{'='*60}")

    historico = {"treino_loss": [], "val_acc": []}

    for epoca in range(n_epocas):
        # ── Treinamento ──────────────────────────────────────────────
        modelo.train()
        loss_total = 0.0

        for batch in dataloader_treino:
            input_ids = batch["input_ids"].to(dispositivo)
            rotulos   = batch["rotulo"].to(dispositivo)

            logits = modelo(input_ids)
            loss   = criterio(logits, rotulos)

            otimizador.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            scheduler.step()

            loss_total += loss.item()

        # ── Validação ────────────────────────────────────────────────
        modelo.eval()
        corretos = 0
        total    = 0

        with torch.no_grad():
            for batch in dataloader_val:
                input_ids = batch["input_ids"].to(dispositivo)
                rotulos   = batch["rotulo"].to(dispositivo)

                logits     = modelo(input_ids)
                predicoes  = logits.argmax(dim=-1)
                corretos  += (predicoes == rotulos).sum().item()
                total     += rotulos.size(0)

        media_loss = loss_total / len(dataloader_treino)
        acuracia   = corretos / total
        historico["treino_loss"].append(media_loss)
        historico["val_acc"].append(acuracia)

        print(f"  Época {epoca+1}/{n_epocas}  |  Loss: {media_loss:.4f}  |  Val Acc: {acuracia:.2%}")

    print(f"{'='*60}\n")
    return modelo, historico


def prever_sentimento(
    modelo: BERTAnalisesSentimento,
    texto: str,
    vocab: Vocabulario,
    max_len: int = 128,
) -> dict:
    """
    Faz inferência em um novo texto.

    Retorna o sentimento previsto e as probabilidades de cada classe.
    """
    rotulos_nome = {0: "NEGATIVO", 1: "NEUTRO", 2: "POSITIVO"}

    dispositivo = next(modelo.parameters()).device
    modelo.eval()

    ids = torch.tensor([vocab.codificar(texto, max_len)], dtype=torch.long).to(dispositivo)

    with torch.no_grad():
        logits = modelo(ids)
        probs  = F.softmax(logits, dim=-1).squeeze()

    classe = probs.argmax().item()
    return {
        "texto":          texto,
        "sentimento":     rotulos_nome[classe],
        "probabilidades": {rotulos_nome[i]: f"{p:.2%}" for i, p in enumerate(probs.tolist())},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 5 — EXEMPLO DE USO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_PRE_TREINO = [
    "O aprendizado de máquina revolucionou a inteligência artificial moderna.",
    "Redes neurais profundas são inspiradas no funcionamento do cérebro humano.",
    "O processamento de linguagem natural permite que computadores entendam texto.",
    "Transformers mudaram completamente a forma de processar sequências de dados.",
    "O BERT usa atenção bidirecional para entender contexto em ambas as direções.",
    "Modelos de linguagem são pré-treinados em grandes quantidades de texto.",
    "O fine-tuning adapta modelos pré-treinados para tarefas específicas.",
    "A atenção multi-cabeça permite capturar diferentes tipos de relações.",
    "Embeddings representam palavras como vetores em espaços de alta dimensão.",
    "Tokenização divide textos em unidades menores chamadas tokens.",
    "O treinamento supervisionado usa exemplos rotulados para aprender padrões.",
    "Gradientes são usados para atualizar os pesos da rede neural.",
    "Regularização previne o overfitting em modelos de aprendizado profundo.",
    "Batch normalization estabiliza o treinamento de redes profundas.",
    "Dropout é uma técnica de regularização que desativa neurônios aleatoriamente.",
    "A função de perda mede a diferença entre previsões e valores reais.",
    "Otimizadores como Adam ajustam a taxa de aprendizado automaticamente.",
    "Dados de validação são usados para avaliar o desempenho do modelo.",
    "O conjunto de teste mede a generalização do modelo para dados novos.",
    "Transfer learning aproveita conhecimento de uma tarefa para outra.",
]

DADOS_SENTIMENTO_TREINO = [
    ("Este produto é absolutamente incrível, amei cada detalhe!", 2),
    ("Ótima experiência, superou todas as minhas expectativas.", 2),
    ("Produto maravilhoso, recomendo para todos os amigos.", 2),
    ("Excelente qualidade, chegou antes do prazo previsto.", 2),
    ("Muito satisfeito com a compra, vale cada centavo.", 2),
    ("Perfeito em todos os aspectos, estou encantado!", 2),
    ("Produto horrível, quebrou no primeiro dia de uso.", 0),
    ("Péssima qualidade, dinheiro completamente desperdiçado.", 0),
    ("Atendimento terrível, nunca mais compro aqui.", 0),
    ("Decepcionante, não funciona como anunciado no site.", 0),
    ("Produto defeituoso, tive que devolver imediatamente.", 0),
    ("Muito ruim, completamente fora do esperado.", 0),
    ("O produto chegou, está funcionando de forma adequada.", 1),
    ("Nem bom nem ruim, cumpre o básico que promete.", 1),
    ("É razoável pelo preço pago, sem grandes surpresas.", 1),
    ("Produto comum, nada de especial mas faz o que deve.", 1),
    ("Entrega normal dentro do prazo, produto conforme descrito.", 1),
]

DADOS_SENTIMENTO_VAL = [
    ("Produto fantástico, estou muito feliz com a compra!", 2),
    ("Nunca mais compro nessa loja, foi uma experiência terrível.", 0),
    ("Produto ok, entrega dentro do prazo.", 1),
    ("Incrível! Melhor compra que já fiz na vida!", 2),
    ("Lixo absoluto, jogou o meu dinheiro fora.", 0),
]

TEXTOS_INFERENCIA = [
    "Que produto maravilhoso, estou completamente apaixonado!",
    "Horrível, nunca foi tão decepcionado na minha vida.",
    "O produto chegou no prazo e funciona como esperado.",
]


def main():
    print("\n" + "━"*60)
    print("  BERT DO ZERO — PYTORCH")
    print("━"*60)

    random.seed(42)
    torch.manual_seed(42)

    # ── ETAPA 1: Construir vocabulário ────────────────────────────────
    print("\n[1/5] Construindo vocabulário...")
    todos_textos = CORPUS_PRE_TREINO + [t for t, _ in DADOS_SENTIMENTO_TREINO + DADOS_SENTIMENTO_VAL]
    vocab = Vocabulario(tamanho_max=2000)
    vocab.construir(todos_textos)
    print(f"     Vocabulário: {len(vocab)} tokens")
    vocab.salvar_json("vocabulario.json")

    # ── ETAPA 2: Criar modelo BERT ─────────────────────────────────────
    print("\n[2/5] Criando arquitetura BERT (versão micro)...")
    bert = BERT(
        tamanho_vocab=len(vocab),
        d_model=128,    # dimensão dos embeddings
        n_heads=4,      # cabeças de atenção
        n_layers=4,     # blocos encoder
        d_ff=512,       # dimensão interna do FFN
        max_seq_len=128,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in bert.parameters())
    print(f"     Parâmetros totais: {n_params:,}")

    # ── ETAPA 3: Pré-treinamento ───────────────────────────────────────
    print("\n[3/5] Pré-treinamento (MLM + NSP)...")
    modelo_pt = BERTPreTreinamento(bert, len(vocab))

    dataset_pt = DatasetPreTreinamento(CORPUS_PRE_TREINO, vocab, max_len=64)
    loader_pt  = DataLoader(dataset_pt, batch_size=4, shuffle=True)

    modelo_pt = treinar_pretreinamento(modelo_pt, loader_pt, vocab, n_epocas=3)

    # ── ETAPA 4: Fine-tuning ───────────────────────────────────────────
    print("\n[4/5] Fine-tuning para Análise de Sentimento...")
    modelo_sa = BERTAnalisesSentimento(modelo_pt.bert, n_classes=3, dropout=0.3)

    dataset_treino = DatasetSentimento(DADOS_SENTIMENTO_TREINO, vocab, max_len=64)
    dataset_val    = DatasetSentimento(DADOS_SENTIMENTO_VAL,    vocab, max_len=64)

    loader_treino = DataLoader(dataset_treino, batch_size=4, shuffle=True)
    loader_val    = DataLoader(dataset_val,    batch_size=4)

    modelo_sa, historico = treinar_fine_tuning(
        modelo_sa, loader_treino, loader_val, n_epocas=5
    )

    # ── ETAPA 5: Inferência ────────────────────────────────────────────
    print("\n[5/5] Inferência em novos textos:")
    print("─" * 60)

    for texto in TEXTOS_INFERENCIA:
        resultado = prever_sentimento(modelo_sa, texto, vocab)
        print(f"\n  Texto:      \"{resultado['texto']}\"")
        print(f"  Sentimento: {resultado['sentimento']}")
        print(f"  Probabilidades:")
        for sent, prob in resultado["probabilidades"].items():
            print(f"    {sent:10s}: {prob}")

    print("\n" + "━"*60)
    print("  Concluído!")
    print("━"*60 + "\n")


if __name__ == "__main__":
    main()
