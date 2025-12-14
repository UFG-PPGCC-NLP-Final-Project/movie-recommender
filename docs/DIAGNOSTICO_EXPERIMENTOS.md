# Diagnóstico Completo - Experimentos 1-4

## 📊 Resumo Executivo

**Problema identificado**: Todos os 4 experimentos com Trainer original convergiram para **nDCG@10 ≈ 0.044**, que é **74% abaixo** do esperado (artigo reporta 0.13-0.17).

**Causa raiz**: 
1. `BCEWithLogitsLoss()` sem pesos não consegue lidar com desbalanceamento extremo (1:1200)
2. Multi-task learning com pesos iguais causa domínio da tarefa auxiliar (CE loss 2000x maior que BCE loss)

**Solução implementada**: `WeightedTrainer` com:
- ✅ Weighted BCE Loss (`pos_weight=1200`)
- ✅ Multi-task balanceado (`alpha=0.001` para CE loss)
- ✅ Monitoramento detalhado de losses separadas

---

## 🔬 Análise Detalhada dos Experimentos

### Experimento 1: SBERT Baseline (sem RNN, sem multi-task)

**Configuração:**
- Modelo: SBERT (all-MiniLM-L6-v2) + FFN
- Loss: `BCEWithLogitsLoss()` sem pesos
- Hyperparâmetros: lr=2e-5, dropout=0.1, batch_size=32

**Resultados (20 épocas):**
```
Train Loss: 0.0035
Eval Loss: 0.0035
nDCG@10: 0.0440
Recall@10: 0.0594
```

**Diagnóstico:**
- Loss muito baixa (0.0035) indica que modelo "aprendeu" a predizer ~0 para tudo
- Com 6924 classes e média de 5 labels positivos por exemplo, predizer 0 minimiza BCE loss
- nDCG@10 = 0.0440 mostra que recomendações são praticamente aleatórias
- **Conclusão**: BCE loss sem pesos não funciona para extreme multi-label imbalance

---

### Experimento 2: SBERT + RNN (features colaborativas)

**Configuração:**
- Adiciona: GRU bidirectional para processar filmes mencionados
- Embedding dimension: 256 → RNN hidden: 128

**Resultados (20 épocas):**
```
Train Loss: 0.0035
Eval Loss: 0.0035
nDCG@10: 0.0459 (+0.0019 vs baseline)
Recall@10: 0.0683 (+0.0089 vs baseline)
```

**Diagnóstico:**
- Ganho mínimo: +0.0019 nDCG@10 (esperado: +0.035 segundo artigo)
- RNN está funcionando (Recall@10 melhorou 15%), mas BCE loss inadequado impede otimização
- Features colaborativas são aprendidas, mas não conseguem ser aproveitadas
- **Conclusão**: RNN não é o problema, a loss function é o gargalo

---

### Experimento 3: SBERT + Multi-Task (user tags)

**Configuração:**
- Tarefa principal: BCE para recomendar filmes (6924 classes)
- Tarefa auxiliar: CE para predizer filme a partir de tag (6924 classes)
- Peso: `loss = bce_loss + ce_loss` (igual)

**Resultados (20 épocas):**
```
Train Loss: 6.2071 ⚠️
Eval Loss: 0.0035
nDCG@10: 0.0443 (-0.0003 vs baseline!)
Recall@10: 0.0594
```

**Diagnóstico CRÍTICO:**
- Train Loss = 6.2071 enquanto Eval Loss = 0.0035 → problema grave
- Decomposição estimada:
  * BCE loss (filmes) ≈ 0.003
  * CE loss (tags) ≈ 6.204
  * Total = 6.207
- Gradiente é 99.95% dominado pela tarefa de tags
- Modelo otimiza predição de tags, ignora completamente recomendação de filmes
- **Conclusão**: Multi-task sem balanceamento é contraproducente

**Por que CE loss é tão maior?**
```python
# BCE Loss (multi-label):
# - 5 positivos, 6919 negativos por exemplo
# - Loss média ≈ 0.003 (modelo "acerta" ao predizer zero)

# CE Loss (single-label):
# - 1 correto, 6923 incorretos
# - Loss média ≈ ln(6924) ≈ 8.8 no início
# - Converge para ≈ 6.2 (ainda alto pois tarefa é difícil)

# Resultado: 6.2 / 0.003 = 2067x diferença!
```

---

### Experimento 4: SBERT + RNN + Multi-Task (modelo completo)

**Configuração:**
- Combina RNN (Exp 2) + Multi-Task (Exp 3)
- Todas as features ativadas

**Resultados (20 épocas):**
```
Train Loss: 6.2516 ⚠️
Eval Loss: 0.0035
nDCG@10: 0.0462 (+0.0003 vs Exp 3, +0.0022 vs baseline)
Recall@10: 0.0646
```

**Diagnóstico:**
- Mesmo problema do Exp 3: CE loss domina
- RNN adiciona pequeno ganho (+0.0003), mas é negligível
- Train Loss ainda maior (6.2516 vs 6.2071)
- **Conclusão**: Modelo "completo" herda todos os problemas

---

## 📈 Comparação com Artigo Original

| Configuração | Artigo (BERT) | Experimentos (SBERT) | Diferença |
|--------------|---------------|----------------------|-----------|
| Baseline | 0.130 | 0.0440 | **-66%** |
| + RNN | 0.165 | 0.0459 | **-72%** |
| + Multi-Task | 0.138 | 0.0443 | **-68%** |
| + RNN + Multi-Task | 0.169 | 0.0462 | **-73%** |

**Observações:**
- Diferença não é SBERT vs BERT (ambos são sentence encoders eficazes)
- Diferença é loss function: artigo provavelmente usou weighted BCE ou focal loss
- Multi-task no artigo melhorou +0.008, nos experimentos piorou -0.003

---

## 🛠️ Solução: WeightedTrainer

### Mudança 1: Weighted BCE Loss

**Antes (linha 814):**
```python
self.bce_loss = nn.BCEWithLogitsLoss()  # ❌ Sem pesos
```

**Depois:**
```python
# Calcular desbalanceamento real
avg_labels = sum(len(d['recommended_movies']) for d in train_data) / len(train_data)
pos_weight_value = config.num_movies / avg_labels  # ≈ 1200

# Aplicar peso para classes positivas
pos_weight = torch.full([config.num_movies], pos_weight_value, device=device)
self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # ✅ Balanceado
```

**Efeito esperado:**
- Loss para exemplos positivos é amplificada 1200x
- Modelo é forçado a aprender padrões (não pode mais "colar" predizendo zero)
- nDCG@10 deve subir de 0.044 para > 0.08 em 20 épocas

---

### Mudança 2: Multi-Task Balanceado

**Antes (linha 888):**
```python
loss = bce_loss + ce_loss  # ❌ Pesos iguais (0.003 + 6.2 = 6.203)
```

**Depois:**
```python
alpha = 0.001  # Fator de balanceamento
loss = bce_loss + (alpha * ce_loss)  # ✅ Balanceado (0.003 + 0.006 = 0.009)
```

**Matemática:**
```
Sem balanceamento:
- BCE: 0.003 (0.05% do gradiente)
- CE: 6.200 (99.95% do gradiente)

Com alpha=0.001:
- BCE: 0.003 (33% do gradiente)
- CE: 0.006 (67% do gradiente)
```

**Efeito esperado:**
- Ambas tarefas contribuem significativamente
- Multi-task deve melhorar nDCG@10 em +0.008 (como no artigo)

---

## 🎯 Validação da Solução

### Teste 1: Baseline com Weighted BCE (5 épocas)

**Objetivo**: Validar que weighted BCE resolve problema principal

**Critério de sucesso**: nDCG@10 > 0.06 em 5 épocas

**Como executar**:
```python
weighted_baseline_trainer.train(num_epochs=5)
```

---

### Teste 2: Modelo Completo (20 épocas)

**Objetivo**: Validar solução completa

**Critério de sucesso**: nDCG@10 > 0.10 em 20 épocas

**Como executar**:
```python
weighted_full_trainer.train(num_epochs=20)
```

---

## 🔄 Próximas Iterações (Se Necessário)

### Se nDCG@10 < 0.08 após 20 épocas com Weighted BCE:

**Opção 1: Ajustar pos_weight**
```python
# Aumentar ainda mais o peso para positivos
pos_weight_value = config.num_movies / avg_labels * 1.5  # 1800 ao invés de 1200
```

**Opção 2: Implementar Focal Loss**
```python
class FocalLoss(nn.Module):
    """
    Focal Loss: foca em exemplos difíceis
    Referência: Lin et al. (2017) - Focal Loss for Dense Object Detection
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probabilidade de acerto
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
```

**Opção 3: Class-Balanced Loss**
```python
# Calcular peso baseado em frequência efetiva
def get_cb_weight(num_samples_per_class, beta=0.9999):
    effective_num = 1.0 - np.power(beta, num_samples_per_class)
    weights = (1.0 - beta) / effective_num
    return weights / weights.sum() * len(weights)
```

---

## 📚 Referências

### Loss Functions para Extreme Imbalance:

1. **Focal Loss**: Lin et al. (2017) - "Focal Loss for Dense Object Detection"
   - Foca em exemplos difíceis, reduz peso de exemplos fáceis
   - `loss = -alpha * (1-pt)^gamma * log(pt)`

2. **Class-Balanced Loss**: Cui et al. (2019) - "Class-Balanced Loss Based on Effective Number of Samples"
   - Usa frequência efetiva ao invés de frequência absoluta
   - Funciona bem em long-tail distributions

3. **LDAM Loss**: Cao et al. (2019) - "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
   - Adiciona margem proporcional à frequência da classe
   - Combina com deferred re-weighting schedule

### Multi-Task Learning Balancing:

1. **Uncertainty Weighting**: Kendall et al. (2018) - "Multi-Task Learning Using Uncertainty to Weigh Losses"
   - Aprende pesos automaticamente baseado em incerteza
   - `loss = sum(1/(2*sigma^2) * task_loss + log(sigma))`

2. **GradNorm**: Chen et al. (2018) - "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
   - Balanceia gradientes ao invés de losses
   - Mantém taxas de treinamento uniformes entre tarefas

3. **Dynamic Weight Average**: Liu et al. (2019)
   - Ajusta pesos baseado em taxa de mudança das losses
   - `w_t = w_{t-1} * exp(rate * delta_loss)`

---

## 💡 Lições Aprendidas

### 1. Class Imbalance é o problema #1
- 1:1200 ratio requer tratamento especial
- BCE sem pesos = modelo aprende a predizer zero
- Weighted BCE ou Focal Loss são essenciais

### 2. Multi-Task Learning requer balanceamento
- Losses de diferentes tarefas têm escalas diferentes
- Peso igual ≠ contribuição igual
- Sempre monitorar losses separadamente

### 3. Métricas enganam
- Loss baixa (0.0035) ≠ modelo bom
- Modelo pode "trapacear" predizendo sempre a classe majoritária
- nDCG@10 é métrica verdadeira de sucesso

### 4. Arquitetura não é o problema
- SBERT vs BERT: diferença mínima
- RNN funciona, mas loss inadequado impede aproveitamento
- Foco em loss function > foco em arquitetura

### 5. Debugging sistemático é crucial
- Monitorar losses separadamente (BCE vs CE)
- Calcular desbalanceamento real do dataset
- Comparar Train Loss vs Eval Loss para detectar anomalias
