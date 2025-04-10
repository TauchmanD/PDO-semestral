# TULeCZech

**TULeCZech** je embedovací jazykový model určený pro český jazyk. Je navržen pro úlohy jako sémantické vyhledávání, vyhledávání podle podobnosti (similarity search), nebo jako součást systémů Retrieval-Augmented Generation (RAG).

Model je postaven na architektuře **xlm-RoBERTa-base** a doladěn (fine-tuned) výhradně na českých datech.

📄 Pro detailní informace o trénování, architektuře a použitých datech klikněte [zde](./MODEL.md).

---

## 🧠 Získání modelu

Model je dostupný na [HuggingFace Hubu](https://huggingface.co/tauchmand/tuleczech) a lze ho využít několika způsoby:

- **Manuální stažení** ze záložky `Files and versions`
- **Automatické načtení pomocí knihoven** `transformers` nebo `sentence-transformers` *(doporučeno)*

---

## 🚀 Rychlý start: použití s `transformers`

```python
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = [
    "query: Kolik bílkovin by měla žena jíst",
    "query: 南瓜的家常做法",
    "passage: Obecně platí, že průměrný požadavek na příjem bílkovin ...",
    "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 ..."
]

tokenizer = AutoTokenizer.from_pretrained("tauchmand/tuleczech")
model = AutoModel.from_pretrained("tauchmand/tuleczech")

batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
embeddings = F.normalize(embeddings, p=2, dim=1)

scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
```

---

## ✨ Alternativa: použití se `sentence-transformers`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("tauchmand/tuleczech")

input_texts = [
    "Kolik bílkovin by měla žena jíst",
    "南瓜的家常做法",
    "Obecně platí, že průměrný požadavek na příjem bílkovin ...",
    "1.清炒南瓜丝 原料:嫩南瓜半个 ..."
]

embeddings = model.encode(input_texts, normalize_embeddings=True)
```

> 📦 Požadované závislosti:  
> ```bash
> pip install sentence_transformers~=2.2.2
> ```

---

## 🌍 Jazyková podpora

Model je založen na **xlm-RoBERTa-base**, který podporuje více než 100 jazyků. Vzhledem k doladění výhradně na českých datech je **optimalizován pouze pro češtinu**.

---

## 📊 Výsledky na SQuAD

| **model**                                | **základní model**       | **acc@1 [%]** | **acc@3 [%]** | **acc@10 [%]** |
|------------------------------------------|---------------------------|---------------|---------------|----------------|
| **TULeCZech (náš model)**                | **xlm-RoBERTa-base**      | **63.7**      | **83.6**      | **92.1**       |
| Multilingual-E5-base                     | xlm-RoBERTa-base          | 65.4          | 86.8          | 91.7           |
| Multilingual-E5-small                    | xlm-RoBERTa-small         | 63.0          | 85.0          | 90.5           |
| distiluse-base-multilingual-cased-v2     | BERT                      | 42.6          | 68.5          | 77.3           |
| LaBSE                                    | BERT                      | 42.3          | 68.2          | 76.9           |
| Seznam/simcse-dist-mpnet-czeng-cs-en     | BERT                      | 36.1          | 62.1          | 71.8           |
| Seznam/RetroMAE-small-cs                 | BERT                      | 27.0          | 49.9          | 59.9           |
| Seznam/simcse-small-e-czech              | ELECTRA                   | 3.3           | 9.9           | 14.8           |
---

## 🦚 Trénovací datasety

| **Dataset**     | **Počet dvojic [tis.]** |
|-----------------|--------------------------|
| DaReCzech       | 97                       |
| SQuAD v1.1 (cz) | 18.5                     |
| SQuAD v2.0 (cz) | 18.6                     |
| iDnes           | 500                      |
| **Celkem**      | **634.3**                |


- Pro další podrobnosti přejděte na [MODEL.md](./MODEL.md)

---

## ⚠️ Limitace

- Dlouhé vstupy jsou oříznuty na **maximálně 512 tokenů**
- Model je optimalizovaný pouze pro češtinu

---

## ❓ FAQ

**1. Musím používat prefixy `query:` a `passage:`?**  
Ne, model funguje i bez nich. Prefixy ale mohou mírně zlepšit výkon ve specifických úlohách.

**2. Proč je skóre podobnosti většinou mezi 0.3–0.5?**  
Je to dáno použitou ztrátovou funkcí (MultipleNegativesRankingLoss). Absolutní hodnota skóre není důležitá – klíčové je **pořadí výsledků**.

---

## 📚 Citace

Pokud tento model nebo projekt využíváte, zvažte prosím uvedení následující citace:

```bibtex
@article{tulezech,
    title={Tvorba jazykového modelu},
    author={Tauchman Denis, Martin Poláček},
    journal={...},
    year={2025}
}
