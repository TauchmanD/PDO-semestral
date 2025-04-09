# TULeCZech

Jazykový model urřený pro vektorizaci textu v českém jazyce. Je použitelný v aplikacích jako je hledání pomocí similarity nebo RAG.

Postavený na architektuře xlm-RoBERTa-base. Pro více informací o architektuře a struktuře modelu klikněte [zde](./MODEL.md).

## Získání modelu

Model lze najít na webovém portálu [HuggingFace](https://huggingface.co/). Lze ho získat těmito způsoby:

 - V záložce ``Files and versions`` - manuální stažení a použití
 - Pomocí knihoven ``transformers`` nebo ``sentence-transformers`` - automatické stažení a použití *(preferováno)*


## Použití modelu
Níže je ukázka jak enkódovat dotazy pro nalezení nejbližších dokumentů s použitím knihovny ``transformers`` a průměrováním vektorů pro získání výsledného vektoru.

```python
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Každý vstupní text by měl začínat s "query: " nebo "passage: " pro všechny jazyky.
# Pro použití jiné funkce než vrácení nejbližšího dokumentu použije pouze předpis "query: ".
input_texts = [
    'Kolik bílkovin by měla žena jíst',
    '南瓜的家常做法',
    "Obecně platí, že průměrný požadavek na příjem bílkovin podle CDC (Centra pro kontrolu a prevenci nemocí) pro ženy ve věku od 19 do 70 let je 46 gramů denně. Jak ale můžete vidět v této tabulce, bude třeba tento příjem zvýšit, pokud jste těhotná nebo trénujete na maraton. Podívejte se na tabulku níže, abyste zjistili, kolik bílkovin byste měla denně konzumovat.",
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右, 放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]


tokenizer = AutoTokenizer.from_pretrained('tauchmand/tuleczech')
model = AutoModel.from_pretrained('tauchmand/tuleczech')

# Tokenizace vstupního textu
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# Normalizace embeddingů
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
```

## Podpora Sentence Transformers

Model podporuje použití za pomocí knihovny ``sentence-transformers`` následovně

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('tauchmand/tuleczech')

input_texts = [
    'Kolik bílkovin by měla žena jíst',
    '南瓜的家常做法',
    "Obecně platí, že průměrný požadavek na příjem bílkovin podle CDC (Centra pro kontrolu a prevenci nemocí) pro ženy ve věku od 19 do 70 let je 46 gramů denně. Jak ale můžete vidět v této tabulce, bude třeba tento příjem zvýšit, pokud jste těhotná nebo trénujete na maraton. Podívejte se na tabulku níže, abyste zjistili, kolik bílkovin byste měla denně konzumovat.",
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右, 放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]

embeddings = model.encode(input_texts, normalize_embeddings=True)

```

Požadované závislosti  
```pip install sentence_transformers~=2.2.2```


## Podpora jazyků

Tento model ja založený na xlm-RoBERTa-base, který v základu podporuje více než 100 různůch jazyků. Je ovšem dotrénovaný pouze na českých datech, proto je podpora omezená pouze pro český jazyk.


## Limitace

Delší texty budou zkráceny na maximálně 512 tokenů


## FAQ

1. **Potřebuji používat prefix "query" a "passage:" pro vstupní texty?**  
Ne, jelikož je základní model dotrénovaný bez těchto prefixů, není nutné je používat.
2. **Proč je skóre similarity často kolem 0.3 až 0.5**  
Je to dáno ztrátovou funkcí, která byla využita pro doladění modelu - MultipleNegativesRankingLoss  
Ovšem pro potřeby similarity záleží spíše na pořadí než na samotné hodnotě skore

## Citace

Pokud vám článek nebo model pomohou, zvažte prosím následující citace:
```tex
@article{tulezech,
    title={Tvorba jazykového modelu},
    author={Tauchman Denis, Martin Poláček},
    journal={...},
    year={2025}
}
```