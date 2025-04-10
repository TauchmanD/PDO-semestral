# TULeCZech

Jazykový model pro vektorizaci textu v českém jazyce, vytvořený v rámci bakalářské práce na TUL. Projekt je zaměřený na trénování embedovacího modelu nad českými daty, aby dokázal efektivně reprezentovat význam textu v podobě vektorů. 

## Cíl projektu

Cílem bylo vytvořit embedovací model optimalizovaný pro český jazyk, který bude možné využít při vyhledávání relevantních odpovědí, porovnávání významové podobnosti. Výsledný model je založen na architektuře **xlm-RoBERTa-base** a prošel tréninkem na českých datech.

## Použité metriky

Pro vyhodnocování jazykového modelu se používala metrika **TOP1, TOP3 a TOP5**, označovaná také jako **acc@1, acc@3 a acc@10**. Metrika vyjadřuje pravděpodobnost, že ve vektorech s nejbližší vzdáleností se nachází správná odpověď.

## Výběr základního modelu

Výběr základního modelu probíhal porovnáním dostupných open source embedovacích modelů. Zastoupeny byly modely postavené na architekturách **RoBERTa**, **BERT** a **ELECTRA**.

| **model**                                | **základní model**       | **acc@1 [%]** | **acc@3 [%]** | **acc@10 [%]** |
|------------------------------------------|---------------------------|---------------|---------------|----------------|
| Multilingual-E5-base                     | **xlm-RoBERTa-base**      | **65.4**      | **86.8**      | **91.7**       |
| Multilingual-E5-small                    | xlm-RoBERTa-small         | 63.0          | 85.0          | 90.5           |
| distiluse-base-multilingual-cased-v2     | BERT                      | 42.6          | 68.5          | 77.3           |
| LaBSE                                    | BERT                      | 42.3          | 68.2          | 76.9           |
| Seznam/simcse-dist-mpnet-czeng-cs-en     | BERT                      | 36.1          | 62.1          | 71.8           |
| Seznam/RetroMAE-small-cs                 | BERT                      | 27.0          | 49.9          | 59.9           |
| Seznam/simcse-small-e-czech              | ELECTRA                   | 3.3           | 9.9           | 14.8           |

## Použité datasety

Protože neexistuje veřejně dostupný dataset vhodný pro tento typ tréninku v češtině, byly vytvořeny a upraveny následující datasety:

### SQuAD (cz)

Přeložená verze anglického SQuAD datasetu. Ačkoliv původně určený pro **Question Answering**, byl dataset upraven do podoby dvojic **(korpus, otázka)** pro potřeby tréninku embedovacího modelu. Použito přes **30 000 dvojic**.

### DaReCZech

Dataset od společnosti Seznam.cz, určený pro hodnocení relevance výsledků vyhledávání. Využity byly dvojice **(dotaz, dokument)**. Použita byla menší část obsahující cca **90 000 dvojic**.

### iDnes dataset

Vlastní dataset vytvořený web scrapingem z portálu **iDnes.cz**. Extrahovány byly nadpisy a anotace článků z více než **3 kategorií**. Celkem bylo získáno přes **500 000 článků**. Texty byly očištěny a zpracovány pro následné použití při tréninku.

## Architektura a trénink

Model je postavený na **xlm-RoBERTa-base** a byl doladěn (fine-tuned) pomocí výše uvedených dat. Trénink probíhal jako úloha podobnosti dvojic textů (sentence similarity).

## Hyperparametry

Nastavení tréninku:

- Maximální délka vstupu: 512 tokenů
- Batch size: 56
- Optimalizátor: Adam + AdamW
- Počet epoch: 30
- Learning rate: 2e-5
- Loss funkce: MultipleNegativesRankingLoss

## Výsledky modelu

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


Vytrénovaný model dosahuje velmi dobrých výsledků ve srovnání s dostupnými embedovacími modely. Model založený na **xlm-RoBERTa-base** byl vyhodnocen jako nejlepší volba pro české embedování.

## Možné využití

Model lze použít například pro:

- Vyhledávání podobných dokumentů
- Sémantické porovnávání dotazů a odpovědí
- Srovnání významové podobnosti

## Závěr

Tento projekt přináší nový embedovací model pro český jazyk, trénovaný na kombinaci veřejně dostupných a vlastních dat. Výsledky ukazují, že model založený na **xlm-RoBERTa-base** výrazně překonává jiné přístupy v českém prostředí.
