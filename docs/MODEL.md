# TULeCZech

Jazykový model pro vektorizaci textu v českém jazyce.

## Použité metriky
Pro vyhodnocování jazykového modelu se využívala metrika TOP1, TOP3 a TOP5 neboli acc@1, acc@3 a acc@10. Tato metrika udává šanci, že v TOPx nejbližších vektorech se nachází ten správný.

## Základním model
Výběr základního modelu se odvíjel od vyhodnocení aktuálně dostupných modelů. Vzalo se tak více open source modelů, kde byly zastoupené jednotlivé základní modely - RoBERTa, BERT a ELECTRA.

Následující tabulka ukazuje vyhodnocení modelu za pomocí evaluační sady SQuAD:

## Datasety
Jelikož neexistuje žádný dataset, který by byl vhodný pro trénování tohoto modelu, byl vytvořen vlastní dataset z existujících

### SQuAD
První použitý dataset je SQuAD přeložený do českého jazyka. Tento dataset je určený primárně pro problém QA (Question Answering), ale byl transformován do podoby korpus - otázka.


Z tohoto datasetu bylo použito přes 30 tisíc dvojic

### DaReCZech
Dalším datasetem je DaReCZech. Tento dataset je od společnosti seznam a slouží k hodnocení relevance textu.
Z datasetu se vzali dvojice query a vrácený dokument - využila se pouze menší část datasetu obsahující 90 tisíc dvojic

### iDnes

Posledním datasetem je vlastně vytvořený dataset iDnes. Vzniknul pomocí webscrapingu webového portálu idnes a stahováním nadpisu a anotace ze článků.

Celkově se stáhnulo přes 500 tisíc článků z více jak 3 kategorií


## Hyperparamatry


## Výsledky modelu