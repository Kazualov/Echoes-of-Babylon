Validation: 156 shared documents (data/val_ids.csv). Metric: BLEU=sacrebleu-2.x-13a-exp | chrF++=sacrebleu-2.x-c6-w2-b2 | geo_mean=sqrt(BLEU*chrF++). Bootstrap: 5000 document resamples, seed 12345.

| System | Seeds | BLEU | chrF++ | GM (mean ± std) | GM 95% CI |
|---|---:|---:|---:|---:|---:|
| mbart50 | 1 | 33.40 | 53.93 | 42.44 | [39.73, 44.93] |
| mbr3-ensemble | 1 | 30.26 | 52.31 | 39.79 | [37.11, 42.33] |
| qwen2.5-7b-lora | 1 | 25.73 | 47.56 | 34.98 | [32.33, 37.55] |
| byt5 | 1 | 13.64 | 31.12 | 20.60 | [18.68, 22.77] |
| byt5-mbr | 1 | 10.35 | 28.55 | 17.19 | [15.44, 19.15] |
| byt5-member2 | 1 | 10.23 | 26.12 | 16.35 | [14.63, 18.34] |

Paired bootstrap vs. best system (mbart50):

| Other | GM diff | 95% CI | p (best not better) |
|---|---:|---:|---:|
| mbr3-ensemble | 2.61 | [0.84, 4.71] | 0.000 |
| qwen2.5-7b-lora | 7.35 | [5.52, 9.07] | 0.000 |
| byt5 | 21.70 | [19.03, 24.21] | 0.000 |
| byt5-mbr | 25.13 | [22.73, 27.44] | 0.000 |
| byt5-member2 | 25.97 | [23.61, 28.22] | 0.000 |
