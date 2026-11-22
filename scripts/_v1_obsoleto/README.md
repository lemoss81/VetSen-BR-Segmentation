# Scripts do protocolo v1.0 (OBSOLETOS - nao rodar)

Estes scripts pertencem ao protocolo anterior, que tinha `classes: 6` com
`ignore_index: 0`, encoder treinado do zero, ReduceLROnPlateau com early stopping
e checkpoint pela loss. Rodar qualquer um deles hoje produz runs incompativeis
com os configs atuais e com `scripts/eval_test.py` (que recusa os nomes `r1_*`/`r2_*`).

Ficam aqui so como referencia do que foi executado ate a tag `v1.0-baseline`.

Substitutos: `scripts/screen_lr.sh` (screening) e `scripts/smoke_check.py` (checagem previa).
