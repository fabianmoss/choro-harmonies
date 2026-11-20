[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fabianmoss/choro-harmonies)

### Proposed repo structure:

    scripts
        create_h5_data.py
        train.py
        annotate.py
        write_to_musicxml.py
        csv_from_xml.ipynb
        remove_existing_chords.ipynb
        chord-eval-script.py
        imported chord_eval scripts (Currently in chord_eval/scripts)
        + ...

    data
        choro_corpus
            contains_chords/
            contains_chords_csv/ (currently in chord_eval/originals)
            no_chords/ 
            removed_chords/
            keys_choro_pieces.csv

        h5_data/
        h5_ksm/

    models
        ft (checkpoints/...)
            ccm/...
            csm/...
            ctm/...
            ksm/...
            ktm/...

        non-ft (checkpoints-beat-best/...)
            ccm/...
            csm/...
            ...


    outputs
        inference
            subdirs containing annotate.py output tsvs for the used models
            e.g. non-ft-no-cpm (currently outputs/...)

        playground (model xml outputs) (currently playground/)

        playground_csv (model xml mapped to csv outputs, needed for eval script) (currently choro_model_ft/...)




    eval
        originals/ (currently in chord_eval/originals)
        predictions/ (currently in chord_eval/predictions)
        results/ (currently in chord_eval/results)

    misc
        flowcharts/
        ...


Unclear:
- where to put fundamental inference scripts (note.py, piece.py, etc.)

---

### h5_data seeds

seed0: created 31.10.2026 on 26 scores `h5_data`, 25 scores `h5_ksm`

### flowcharts order

create_h5_data.py => train.py => annotate.py => write_to_musicxml.py

---

training splits default (80%train,10%val,10%test)
## Models trained on seed0:
(train_batch, valid_batch, chunk_size)
### CCM
- version_0 batches = (16,32,64), lr = 0.001, patience=35epochs (46.4%)
- version_1 batches = (16,32,64), lr = 0.005, patience=35epochs (46.1%)
- version_2 batches = (16,32,64), lr = 0.005, patience=20epochs (45.9%)
- version_3 batches = (8,16,32) , lr = 0.005, patience=20epochs (39.9%)
- version_4 batches = (8,16,32) , lr = 0.01,  patience=20epochs (18.6%)
- version_5 batches = (32,64,128) lr = 0.005, patience=20epochs (44.3%)
- *version_6 batches = (16,32,64), lr = 0.001, patience=20epochs (50.4%)*

### CTM
- *version_0 batches = (8,64,64), lr = 0.001, patience=20epochs (98.0%)*
- version_1 batches = (4,32,32), lr = 0.001, patience=20epochs (97.9%)


### CSM
- version_0 batches = (32,64,256), lr = 0.001, patience=20epochs (23.6%)
- version_1 batches = (16,32,128), lr = 0.001, patience=20epochs (23.6%)
- version_2 batches = (8,16,64), lr = 0.001, patience=20epochs   (55.9%)
- *version_3 batches = (8,16,64), lr = 0.005, patience=50epochs   (60.0%)*
- version_4 batches = (8,16,64), lr = 0.005, patience=20epochs   (57.7%)
- version_5 batches = (8,16,64), lr = 0.01,  patience=20epochs   (57.5%)
- version_6 batches = (8,16,64), lr = 0.01,  patience=40epochs   (56.8%)

### KSM
- version_0 batches = (32,64,256), lr = 0.001, patience=20epochs (25.0%)
- version_1 batches = (32,64,256), lr = 0.01, patience=20epochs (75.0%)
- *version_2 batches = (32,64,256), lr = 0.01, patience=35epochs (100%)*

### KTM
- *version_0 batches = (32,64,256), lr = 0.001, patience=20epochs (96.5%)*

---

## seed1 19.11.26
26 pieces, created for retraining CCM and CTM

### Splits

=== TRAIN (21 pieces) ===
  [ 2] score_11265-Por_um_beijo_Terna_saudade-Anacleto_de_Medeiros.xml
  [20] score_4990-Menino_de_Ouro-Ernesto_Nazareth.xml
  [14] score_280-Roceira-Mario_Alvares.xml
  [17] score_4566-Isto_nao_e_vida-Macarico.xml
  [ 5] score_1235-Valsa-Joaquim_Callado.xml
  [11] score_1985-Tira_Poeira-Satyro_Bilhar.xml
  [22] score_5602-Flor_do_Abacate-Alvaro_Sandim.xml
  [13] score_265-Paixao_encoberta-Mario_Alvares.xml
  [18] score_4637-Desmantelando_Relogios-Cicero_Telles_de_Meneses.xml
  [19] score_4800-Atlantico-Ernesto_Nazareth.xml
  [16] score_3791-Plangente-Chiquinha_Gonzaga.xml
  [ 1] score_11143-Gemea-Chiquinha_Gonzaga.xml
  [10] score_1932-Ondina-Eurico_Batista.xml
  [25] score_846-Conceicao-Joaquim_Callado.xml
  [24] score_6879-Tema_de_apresentacao_da_V_Sinfonia-Beethoven.xml
  [ 8] score_1425-Terna_Saudade-Anacleto_de_Medeiros.xml
  [ 6] score_1324-Em_ti_Pensando-Anacleto_de_Medeiros.xml
  [ 4] score_12248-Gloria-Bonfiglio_de_Oliveira.xml
  [ 9] score_192-Qualquer_coisa-Irineu_de_Almeida.xml
  [ 7] score_1361-Medrosa-Anacleto_de_Medeiros.xml
  [23] score_5789-Tristeza-Eduardo_Souto.xml

=== VALID (2 pieces) ===
  [ 3] score_117-Bohemia_Terra-Irineu_de_Almeida.xml
  [ 0] score_10805-Meu_sabia-Raul_Silva.xml

=== TEST (3 pieces) ===
  [21] score_5063-Pinguim-Ernesto_Nazareth.xml
  [15] score_2802-Belezas_do_Recife-Misael_Domingues.xml
  [12] score_2622-Tu_passaste_por_este_Jardim-Alfredo_Dutra.xml

### CCM

**version7 onwards: fixed note mapping bug in train.py**
- version_7 batches = (128,256, 512) lr= 0.001, patience=20epochs (16.6%)
- version_8 batches = (16,32,64) lr= 0.001, patience=20epochs

### CTM

version2 onwards: fixed note mapping bug in train.py
- version_2 batches = (8,64,64), lr = 0.001, patience=20epochs (95.85%)

# Runs

ft1 = {ccm7,ctm2,csm3,ksm2,ktm0,no-cpm}