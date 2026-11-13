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

        h5_data
        h5_ksm

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
        flowcharts
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
