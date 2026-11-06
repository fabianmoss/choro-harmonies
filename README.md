[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fabianmoss/choro-harmonies)

### h5_data seeds

seed0: created 31.10.2026 on 26 scores `h5_data`, 25 scores `h5_ksm`

### flowcharts order

create_h5_data.py => train.py => annotate.py => write_to_musicxml.py

### Best performing models 

CCM: 
lightning_logs/v09_ft_batch-tiny-lr-0.005 *(52.7% acc)*
CTM: 
lightning_logs/v02_ft_moredata *(97.97% acc)*
CSM: lightning_logs/v04_ft_batch-tiny-lr-0.01 *(57.25% acc)*
KTM: lightning_logs/v02_scratch_standard *(96.72% acc)*
KSM: 
lightning_logs/v04_ft_batch-lower-lr-0.005 *(75% acc)*

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
- **version_6 batches = (16,32,64), lr = 0.001, patience=20epochs (50.4%)**

### CTM
- version_0 batches = (8,64,64), lr = 0.001, patience=20epochs (98.0%)
- version_1 batches = (4,32,32), lr = 0.001, patience=20epochs (97.9%)

### CSM
- version_0 batches = (32,64,256), lr = 0.001, patience=20epochs (23.6%)
- version_1 batches = (16,32,128), lr = 0.001, patience=20epochs (23.6%)
- version_2 batches = (8,16,64), lr = 0.001, patience=20epochs   (55.9%)
- **version_3 batches = (8,16,64), lr = 0.005, patience=50epochs   (60.0%)**
- version_4 batches = (8,16,64), lr = 0.005, patience=20epochs   (57.7%)
- version_5 batches = (8,16,64), lr = 0.01,  patience=20epochs   (57.5%)
- version_6 batches = (8,16,64), lr = 0.01,  patience=40epochs   (56.8%)

### KSM
- version_0 batches = (32,64,256), lr = 0.001, patience=20epochs (25.0%)
- version_1 batches = (32,64,256), lr = 0.01, patience=20epochs (75.0%)
- **version_2 batches = (32,64,256), lr = 0.01, patience=35epochs (100%)**

### KTM
- **version_0 batches = (32,64,256), lr = 0.001, patience=20epochs (96.5%)**
