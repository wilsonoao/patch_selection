python == 3.10.12

Feature Structure

TCGA-LUAD-FS/
└── CHIEF/
    └── 20X/
        ├── h5_files(stain_norm)/
        │   ├── TCGA-XX-XXXX-01A-01-TS1.h5
        │   ├── TCGA-XX-XXXX-01A-01-TS2.h5
        │   └── ...
        │
        ├── pt_files(stain_norm)/
        │   ├── TCGA-XX-XXXX-01A-01-TS1.pt
        │   ├── TCGA-XX-XXXX-01A-01-TS2.pt
        │   └── ...
        │
        └── cluster_record_spatialleiden.pkl


PKL Structure

04_LUAD/
    ├── LUAD_Common_Genes_CSMD3-Percentage_39.9_.pkl
    ├── LUAD_Common_Genes_MUC16-Percentage_42.8_.pkl
    ├── LUAD_Common_Genes_RYR2-Percentage_38.3_.pkl
    ├── LUAD_Common_Genes_TP53-Percentage_52.1_.pkl
    └── LUAD_Common_Genes_TTN-Percentage_48.1_.pkl


Execution Order

run_clustering.sh | run_train_baseline.sh   (can be executed simultaneously)
run_train.sh


Notes

SAVE_BASE_DIR must be the same for all three stages.
