# imbalanced_synth

An evaluation of the approaches to handling imbalanced classification tasks insipired by https://sdv.dev/blog/synthetic-label-balancing/.

The article mentions that generators from Synthetic Data Vault (SDV) can be used to handle data imbalances. Although you can definitely use their generators, the article does not provide any evidence as to whether you should as well. This blog aims to provide an analysis of common approaches to dealing with imbalanced datasets and compare them against the use of SDV generators. Specifically, this blog compares:
- Random Oversampling (ROS)
- Synthetic Minority Over-sampling TEchnique (SMOTE)
- Cost-sensitive Learning
- SDV generators


Other stuff to integrate for finalizing project:
- Still move the scoring function to helper_functions
- Results
    - Write text
    - Make animation for CV setup (proper + improper way)
- Look at TDS requirement for publishment