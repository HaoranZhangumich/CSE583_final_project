# DeepTune IR Transformer

## Files
- `config.py`: constants, config, utility functions, repo CSV loading
- `ir_pipeline.py`: OpenCL source -> LLVM IR, IR normalization, IR cache
- `tokenizer_utils.py`: WordPiece tokenizer training and IR encoding
- `model.py`: Transformer classifier, dataset wrapper, train/predict helpers
- `tasks_eval.py`: device mapping + thread coarsening preparation and evaluation
- `main.py`: CLI entry point

## Run
```bash
pip install pandas numpy torch scikit-learn tokenizers tqdm
python3 main.py --repo-root .. --out-dir ./out --task all
```