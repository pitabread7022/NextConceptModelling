1. git submodule add https://github.com/AndrewBoessen/vq-vae.git vqvae
2. uv pip install -r .\vqvae\requirements.txt
3. git remote add origin https://github.com/pitabread7022/NextConceptModelling.git
4. uv run --no-sync python -m vqvae.train --max-updates 1
5. uv run .\src\dwnld_dota.py
6. uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
7. uv run python -c "import torch; print(torch.cuda.is_available())"

uv run python -m src.ncm_modelling.train --steps 300 --eval_every 150 --batch_size 8
uv run python -m src.ncm_modelling.train --steps 6000 --batch_size 16
uv run python -m src.ncm_modelling.cache_codes --dihedral
uv run python -m src.ncm_modelling.export_onnx --level_steps --verify