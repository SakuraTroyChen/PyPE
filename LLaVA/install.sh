pip install --upgrade pip 
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install -e ".[train]"
pip install triton==2.1.0 pynvml==11.5.0 --upgrade
pip install flash-attn==2.5.8 --no-build-isolation --no-cache-dir
pip install -U xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
pip install scikit-image
pip install seaborn
