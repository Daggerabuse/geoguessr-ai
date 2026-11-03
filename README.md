# geoguessr-ai

Sokak fotoğrafından ülke tahmini.

## Kaynaklar ve lisanslar
- Veri: [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m), **CC BY-SA 4.0**.
  Astruc et al., "OpenStreetView-5M: The Many Roads to Global Visual Geolocation", CVPR 2024.
- Modeller: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14);
  [geolocal/StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) (**CC BY-NC 4.0**, ticari kullanım yok).

## Kurulum
Yerelde (torch/transformers yok; sadece keşif, veri hazırlama ve kod):
```
uv sync
```
Kaggle/Colab (internet açık). torch ve transformers genelde hazır gelir; `pip install -e .`
yalnızca çekirdek bağımlılıkları kurar. Hazır değilse: `pip install -e ".[ml]"`.
```
pip install -e .
```
Yerelde de ML paketleri gerekirse: `uv sync --extra ml` (büyük indirme).

## Çalıştırma
```
# Keşif (yerel, data/sample gerekir)
uv run python -m geoguessr_ai.download_sample
uv run python -m geoguessr_ai.explore

# Kaggle: önce dağılım, sonra alt küme (manifest + görüntüler)
python -m geoguessr_ai.prep_data --stats-only
python -m geoguessr_ai.prep_data --out /kaggle/working/osv5m_subset

# Kaggle: CLIP zero-shot baseline (sonuçlar results/ altına)
python -m geoguessr_ai.baseline_clip --root /kaggle/working/osv5m_subset
python -m geoguessr_ai.baseline_clip --root /kaggle/working/osv5m_subset \
    --model geolocal/StreetCLIP --out results/streetclip_zeroshot
```

`data/` ve `models/` repoya girmez; `results/` girer.
