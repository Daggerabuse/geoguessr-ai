# geoguessr-ai: Codex için notlar

## Proje
Sokak fotoğrafından ülke tahmini. Veri: OpenStreetView-5M (Hugging Face, osv5m).
Amaç sadece yüksek doğruluk değil: süreci anlamak ve sonuçları dürüstçe karşılaştırmak
(zero-shot baseline → fine-tuning).

## Kurallar
- Paket yönetimi uv ile: `uv add`, `uv run`. pip kullanma.
- Commit'leme; bitince değişiklikleri özetle, ben inceleyip commit'lerim.
- Davranışı değiştirecek bir şey ya da bir tasarım kararı varsa önce sor;
  seçenekleri artı ve eksileriyle sun, kararı bana bırak.
- Fotoğraflar ve modeller `data/` ve `models/` altında; ikisi de repoya girmez.
- Yorumlarda ve docstring'lerde deneyden gelen sayıları (doğruluk vb.) yazma;
  sonuçlar ayrı bir dosyada ya da README'de tutulur.
- Eğitim GPU gerektirir: Kaggle Notebooks ya da Colab'da çalışacak.
  Yerelde sadece keşif, küçük denemeler ve kod.
- Rastgelelik içeren her şeyde seed sabitle (tekrarlanabilirlik).
- Veri setinin lisansına uy ve README'de kaynak göster.
- Eğitimde yatay çevirme (horizontal flip) KULLANMA: Arabaların yolun hangi tarafından gittiği önemli bir ipucu.
