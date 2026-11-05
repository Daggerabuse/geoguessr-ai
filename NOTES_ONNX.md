# NOTES_ONNX: ONNX export + tarayıcı demosu (Static Space)

Gece yapılan iş (2026-11-05). Commit/push yok, Hub'a hiçbir şey yüklenmedi, dosya silinmedi,
`app/` klasörüne dokunulmadı. `README.md`'deki değişiklik (License satırı) senin önceden
yaptığın değişiklik; ona da dokunmadım.

## Özet

- `src/geoguessr_ai/export_onnx.py`: export_hf klasörü -> tek ONNX grafiği (vision encoder +
  başlık, çıktı 33 logit), fp32 + 8-bit. `--tiny` ile indirmeden uçtan uca test, `--verify`
  ile PyTorch karşılaştırması.
- Model Hub'dan indirildi (`models/hf_export/`). Downloads'taki kopya ile sha256'ları birebir
  aynı (model.safetensors, head.safetensors, config.json, classes.json, inference_config.json).
  Gerçek dönüşüm yapıldı; sabaha bırakılan dönüşüm yok.
- `web/`: statik demo (`index.html`, `app.js`, `preprocess.js`, `style.css`, Space için `README.md`).
- `src/geoguessr_ai/webdemo.py`: demoyu yerelde sunma (`serve`) ve gerçek tarayıcıda
  (Edge headless, Playwright) PyTorch ile karşılaştırma (`check`).
- Testler: `tests/test_export_onnx.py`, `tests/test_web_preprocess.py` (+ `tests/js/preprocess_cli.mjs`),
  `tests/test_webdemo.py`. `uv run ruff check` temiz; `uv run --extra ml --extra onnx pytest`: 138 geçti.
- Sonuç dosyaları: `results/onnx_parity/report.json` (Python/ORT), `web_int8.json`, `web_fp32.json` (tarayıcı).

## Bağımlılıklar (uv ile)

| Ne | Nereye | Neden |
|---|---|---|
| `onnx`, `onnxruntime`, `onnxscript` | yeni extra `onnx` | export + quantization + doğrulama; torch 2.15'in ONNX exporter'ı (dynamo) onnxscript ister |
| `torch`, `transformers` | zaten vardı (extra `ml`) | `uv sync --extra ml` ile kuruldu |
| `nodejs-wheel-binaries` | `dev` grubu | JS ön işleme testi node'da çalışıyor; dev grubunda olduğu için CI'da (`uv sync --locked`) da çalışır. Sistemde node yoktu, sisteme bir şey kurmadım |
| `playwright` | yeni grup `browser` (varsayılan değil) | tarayıcı doğrulaması; kurulu Edge'i kullanıyor (`channel=msedge`), tarayıcı indirmiyor. CI'ya girmiyor |

CI etkisi: torch/onnx gerektiren testler `importorskip` ile CI'da atlanır (CI sadece dev grubunu
kuruyor). JS testi CI'da çalışır.

## Kararlar ve gerekçeleri

1. **Tek grafik, çıktı `(N, 33)`; kırpıntı ortalaması grafiğin dışında.** Grafik
   `logit_scale * normalize(image_embeds) @ normalize(head).T` hesaplıyor; `N` dinamik
   (1..64), TTA'nın 3 kırpıntısı tek batch'te gidiyor. Ortalamayı JS yapıyor (app/app.py
   PyTorch'ta da aynısını yapıyor). Gerekçe: aynı grafik tek kırpıntı (N=1) için de
   kullanılabilir, kırpıntı başına logitler doğrulanabilir. Başlık export sırasında bir kez
   normalize edilip sabit olarak gömülü.
2. **Exporter:** `torch.onnx.export(dynamo=True)`, opset 18, `attn_implementation="eager"`
   (SDPA yerine düz MatMul/Softmax: her runtime'da, web dahil, çalışsın diye). Dosyalar tek
   parça (external data yok) ki tarayıcı tek fetch'le alsın. Exporter'ın yazdığı `value_info`
   içinde onnx shape inference'ın reddettiği bir kayıt vardı (quantization'ı durduruyordu);
   `value_info` isteğe bağlı metadata olduğu için silinip yeniden çıkarılıyor. Grafiğin
   doğruluğu aşağıdaki karşılaştırmayla ayrıca kontrol edildi.
3. **8-bit yöntemi: MatMulNBits (bits=8, blok 32, simetrik), yalnızca ağırlıklar 8-bit.**
   Önce onnxruntime `quantize_dynamic` (aktivasyonlar da 8-bit) denedim; PyTorch ile top-1
   uyumu çok düştü. Seçim **ilk 500 örnek görüntüde** yapıldı, rapor edilen sayılar ise
   seçimde kullanılmayan **kalan 2500 görüntüde** (seçim ile ölçüm aynı veride olmasın diye).
   Seçim kümesindeki sonuçlar (500 görüntü, PyTorch'a göre, TTA):

   | Yöntem | top-1 uyumu | top-5 küme uyumu | max logit farkı | boyut |
   |---|---|---|---|---|
   | quantize_dynamic QInt8 per-channel | 0.854 | 0.302 | 3.14 | 90 MB |
   | quantize_dynamic QInt8 per-tensor | 0.828 | 0.296 | 4.07 | 90 MB |
   | quantize_dynamic QUInt8 per-tensor | 0.818 | 0.308 | 3.42 | 90 MB |
   | quantize_dynamic QUInt8 per-channel | 0.838 | 0.308 | 3.02 | 90 MB |
   | quantize_dynamic QInt8 per-channel, reduce_range | 0.844 | 0.252 | 3.77 | 90 MB |
   | weight-only, kanal başına int8 (QDQ, elle) | 0.956 | 0.566 | 0.70 | 90 MB |
   | **MatMulNBits 8-bit, blok 32** | **0.990** | **0.882** | **0.24** | **101 MB** |
   | MatMulNBits 8-bit, blok 16 | 0.996 | 0.906 | 0.22 | 112 MB |
   | MatMulNBits 8-bit, blok 64 | 0.980 | 0.886 | 0.26 | 96 MB |
   | MatMulNBits 8-bit, blok 128 | 0.978 | 0.836 | 0.32 | 94 MB |

   Yorum: aktivasyonları 8-bit'e indirmek bu modelde zarar veriyor (CLIP ViT'lerde bilinen
   aktivasyon aykırı değerleri ile uyumlu, ama bunu ayrıca kanıtlamadım). Kanal başına tek
   ölçek de yetmiyor; blok başına ölçek (32 satır) farkı büyük ölçüde kapatıyor. Blok 16
   biraz daha iyi ama fark 500 görüntüde 3 görüntü (gürültü düzeyinde) ve dosya 11 MB büyük;
   onnxruntime'ın varsayılanı olan 32'yi seçtim. `--int8-method dynamic` eski yöntemi hâlâ
   üretebiliyor (karşılaştırmayı tekrarlamak için). Görsel projeksiyon katmanı (Gemm, ~0.4M
   parametre) MatMulNBits'e çevrilmiyor, fp32 kalıyor; 33 sınıflık başlık 8-bit oluyor.
   İlk (dinamik int8) export'u silmedim, `models/onnx_superseded_dynamic_int8/` altına taşıdım;
   istersen silebilirsin.
4. **Ön işleme JS'de PIL ile birebir.** `web/preprocess.js` Pillow'un `Resample.c`'sini
   (bicubic a=-0.5, küçültmede genişleyen destek, 22-bit sabit noktalı katsayılar, her
   geçişte uint8'e yuvarlama/kırpma) yeniden uyguluyor; Python `round()` (half-to-even) ve
   float32 aritmetik sırası da aynı. Canvas'ın kendi ölçeklemesini kullanmadım (aşağıda neden).
5. **Pillow sürüm farkı (önemli bir bulgu).** Pillow 13.0 (uv.lock'taki sürüm) yeni bir kural
   getirmiş: yükseklik, genişliğin iki katından fazla piksel küçülüyorsa önce **dikey** geçiş
   yapılıyor; Pillow ≤ 12.1 her zaman önce yatay geçiyordu. Her geçiş uint8'e yuvarladığı
   için sıra birkaç pikseli birkaç seviye değiştiriyor (testte 97x1301 görüntüde max 10
   seviye). JS, Pillow 13'ü izliyor (test birebir geçsin, yereldeki Python ile aynı olsun).
   Eğitimin Kaggle'da hangi Pillow ile yapıldığı kayıtlı değil (muhtemelen ≤ 12). Pratik
   etkisi: kural sadece uzun dikey (portre) görüntülerde tetikleniyor; `data/sample`'daki
   3000 görüntünün **0**'ında tetikleniyor (4'ü portre, hiçbiri eşiği geçmiyor).
6. **EXIF ve renk profili.** Tarayıcıda EXIF yönü uygulanıyor (`imageOrientation: "from-image"`):
   telefondan yan çekilmiş bir fotoğraf kullanıcının gördüğü gibi işlenir. Python tarafı
   (`train.eval_views`) EXIF'i uygulamıyor; bu bilinçli bir sapma. ICC renk profili
   uygulanmıyor (`colorSpaceConversion: "none"`), PIL de uygulamıyor. Örnek görüntülerde ne
   EXIF yönü ne ICC profili var, yani doğrulamayı etkilemiyor.
7. **Çok büyük fotoğraflar** (> 16.7 MP, iOS Safari'nin canvas sınırı): tarayıcı önce
   `createImageBitmap(resizeQuality: "high")` ile küçültüyor, sonra PIL-birebir yol. Bu durumda
   ön işleme PIL ile birebir değil. Test edilmedi.
8. **Varsayılan model 8-bit, `?model=fp32` ile fp32.** Tarayıcıda hız neredeyse aynı (aşağıda),
   ama indirme 101 MB'a karşı 346 MB ve mobil bellek için daha güvenli. Model Cache Storage'da
   saklanıyor (ikinci ziyarette indirme yok). Hub'daki ONNX dosyalarını değiştirirsen
   `web/app.js`'deki `CACHE_NAME`'i artır, yoksa eski ziyaretçiler eski dosyayı kullanır.
9. **Tek iş parçacığı (WASM).** Çok iş parçacığı sayfanın cross-origin isolated olmasını
   (COOP/COEP başlıkları) gerektiriyor; bu, Hub'dan ve CDN'den yapılan fetch'leri bozma riski
   taşıyor ve test edemedim. Kod, sayfa isolated ise otomatik olarak 4 thread kullanıyor;
   ileride Space README'sine `custom_headers` eklenerek denenebilir. WebGPU denemedim.
10. **onnxruntime-web 1.31.1**, jsDelivr'den sabit sürümle (npm'deki son sürüm, Python
    onnxruntime ile aynı). Sayfa İngilizce (Gradio demosu ve README ile tutarlı).
11. **Yerelde model yolu:** `?base=URL` parametresi ile model Hub yerine başka bir yerden
    (örn. `webdemo serve`'ün `/model/`'i) okunuyor. Varsayılan Hub:
    `https://huggingface.co/Daggerabuse/geoguessr-clip-vitb16/resolve/main/` + `onnx/model_int8.onnx`.
    Hub `resolve` uçlarının CORS izni verdiğini kontrol ettim (static.hf.space origin'i ile).

## Doğrulama sonuçları

Hepsi TTA (sol/orta/sağ kırpıntı, logit ortalaması), referans PyTorch modeli
(`CLIPVisionModelWithProjection` + başlık, fp32, CPU). "logit farkı" = |ONNX − PyTorch|,
logitler `logit_scale = 100` ile ölçekli kosinüs benzerliği. Bunlar **uyum** ölçüleri,
doğruluk değil (örnek görüntüler eğitim havuzuyla örtüşüyor olabilir; doğruluk hesaplamadım).

### Python, onnxruntime CPU — `data/sample` görüntüleri 501..3000 (2500 görüntü)

| | fp32 | 8-bit (MatMulNBits) |
|---|---|---|
| max logit farkı | 9.9e-05 | 0.269 |
| ortalama logit farkı | 3.7e-06 | 0.043 |
| p99 (görüntü başına max) | 2.1e-05 | 0.206 |
| max olasılık farkı | 1.3e-05 | 0.042 |
| top-1 uyumu | **1.0000** | **0.9856** |
| PyTorch top-1'i ONNX top-5 içinde | 1.0000 | 1.0000 |
| top-5 kümesi aynı | 1.0000 | 0.8896 |
| süre / görüntü (3 kırpıntı, bu laptop) | 0.27 s (PyTorch 0.39 s) | 0.65 s |

Not: 8-bit, onnxruntime CPU'da fp32'den yavaş (ağırlıklar her çarpımda açılıyor); kazancı
boyut. 8-bit'in top-1'de ayrıştığı ~%1.4 görüntüde PyTorch'un cevabı hep 8-bit'in top-5'inde.

### Gerçek tarayıcı (Edge 156 headless, onnxruntime-web 1.31.1 WASM, tek thread) — görüntüler 501..700 (int8) / 501..600 (fp32)

| | 8-bit (200 görüntü) | fp32 (100 görüntü) |
|---|---|---|
| A. PIL pikselleri -> preprocess.js -> ORT-web, max logit farkı | 0.216 | 2e-05 |
| A. top-1 uyumu | 0.995 | 1.000 |
| A. top-5 kümesi aynı | 0.885 | 1.000 |
| B. JPEG'i tarayıcı çözüyor (demodaki yol): top-1 uyumu | 0.995 | 1.000 |
| B. tarayıcı JPEG çözümü vs PIL: max piksel farkı | 0 | 0 |
| model süresi / görüntü (3 kırpıntı) | ~2.3 s | ~2.2 s |
| ön işleme süresi / görüntü | ~16 ms | ~16 ms |

Edge'in JPEG çözücüsü bu görüntülerde PIL ile bit düzeyinde aynı çıktı verdi, yani B = A.
Safari/Firefox/Chrome-Android'in çözücüsü farklı olabilir; denemedim.

### JS ön işleme testi (`tests/test_web_preprocess.py`)

Aynı çözülmüş pikseller Python'a ve node'daki `preprocess.js`'e veriliyor; yeniden
boyutlandırılmış görüntü (uint8) ve normalize edilmiş float32 dizisi karşılaştırılıyor.
22 görüntü × (TTA var/yok) = 44 vaka: gradyanlar, gürültü, keskin kenarlı dama (uint8
kırpmayı zorlar), RGBA girdi, büyütme (100x80), dikey-önce kuralını tetikleyen portreler,
3000x4000, ve `data/` varsa 8 gerçek fotoğraf. Sonuç: **piksel farkı 0, float dizileri
birebir eşit** (`np.array_equal`).

### Bicubic (PIL) ile canvas `drawImage` farkı

Aynı görüntüyü canvas ile (`imageSmoothingQuality: "high"`, Edge) kısa kenar 224'e
küçültüp PIL bicubic ile karşılaştırdım (200 görüntü):

- piksel farkı: görüntü başına ortalama 0.79 seviye (en kötü görüntüde 2.9), max fark
  ortalama 20 seviye (en kötü 57), piksellerin ~%44'ü farklı.
- Bu pikselleri PyTorch modeline verince (fp32, PIL yolu referans): top-1 uyumu **0.95**,
  top-5 kümesi aynı 0.605, max logit farkı 1.41.

Yani canvas ölçeklemesi 8-bit quantization'dan (%98.6–99.5) daha fazla tahmin değiştiriyor.
Canvas'ın "high" kalitesinin hangi filtre olduğu standartta tanımlı değil, tarayıcıdan
tarayıcıya değişebilir (sadece Edge'i ölçtüm); bu yüzden demoda canvas
yalnızca çözülmüş fotoğrafı piksellere dökmek için kullanılıyor, ölçekleme `preprocess.js`'de.

## Sabah senin yapman gerekenler

### 1. ONNX dosyalarını model reposuna yükle

Dosyalar `models/onnx/` altında hazır: `model.onnx` (346 MB), `model_int8.onnx` (101 MB),
`export_info.json`. Demo bunları reponun `onnx/` klasöründe arıyor.

```
uv run hf auth login      # write token ile (ya da HF_TOKEN ortam değişkeni)
uv run hf upload Daggerabuse/geoguessr-clip-vitb16 models/onnx/model_int8.onnx onnx/model_int8.onnx
uv run hf upload Daggerabuse/geoguessr-clip-vitb16 models/onnx/model.onnx onnx/model.onnx
uv run hf upload Daggerabuse/geoguessr-clip-vitb16 models/onnx/export_info.json onnx/export_info.json
```

İsteğe bağlı: model kartına (Hub'daki README.md) bir "ONNX" bölümü eklemek (girdi
`pixel_values (N,3,224,224)`, çıktı `logits (N,33)`, kırpıntı ortalaması dışarıda, 8-bit =
MatMulNBits). `export_hf.model_card`'ı değiştirmedim.

### 2. Static Space'i kur

```
uv run hf repos create Daggerabuse/geoguessr-ai --type space --space-sdk static --public
uv run hf upload Daggerabuse/geoguessr-ai web . --repo-type space
```

`web/README.md` Space'in yapılandırması (`sdk: static`, `app_file: index.html`, model linki).
Space adı (`geoguessr-ai`) benim seçimim; istediğin adı verebilirsin (kodda Space adı geçmiyor).
Space açılınca: https://huggingface.co/spaces/Daggerabuse/geoguessr-ai . İlk açılışta model
indirilir (101 MB), sonra cache'ten gelir. Gerçek Space'te test edemedim (yükleme yasaktı);
ilk açılışta tarayıcı konsoluna bak.

Web arayüzünden yapmak istersen: huggingface.co/new-space -> SDK "Static" -> "Blank", sonra
`web/` içindeki 5 dosyayı (index.html, app.js, preprocess.js, style.css, README.md) yükle;
README.md'nin üstündeki YAML bloğunun kalması gerekiyor.

### 3. Yerelde dene

Hub'a yüklemeden önce (model yerel klasörlerden sunulur):

```
uv run python -m geoguessr_ai.webdemo serve
```

Sonra tarayıcıda http://localhost:8000/web/?base=/model/ (fp32 için `&model=fp32`).
Hub'a yükledikten sonra http://localhost:8000/web/ Hub'daki modeli kullanır.
(`file://` ile açmak çalışmaz: ES modülleri ve fetch bir HTTP sunucusu ister.)

Tarayıcı doğrulamasını tekrarlamak (kurulu Edge ile, ~15 dk):

```
uv run --extra ml --extra onnx --group browser python -m geoguessr_ai.webdemo check \
    --offset 500 --limit 200 --variant int8 --report results/onnx_parity/web_int8.json
```

### 4. Export'u ve Python doğrulamasını tekrarlamak

```
uv sync --extra ml --extra onnx
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Daggerabuse/geoguessr-clip-vitb16', local_dir='models/hf_export')"
uv run python -m geoguessr_ai.export_onnx --model models/hf_export --out models/onnx \
    --verify data/sample/images --offset 500 --report results/onnx_parity/report.json
uv run python -m geoguessr_ai.export_onnx --tiny --out models/onnx_tiny   # indirmesiz duman testi
```

Export ~15 s, 2500 görüntülük doğrulama bu laptopta ~1 saat. `--out` klasörü varsa içindeki
model dosyalarının üzerine yazar.

## Açık kalanlar / test edilmeyenler

- Gerçek Static Space, mobil tarayıcılar, Safari, Firefox: denenmedi (sadece Edge headless).
- Çok iş parçacıklı WASM / WebGPU: denenmedi (karar 9).
- 16.7 MP üstü fotoğraflarda ön küçültme yolu: denenmedi.
- Pillow 12/13 farkı eğitim ortamı açısından belirsiz (karar 5); örnek verilerde etkisiz.
- `README.md`'ye demo linki/ONNX bölümü eklemedim; Space açılınca eklemek isteyebilirsin.
- Örnek görüntülerin seçim/rapor bölümü (ilk 500 / kalan 2500) dosya adına göre sıralı;
  rastgelelik yok, bu yüzden seed gerekmedi. Tiny model `torch.manual_seed(0)` kullanıyor.
