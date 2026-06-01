# Test-Time Adaptation via Entropy Minimization (TENT) Analiz Raporu

## 1. Proje Özeti
Bu çalışma, test aşamasında karşılaşılan veri dağılımı kaymalarına (distribution shift) karşı derin öğrenme modellerinin dayanıklılığını artırmayı hedefleyen **TENT (Test-Time Adaptation)** algoritmasının PyTorch tabanlı bir gerçeklemesini içermektedir. Proje kapsamında, CIFAR-10 veri seti üzerinde önceden eğitilmiş ResNet-56 mimarisi kullanılmış olup, test verisi üzerine uygulanan Gaussian gürültüsü altında etiketsiz (unsupervised) ve çevrimiçi (online) adaptasyon süreçleri incelenmiştir.

Makale Referansı: *Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021.*

## 2. Teorik Çerçeve ve Algoritma Mantığı
TENT algoritması, test verisindeki dağılım kaymalarına uyum sağlamak için modelin çıktı belirsizliğini temsil eden Shannon Entropisini minimize etmeyi amaçlar. 
- **Optimizasyon Hedefi:** $H(p) = -\sum p \cdot \log(p)$ (burada $p$, softmax çıktı dağılımını ifade eder).
- **Parametre Güncellemesi:** Modelin tamamı yerine yalnızca `BatchNorm2d` katmanlarındaki ölçekleme (scale, $\gamma$) ve kaydırma (shift, $ eta$) parametreleri güncellenir. Bu yaklaşım, modelin aşırı öğrenmesini (overfitting) ve gradyan patlamasını engellerken, adaptasyon sürecini hızlandırır.
- **Uygulama Modeli:** Adaptasyon süreci, herhangi bir etiket verisine ihtiyaç duymadan (tamamen denetimsiz) ve test verisi akarken eşzamanlı (online) olarak gerçekleştirilir.

## 3. Deney Kurulumu ve Metodoloji
Deneyler aşağıda belirtilen konfigürasyon üzerinden gerçekleştirilmiştir:
- **Kullanılan Model:** ResNet-56 (CIFAR-10 üzerinde önceden eğitilmiş)
- **Uygulanan Bozulma (Corruption):** Gaussian Gürültüsü (Standart sapma, $\sigma = 0.25$)
- **Veri İşleme (Batch) Yapısı:** 64 örneklem içeren 50 iterasyon
- **Optimizasyon Algoritması:** Adam Optimizer (Öğrenme oranı, $ lpha = 0.001$)
- **Donanım Altyapısı:** CUDA

## 4. Deneysel Bulgular ve Performans Analizi
Yapılan testler sonucunda, herhangi bir adaptasyon uygulanmayan "Baseline" model ile TENT algoritmasıyla adapte edilen modelin başarım oranları karşılaştırılmıştır.

### 4.1. Doğruluk (Accuracy) Karşılaştırması
| Değerlendirme Aşaması | Ortalama Top-1 Doğruluk Oranı |
| :--- | :--- |
| **Baseline (Adaptasyon Yok)** | **% 12.12** |
| **TENT (Çevrimiçi Adaptasyon)** | **% 29.09** |

* **Mutlak İyileşme (Absolute Improvement):** +% 16.97
* **Göreceli İyileşme (Relative Improvement):** +% 139.95

### 4.2. Entropi Minimizasyon Analizi
Modelin tahmin kararlılığını ölçen entropi değerleri, optimizasyon süreci boyunca başarılı bir şekilde düşürülmüştür.
- **Başlangıç Entropi Değeri:** 0.6780
- **Bitiş Entropi Değeri:** 0.4640
- **Entropi Düşüş Miktarı:** 0.2141
- **Optimizasyon Durumu:** Başarıyla minimize edildi ve yakınsama gözlemlendi.

## 5. Çıktılar ve Görselleştirme
Script çalıştırıldığında `tent_analiz_grafikleri.png` isimli üç panelli bir analiz görseli oluşturulmaktadır. Bu grafik seti şunları içerir:
1. Batch bazında Doğruluk Karşılaştırması (Baseline vs TENT)
2. Entropi Düşüş Eğrisi (Hareketli ortalama ile düzleştirilmiş)
3. Kümülatif Ortalama Doğruluk (Kararlılık analizi)
## 6. Örnek Konsol Çıktısı

Projeyi çalıştırdığınızda terminalde aşağıdaki gibi detaylı bir analiz raporu üretilecektir:

```text
┌──────────────────────────────────────────────────────────────┐
│  ORTAM BİLGİSİ                                               │
└──────────────────────────────────────────────────────────────┘
  Cihaz   : CUDA
  PyTorch : 2.10.0+cu130
  CUDA    : 13.0

┌──────────────────────────────────────────────────────────────┐
│  AŞAMA 1 ▸ BASELINE  (Adaptasyon Yok)                        │
└──────────────────────────────────────────────────────────────┘
  Baseline: 100%|██████████████████████████████| 50/50 [00:00<00:00, 91.66it/s, Doğruluk=%14.06]

  ● Baseline Ortalama Doğruluk : %12.12

┌──────────────────────────────────────────────────────────────┐
│  AŞAMA 2 ▸ TENT  (Çevrimiçi Adaptasyon — Etiketsiz)          │
└──────────────────────────────────────────────────────────────┘
  TENT    : 100%|██████████████████████████████| 50/50 [00:00<00:00, 51.25it/s, Entropi=0.4640, Doğruluk=%35.94]

  ● TENT Ortalama Doğruluk     : %29.09


╔══════════════════════════════════════════════════════════════╗
║          TEST-TIME ADAPTATION  ·  ANALİZ RAPORU              ║
╚══════════════════════════════════════════════════════════════╝

  Deney Kurulumu
  ├─ Model     : ResNet-56  (CIFAR-10 pretrained)
  ├─ Bozulma   : Gaussian Gürültüsü  σ = 0.25
  ├─ Batch     : 64 örnek  ×  50 iterasyon
  └─ Optimize  : Adam  (lr = 0.001)

  📊 Performans Karşılaştırması
  ├─ Baseline (Adaptasyon Yok)  : % 12.12
  ├─ TENT     (Online Adapt.)   : % 29.09
  ├─ Mutlak İyileşme            : +%16.97
  └─ Göreceli İyileşme          : +%139.95

  📉 Entropi Analizi  H(p) = −Σ p·log(p)
  ├─ Başlangıç Entropisi        : 0.6780
  ├─ Bitiş Entropisi            : 0.4640
  ├─ Düşüş Miktarı             : 0.2141
  └─ Sonuç                      : ✅  Başarıyla minimize edildi
