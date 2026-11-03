# Confusion analizi (test)

## clip_vitb16_zeroshot
top-1 (recall ağırlıklı ort.) = 0.468, ülke sayısı = 33

### Çekici ülkeler (tahmin/gerçek ≥ eşik ve precision < recall)
| country | recall | precision | pred_ratio | attractor_clear |
|---|---|---|---|---|
| NZ | 0.92 | 0.33 | 2.81 | True |
| US | 0.80 | 0.29 | 2.75 | True |
| FI | 0.95 | 0.41 | 2.33 | True |
| AU | 0.92 | 0.44 | 2.10 | True |
| ZA | 0.77 | 0.43 | 1.77 | True |
| NG | 0.65 | 0.40 | 1.63 | True |
| BR | 0.49 | 0.31 | 1.59 | True |
`attractor_clear=True`: precision aralığı recall aralığından ayrışıyor.

### Odak ülkeler
| country | recall | recall_lo | recall_hi | precision | pred_ratio |
|---|---|---|---|---|---|
| FI | 0.95 | 0.91 | 0.97 | 0.41 | 2.33 |
| SE | 0.28 | 0.22 | 0.35 | 0.29 | 0.97 |

### En çok karışan 10 çift (gerçek → tahmin)
| true | pred | count | share_of_true |
|---|---|---|---|
| SE | FI | 115 | 0.575 |
| GH | NG | 108 | 0.540 |
| CL | NZ | 69 | 0.345 |
| AR | US | 44 | 0.220 |
| RU | FI | 43 | 0.215 |
| NO | FI | 40 | 0.200 |
| MX | US | 39 | 0.195 |
| CA | US | 38 | 0.190 |
| DE | PL | 37 | 0.185 |
| GB | NZ | 36 | 0.180 |

### İlgilenilen çiftler (gerçek → tahmin; yönlü)
| true | pred | count | share_of_true | lo | hi |
|---|---|---|---|---|---|
| SE | NO | 3 | 0.015 | 0.005 | 0.043 |
| SE | FI | 115 | 0.575 | 0.506 | 0.641 |
| SE | DK | 10 | 0.050 | 0.027 | 0.090 |
| NO | SE | 28 | 0.140 | 0.099 | 0.195 |
| NO | FI | 40 | 0.200 | 0.150 | 0.261 |
| NO | DK | 1 | 0.005 | 0.001 | 0.028 |
| FI | SE | 4 | 0.020 | 0.008 | 0.050 |
| FI | NO | 1 | 0.005 | 0.001 | 0.028 |
| FI | DK | 0 | 0.000 | 0.000 | 0.019 |
| DK | SE | 32 | 0.160 | 0.116 | 0.217 |
| DK | NO | 0 | 0.000 | 0.000 | 0.019 |
| DK | FI | 8 | 0.040 | 0.020 | 0.077 |
| AR | CL | 12 | 0.060 | 0.035 | 0.102 |
| CL | AR | 12 | 0.060 | 0.035 | 0.102 |
| ES | PT | 18 | 0.090 | 0.058 | 0.138 |
| PT | ES | 5 | 0.025 | 0.011 | 0.057 |
| NG | GH | 24 | 0.120 | 0.082 | 0.172 |
| GH | NG | 108 | 0.540 | 0.471 | 0.608 |
| US | CA | 21 | 0.105 | 0.070 | 0.155 |
| CA | US | 38 | 0.190 | 0.142 | 0.250 |

## clip_vitl14_zeroshot
top-1 (recall ağırlıklı ort.) = 0.544, ülke sayısı = 33

### Çekici ülkeler (tahmin/gerçek ≥ eşik ve precision < recall)
| country | recall | precision | pred_ratio | attractor_clear |
|---|---|---|---|---|
| US | 0.97 | 0.15 | 6.62 | True |
| GB | 0.94 | 0.36 | 2.60 | True |
| FI | 0.91 | 0.51 | 1.77 | True |
| NG | 0.74 | 0.49 | 1.52 | True |
| AU | 0.94 | 0.68 | 1.38 | True |
`attractor_clear=True`: precision aralığı recall aralığından ayrışıyor.

### Odak ülkeler
| country | recall | recall_lo | recall_hi | precision | pred_ratio |
|---|---|---|---|---|---|
| FI | 0.91 | 0.86 | 0.94 | 0.51 | 1.77 |
| SE | 0.17 | 0.12 | 0.23 | 0.51 | 0.34 |

### En çok karışan 10 çift (gerçek → tahmin)
| true | pred | count | share_of_true |
|---|---|---|---|
| CA | US | 150 | 0.750 |
| SE | FI | 119 | 0.595 |
| GH | NG | 104 | 0.520 |
| JP | US | 79 | 0.395 |
| DK | GB | 78 | 0.390 |
| TW | US | 73 | 0.365 |
| AR | US | 71 | 0.355 |
| TR | US | 71 | 0.355 |
| CL | US | 68 | 0.340 |
| MX | US | 61 | 0.305 |

### İlgilenilen çiftler (gerçek → tahmin; yönlü)
| true | pred | count | share_of_true | lo | hi |
|---|---|---|---|---|---|
| SE | NO | 24 | 0.120 | 0.082 | 0.172 |
| SE | FI | 119 | 0.595 | 0.526 | 0.661 |
| SE | DK | 4 | 0.020 | 0.008 | 0.050 |
| NO | SE | 10 | 0.050 | 0.027 | 0.090 |
| NO | FI | 26 | 0.130 | 0.090 | 0.184 |
| NO | DK | 0 | 0.000 | 0.000 | 0.019 |
| FI | SE | 1 | 0.005 | 0.001 | 0.028 |
| FI | NO | 4 | 0.020 | 0.008 | 0.050 |
| FI | DK | 0 | 0.000 | 0.000 | 0.019 |
| DK | SE | 17 | 0.085 | 0.054 | 0.132 |
| DK | NO | 8 | 0.040 | 0.020 | 0.077 |
| DK | FI | 5 | 0.025 | 0.011 | 0.057 |
| AR | CL | 11 | 0.055 | 0.031 | 0.096 |
| CL | AR | 20 | 0.100 | 0.066 | 0.149 |
| ES | PT | 11 | 0.055 | 0.031 | 0.096 |
| PT | ES | 19 | 0.095 | 0.062 | 0.144 |
| NG | GH | 44 | 0.220 | 0.168 | 0.282 |
| GH | NG | 104 | 0.520 | 0.451 | 0.588 |
| US | CA | 2 | 0.010 | 0.003 | 0.036 |
| CA | US | 150 | 0.750 | 0.686 | 0.805 |

## streetclip_zeroshot
top-1 (recall ağırlıklı ort.) = 0.708, ülke sayısı = 33

### Çekici ülkeler (tahmin/gerçek ≥ eşik ve precision < recall)
| country | recall | precision | pred_ratio | attractor_clear |
|---|---|---|---|---|
| NG | 0.85 | 0.47 | 1.80 | True |
| JP | 0.94 | 0.66 | 1.41 | True |
| SE | 0.74 | 0.55 | 1.35 | True |
| US | 0.91 | 0.69 | 1.31 | True |
`attractor_clear=True`: precision aralığı recall aralığından ayrışıyor.

### Odak ülkeler
| country | recall | recall_lo | recall_hi | precision | pred_ratio |
|---|---|---|---|---|---|
| FI | 0.57 | 0.51 | 0.64 | 0.76 | 0.76 |
| SE | 0.74 | 0.68 | 0.80 | 0.55 | 1.35 |

### En çok karışan 10 çift (gerçek → tahmin)
| true | pred | count | share_of_true |
|---|---|---|---|
| GH | NG | 104 | 0.520 |
| FI | SE | 64 | 0.320 |
| TW | JP | 40 | 0.200 |
| CA | US | 37 | 0.185 |
| NO | SE | 30 | 0.150 |
| PL | DE | 29 | 0.145 |
| PL | RU | 29 | 0.145 |
| DE | AT | 27 | 0.135 |
| MA | ES | 26 | 0.130 |
| CO | BR | 25 | 0.125 |

### İlgilenilen çiftler (gerçek → tahmin; yönlü)
| true | pred | count | share_of_true | lo | hi |
|---|---|---|---|---|---|
| SE | NO | 11 | 0.055 | 0.031 | 0.096 |
| SE | FI | 23 | 0.115 | 0.078 | 0.167 |
| SE | DK | 8 | 0.040 | 0.020 | 0.077 |
| NO | SE | 30 | 0.150 | 0.107 | 0.206 |
| NO | FI | 4 | 0.020 | 0.008 | 0.050 |
| NO | DK | 0 | 0.000 | 0.000 | 0.019 |
| FI | SE | 64 | 0.320 | 0.259 | 0.388 |
| FI | NO | 4 | 0.020 | 0.008 | 0.050 |
| FI | DK | 0 | 0.000 | 0.000 | 0.019 |
| DK | SE | 18 | 0.090 | 0.058 | 0.138 |
| DK | NO | 0 | 0.000 | 0.000 | 0.019 |
| DK | FI | 0 | 0.000 | 0.000 | 0.019 |
| AR | CL | 16 | 0.080 | 0.050 | 0.126 |
| CL | AR | 15 | 0.075 | 0.046 | 0.120 |
| ES | PT | 15 | 0.075 | 0.046 | 0.120 |
| PT | ES | 24 | 0.120 | 0.082 | 0.172 |
| NG | GH | 22 | 0.110 | 0.074 | 0.161 |
| GH | NG | 104 | 0.520 | 0.451 | 0.588 |
| US | CA | 7 | 0.035 | 0.017 | 0.070 |
| CA | US | 37 | 0.185 | 0.137 | 0.245 |
