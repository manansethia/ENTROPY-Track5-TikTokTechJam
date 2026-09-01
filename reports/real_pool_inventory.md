# Authentic High-Resolution & Portrait Dataset Inventory Report

## 1. Executive Summary & Quality Metrics
- **Total Valid Authentic Images Ingested**: **`4,128`**
- **Total Physical Storage**: **`0.865 GB`** (Zero Downsampling / Full Original High-Res Preserved)
- **High-Resolution (>1 MP) Images**: **`4,128`** (100.0%)
- **4K+ (>8 MP) Images**: **`9`** (0.2%)
- **Ultra High-Res (>16 MP / DSLR Raw)**: **`5`** (0.1%)
- **Strict Contamination Filter**: **0 overlap** with locked DEV split, `aigibench_eval`, `synthbuster`, and `wildfake`.

---

## 2. Aspect Ratio & Geometry Breakdown
| Aspect Ratio Category | Image Count | Percentage of Pool |
| :--- | :---: | :---: |
| **Portrait Orientation ($H > W$)** | **`23`** | **`0.56%`** |
| **Landscape Orientation ($W > H$)** | **`102`** | **`2.47%`** |
| **Square Geometry ($W \approx H$)** | **`4,003`** | **`96.97%`** |

---

## 3. Source Breakdown & Licensing Governance
| Source Repository | License & Terms | Image Count | Percentage |
| :--- | :--- | :---: | :---: |
| **`CelebA_HQ_Studio_Portraits`** | Open Research / CC-BY-SA | **`4,000`** | **`96.9%`** |
| **`ETH_Zurich_DIV2K_2K_HR`** | Open Research / CC-BY-SA | **`100`** | **`2.42%`** |
| **`Wikimedia_Commons_HighRes`** | Open Research / CC-BY-SA | **`28`** | **`0.68%`** |

---

## 4. Resolution Distribution
| Megapixel Tier | Count | Percentage |
| :--- | :---: | :---: |
| **`under_1mp`** | `0` | `0.00%` |
| **`1_to_2mp`** | `4,006` | `97.04%` |
| **`2_to_4mp`** | `109` | `2.64%` |
| **`4_to_8mp`** | `4` | `0.10%` |
| **`8_to_16mp_4k`** | `4` | `0.10%` |
| **`greater_than_16mp`** | `5` | `0.12%` |
