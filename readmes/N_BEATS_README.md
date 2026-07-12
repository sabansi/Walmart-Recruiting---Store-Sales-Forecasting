# N-BEATS — Walmart Store Sales Forecasting

MLFlow experiments:

https://dagshub.com/sansi23/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/5/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

## პროექტის აღწერა

ეს პროექტი შექმნილია Kaggle-ის კონკურსისთვის **Walmart Recruiting — Store Sales Forecasting**.  
ამოცანის მიზანია Walmart-ის თითოეული `(Store, Dept)` დროითი მწკრივისთვის მომავალი ყოველკვირეული გაყიდვების პროგნოზირება.

მოდელად გამოყენებულია **N-BEATS** — სრულად მრავალშრიან პერცეპტრონებზე დაფუძნებული ღრმა სწავლების არქიტექტურა, რომელიც დროითი მწკრივის 
პროგნოზირებისთვის არ იყენებს არც რეკურენტულ ფენებს და არც attention მექანიზმს.

N-BEATS ამ პროექტში თავიდან არის იმპლემენტირებული **PyTorch**-ით.

---

## Kaggle-ის შედეგი

ფაილი: `submission_nbeats.csv`

| შედეგი | WMAE |
|---|---:|
| Public leaderboard | **3513.17761** |
| Private leaderboard | **3453.97720** |

Kaggle-ის შეფასებისას **ნაკლები WMAE უკეთეს შედეგს ნიშნავს**.

---

## შეფასების მეტრიკა

კონკურსში გამოყენებულია **Weighted Mean Absolute Error — WMAE**:

\[
WMAE =
\frac{
\sum_i w_i |y_i-\hat{y}_i|
}{
\sum_i w_i
}
\]

სადაც:

\[
w_i =
\begin{cases}
5, & \text{თუ კვირა სადღესასწაულოა} \\
1, & \text{სხვა შემთხვევაში}
\end{cases}
\]

სადღესასწაულო კვირებში დაშვებულ შეცდომას ხუთჯერ მეტი გავლენა აქვს საბოლოო ქულაზე.

---

## N-BEATS-ის არქიტექტურა

N-BEATS შედგება რამდენიმე ერთმანეთის მიმდევრობით განლაგებული block-ისგან.

თითოეული block იღებს წინა block-ის residual-ს და ქმნის ორ გამოსავალს:

- **Backcast** — შემავალი ისტორიის რეკონსტრუქცია;
- **Forecast** — მომავალი პერიოდის პროგნოზში block-ის წვლილი.

მოდელი იყენებს ე.წ. **doubly residual topology-ს**:

```text
Input
  │
  ▼
Block 1 ──► Forecast 1
  │
Residual 1
  ▼
Block 2 ──► Forecast 2
  │
Residual 2
  ▼
...
```

საბოლოო პროგნოზი მიიღება ყველა block-ის forecast-ის შეკრებით:

\[
\hat{y} =
\hat{y}^{(1)} +
\hat{y}^{(2)} + \cdots +
\hat{y}^{(B)}
\]

---

## გამოყენებული N-BEATS ვარიანტები

პროექტში შემოწმდა ორი ძირითადი არქიტექტურული მიდგომა.

### Generic N-BEATS

Generic ვერსიაში basis ფუნქციები წინასწარ შეზღუდული არ არის. მოდელი თვითონ სწავლობს, რა ფორმით წარმოადგინოს:

- ისტორიული სიგნალი;
- სეზონურობა;
- trend;
- პროგნოზის სხვადასხვა კომპონენტი.

### Interpretable N-BEATS

Interpretable ვერსიაში გამოყენებულია წინასწარ განსაზღვრული basis ფუნქციები:

- **Polynomial basis** — ტრენდისთვის;
- **Fourier basis** — სეზონურობისთვის.

ეს მიდგომა პროგნოზის trend და seasonal კომპონენტებად ინტერპრეტაციის საშუალებას იძლევა.

---


ძირითადი ცვლადებია:

- `Store` — მაღაზიის ნომერი;
- `Dept` — დეპარტამენტის ნომერი;
- `Date` — კვირის თარიღი;
- `Weekly_Sales` — სამიზნე ცვლადი;
- `IsHoliday` — სადღესასწაულო კვირის ინდიკატორი.

მიუხედავად იმისა, რომ notebook იტვირთავს `features.csv`-სა და `stores.csv`-ს, მიმდინარე N-BEATS არქიტექტურის უშუალო input არის მხოლოდ ისტორიული `Weekly_Sales`. `IsHoliday` გამოიყენება ვალიდაციის WMAE-ის გამოსათვლელად და არა მოდელის input-ად.

---

## მონაცემების მომზადება

### დროითი მწკრივების ფილტრაცია

თითოეული `(Store, Dept)` წყვილი განიხილება დამოუკიდებელ დროით მწკრივად.

მოკლე სერიები იფილტრება შემდეგი პირობით:

\[
SeriesLength \geq InputSize + Horizon + 5
\]

გამოყენებული მნიშვნელობებია:

```python
INPUT_SIZE = 52
HORIZON = 13
```

შესაბამისად, მოდელი პროგნოზის შესაქმნელად იყენებს წინა **52 კვირის ისტორიას** და ერთ forward pass-ში პროგნოზირებს მომდევნო **13 კვირას**.

Notebook-ში მიღებული მონაცემების სტატისტიკა:

| მახასიათებელი | მნიშვნელობა |
|---|---:|
| ყველა `(Store, Dept)` სერია | 3,323 |
| გამოყენებული სერია | 2,949 |
| გაფილტრული training rows | 413,346 |
| შექმნილი training windows | 186,987 |

### Sliding windows

თითოეული სერიიდან იქმნება ფანჯრები:

```text
52 კვირა input + 13 კვირა target
```

მაგალითად:

```text
[x₁, x₂, ..., x₅₂] → [y₁, y₂, ..., y₁₃]
```

სასწავლო ფუნქციაში გამოყენებულია `stride=2`, რაც ამცირებს ერთმანეთთან ძალიან მსგავს ფანჯრებს და აჩქარებს მოდელის სწავლებას.

---

## ნორმალიზაცია

თითოეული input window ნორმალიზდება თავისი საშუალო მნიშვნელობით:

\[
scale =
\max\left(
\frac{1}{52}\sum_{t=1}^{52}x_t,
1
\right)
\]

\[
x_t^{norm} = \frac{x_t}{scale}
\]

\[
y_t^{norm} = \frac{y_t}{scale}
\]

Inference-ის შემდეგ პროგნოზი ისევ საწყის მასშტაბზე ბრუნდება:

\[
\hat{y}_t =
\hat{y}_t^{norm}\cdot scale
\]

ეს მიდგომა ეხმარება ერთ global მოდელს ერთად ისწავლოს როგორც მცირე, ისე დიდი გაყიდვების მქონე Store–Department სერიები.

---

## სწავლების პროცესი

გამოყენებული ოპტიმიზაციის კომპონენტები:

- **Optimizer:** AdamW;
- **Loss:** L1 Loss / MAE ნორმალიზებულ სივრცეში;
- **Learning-rate scheduler:** Cosine Annealing;
- **Gradient clipping:** `max_norm=1.0`;
- **Random seed:** 42;
- **Hardware:** CUDA GPU, თუ ხელმისაწვდომია.

სასწავლო loss:

\[
L =
\frac{1}{N}
\sum_{i=1}^{N}
\left|
y_i^{norm} -
\hat{y}_i^{norm}
\right|
\]

მოდელის სწავლებისას უშუალოდ holiday-weighted loss არ გამოიყენება. Holiday weights გამოიყენება საბოლოო validation WMAE-ის დათვლისას.

---

## ვალიდაციის სტრატეგია

გამოყენებულია **walk-forward holdout**:

- ბოლო 13 უნიკალური კვირა გამოიყოფა ვალიდაციისთვის;
- ყველა უფრო ადრეული კვირა გამოიყენება სწავლებისთვის;
- თითოეული სერიის ბოლო 52 training კვირა გამოიყენება 13-კვირიანი პროგნოზის შესაქმნელად;
- პროგნოზი ფასდება WMAE-ით.

ეს სტრატეგია ინარჩუნებს დროით თანმიმდევრობას და არ იყენებს შემთხვევით train/validation გაყოფას.

---

## ექსპერიმენტები

### ძირითადი მოდელები

| ვარიანტი | Stack-ები | Validation WMAE |
|---|---|---:|
| Generic Baseline | `generic + generic` | 1489.53 |
| Interpretable | `trend + seasonality` | 1556.81 |

### Hyperparameter search

| ვარიანტი | კონფიგურაცია | Validation WMAE |
|---|---|---:|
| **Generic_256_shallow** | 2 generic stack, 2 block, 256 hidden units | **1472.15** |
| Generic_512_deep | 3 generic stack, 3 block, 512 hidden units | 1500.99 |
| Trend_Season_256 | trend + seasonality + generic | 1521.23 |

საუკეთესო validation შედეგი აჩვენა:

```text
Generic_256_shallow
Validation WMAE: 1472.15
```

მისი ძირითადი პარამეტრები იყო:

```python
stack_types = ("generic", "generic")
n_blocks_per_stack = 2
hidden_units = 256
n_layers = 2
batch_size = 512
learning_rate = 2e-3
weight_decay = 1e-4
epochs = 50
```

საბოლოო მოდელის სწავლებისას epoch-ების რაოდენობა გაიზარდა 70-მდე.

---

## ადგილობრივი და Kaggle-ის შედეგების განსხვავება

საუკეთესო ადგილობრივი validation WMAE იყო:

```text
1472.15
```

Kaggle-ის შედეგები კი იყო:

```text
Public WMAE:  3513.17761
Private WMAE: 3453.97720
```

ამ განსხვავებას რამდენიმე მიზეზი შეიძლება ჰქონდეს.

### 1. განსხვავებული forecast horizon

ადგილობრივი ვალიდაცია მოიცავს მხოლოდ 13 კვირას, ხოლო Walmart-ის Kaggle test პერიოდი 39 კვირას მოიცავს.
-რაც უფრო გრძელია forecast horizon, მით უფრო რთულია პროგნოზირება და მით უფრო იზრდება დაგროვილი შეცდომა.

### 2. მოდელი მხოლოდ historical sales-ს იყენებს

მიმდინარე N-BEATS მოდელი არ იყენებს შემდეგ ცნობილ future covariates-ს:

- holidays;
- markdowns;
- CPI;
- unemployment;
- temperature;
- fuel price;
- store type;
- store size.

ამიტომ მოდელს უჭირს იმ მომავალი ცვლილებების პროგნოზირება, რომლებიც მხოლოდ წარსული გაყიდვების ფორმიდან პირდაპირ არ ჩანს.

### 3. ერთი validation block

მოდელი ფასდება მხოლოდ ერთ 13-კვირიან holdout პერიოდზე. უფრო საიმედო შეფასებისთვის სასურველია რამდენიმე rolling-origin fold-ის გამოყენება.

### 4. Hyperparameter selection

რამდენიმე არქიტექტურა იმავე validation block-ზე შეირჩა. ამან შეიძლება validation პერიოდის მიმართ ნაწილობრივი overfitting გამოიწვიოს.

### 5. სერიების ფილტრაცია

მოკლე Store–Department სერიები training-იდან გამორიცხულია. ასეთი სერიებისთვის აუცილებელია საიმედო fallback პროგნოზი.

---

## მნიშვნელოვანი ტექნიკური შენიშვნები

### 39-კვირიანი Kaggle horizon

მიმდინარე `NBEATSPipeline.predict()` თითო `(Store, Dept)` ჯგუფისთვის პირდაპირ მხოლოდ პირველ 13 პროგნოზს ავსებს:

```python
n = min(len(group), self.horizon)
```

თუ test-ში თითო სერია 39 კვირას შეიცავს, submission-ის შექმნისას აუცილებელია:

1. მოდელის სამჯერ, rolling 13-week რეჟიმში გამოყენება; ან
2. `HORIZON = 39`-ზე მოდელის თავიდან სწავლება.

Submission-ის გენერაციამდე უნდა შემოწმდეს:

```python
assert len(predictions) == len(test_raw)
assert np.isfinite(predictions).all()
assert (predictions != 0).all()
```

ნულოვანი პროგნოზების დიდი რაოდენობა Kaggle-ის WMAE-ს მნიშვნელოვნად გააუარესებს.

### საბოლოო retraining-ის metric

Notebook-ში საუკეთესო კონფიგურაცია თავიდან ისწავლება სრულ `train_filtered` მონაცემებზე, რომელიც უკვე მოიცავს საწყის validation პერიოდს.

ამიტომ საბოლოო ეტაპზე დაბეჭდილი:

```text
Holdout WMAE: 3617.90
```

არ უნდა ჩაითვალოს სუფთა holdout შეფასებად. მოდელის არჩევისთვის გამოყენებული მთავარი local metric არის:

```text
Best validation WMAE: 1472.15
```

საბოლოო მოდელის რეალური გენერალიზაციის უნარი უკეთესად არის წარმოდგენილი Kaggle-ის Public და Private score-ის შედეგებით.

---

## MLflow tracking

ექსპერიმენტები ინახება MLflow-ში:

```text
Experiment: NBEATS_Training
```

ლოგირდება:

- hyperparameters;
- epoch-level training loss;
- learning rate;
- validation WMAE;
- მოდელის weights;
- forecast plots;
- loss curves;
- რეგისტრირებული PyTorch მოდელი.

საბოლოო მოდელი Model Registry-ში რეგისტრირდება სახელით:

```text
NBEATS_WalmartSales
```

---

---

## პროექტის სტრუქტურა

```text
.
├── model_experiment_NBEATS.ipynb
├── features.py
├── metrics.py
├── mlflow_setup.py
├── train.csv
├── test.csv
├── features.csv
├── stores.csv
├── nbeats_final.pt
├── nbeats_pipeline.pkl
├── nbeats_forecasts.png
├── nbeats_loss_curves.png
└── submission_nbeats.csv
```

---

## მოდელის უპირატესობები

- არ საჭიროებს ხელით შექმნილ lag features-ს;
- ყველა forecast step-ს ერთ forward pass-ში პროგნოზირებს;
- Generic არქიტექტურას შეუძლია რთული pattern-ების სწავლა;
- Interpretable არქიტექტურა გამოყოფს trend-სა და seasonality-ს;
- instance normalization საშუალებას აძლევს ერთ მოდელს იმუშაოს სხვადასხვა მასშტაბის სერიებზე;
- MLP არქიტექტურა მარტივად პარალელდება GPU-ზე.

---

## მოდელის შეზღუდვები

- მიმდინარე ვერსია univariate-ია;
- არ იყენებს future covariates-ს;
- 13-კვირიანი მოდელი პირდაპირ არ ფარავს 39-კვირიან Kaggle horizon-ს;
- მოკლე სერიები training-იდან გამორიცხულია;
- validation მხოლოდ ერთ დროით მონაკვეთს იყენებს;
- უფრო გრძელ horizon-ზე recursive ან block-wise პროგნოზირებისას შეცდომა შეიძლება გაიზარდოს.

---

## შესაძლო გაუმჯობესებები

1. მოდელის horizon-ის 39 კვირამდე გაზრდა;
2. N-BEATSx-ის გამოყენება exogenous covariates-ით;
3. holiday-weighted loss-ის გამოყენება უშუალოდ training-ში;
4. rolling-origin cross-validation;
5. lag-52 seasonal baseline-თან blending;
6. Store და Department embeddings;
7. fallback პროგნოზების გაუმჯობესება მოკლე სერიებისთვის;
8. early stopping;
9. model ensemble LightGBM, TiDE ან TimeXer მოდელებთან;
10. თითოეული forecast horizon-ისთვის ცალკე error analysis.

---

## დასკვნა

N-BEATS-მა Walmart-ის გაყიდვების პროგნოზირების ამოცანაზე აჩვენა ძლიერი შედეგი მხოლოდ ისტორიული გაყიდვების გამოყენებით.

საუკეთესო ადგილობრივი არქიტექტურა იყო მცირე Generic N-BEATS:

```text
Generic_256_shallow
Validation WMAE: 1472.15
```

Kaggle-ის საბოლოო შედეგები:

```text
Public WMAE:  3513.17761
Private WMAE: 3453.97720
```

Private და Public შედეგების მცირე განსხვავება მიუთითებს, რომ მოდელი leaderboard-ის ორ ნაწილზე შედარებით სტაბილურად მუშაობდა. ამავე დროს, 
local validation-სა და Kaggle-ის შედეგს შორის სხვაობა აჩვენებს, რომ მომავალ ექსპერიმენტებში საჭიროა validation horizon-ის Kaggle-ის 39-კვირიან 
test horizon-თან უკეთ შესაბამისობა და exogenous მახასიათებლების დამატება.
