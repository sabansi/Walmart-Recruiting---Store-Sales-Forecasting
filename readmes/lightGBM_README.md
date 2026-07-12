# LightGBM — Walmart Store Sales Forecasting

MLFlow experiments:

https://dagshub.com/sansi23/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/experiments/1/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

---

## LightGBM

LightGBM Gradient Boosting-ზე დაფუძნებული გადაწყვეტილების ხეების მოდელია. ის ქმნის ბევრ გადაწყვეტილების ხეს, სადაც ყოველი ახალი ხე წინა ხეების მიერ დაშვებული შეცდომების შემცირებას ცდილობს, ხოლო საბოლოო პროგნოზი ყველა ხის შედეგის გაერთიანებით მიიღება.

LightGBM ავირჩიეთ, რადგან Walmart-ის მონაცემები კლასიკური ტაბულარული ფორმატისაა და შეიცავს ერთმანეთისგან განსხვავებული ტიპის ბევრ feature-ს:

- მაღაზიისა და დეპარტამენტის იდენტიფიკატორები;
- კალენდარული ნიშნები და holiday-ებთან დაკავშირებული ნიშნები;
- Markdown ფასდაკლებები და ეკონომიკური მაჩვენებლები (`CPI`, `Unemployment`);
- მაღაზიის ტიპი და ზომა;
- ისტორიულ გაყიდვებზე დაფუძნებული lag და rolling feature-ები.

LightGBM განსაკუთრებით ეფექტურია დიდ tabular dataset-ებზე — სწრაფად სწავლობს, კარგად ამუშავებს missing მნიშვნელობებს და აღმოაჩენს რთულ არაწრფივ დამოკიდებულებებს. ამ ამოცანაში LightGBM პირდაპირ time-series მოდელი არ არის, ამიტომ დროითი ინფორმაცია მოდელს ხელით შექმნილი feature-ების საშუალებით გადავეცით.

---

## როგორ მუშაობს LightGBM

Gradient Boosting-ის იდეაა, რომ ყოველი ახალი ხე არსებული მოდელის residual-ს (შეცდომას) სწავლობს. თუ წინა ეტაპის პროგნოზია \(\hat{y}^{(m-1)}\), მაშინ ახალი ხე სწავლობს დარჩენილ შეცდომას:

\[
r_i^{(m)} = y_i - \hat{y}_i^{(m-1)}
\]

განახლებული პროგნოზი: \(\hat{y}^{(m)} = \hat{y}^{(m-1)} + \eta f_m(X)\), სადაც \(f_m(X)\) ახალი ხეა, \(\eta\) — learning rate, \(m\) — boosting iteration.

LightGBM-ის მთავარი თავისებურებაა **leaf-wise tree growth**: level-wise ზრდის ნაცვლად, ის ყოველ ნაბიჯზე ყოფს იმ leaf-ს, რომლის გაყოფაც loss-ს ყველაზე მეტად ამცირებს. ეს ხშირად უკეთეს სიზუსტეს იძლევა, მაგრამ ზრდის overfitting-ის რისკს, ამიტომ მნიშვნელოვანია ისეთი პარამეტრების კონტროლი, როგორებიცაა `num_leaves`, `min_child_samples`, `learning_rate`, `reg_alpha`, `reg_lambda`, `feature_fraction`, `bagging_fraction`.


## EDA

Exploratory Data Analysis-ის დროს გამოვლინდა შემდეგი კანონზომიერებები:

1. `Store` და `Dept` კომბინაციებს შორის გაყიდვების მასშტაბი მნიშვნელოვნად განსხვავდება.
2. `Type A` მაღაზიებს საშუალოდ უფრო მაღალი გაყიდვები აქვთ, ვიდრე `Type B` და `Type C`.
3. წლის ბოლოს, განსაკუთრებით Thanksgiving-სა და Christmas-ის პერიოდში, გაყიდვების მკვეთრი ზრდაა.
4. მონაცემებში ძლიერი 52-კვირიანი სეზონურობაა.
5. Markdown მონაცემების დიდი ნაწილი missing-ია და მათი ისტორია მხოლოდ გვიან პერიოდში იწყება.
6. `Weekly_Sales` right-skewed განაწილებას შეიცავს, იშვიათი ძალიან მაღალი spike-ებით.
7. უარყოფითი გაყიდვები რეალურ returns-ს შეიძლება ასახავდეს, ამიტომ training-იდან არ წავშალეთ.

| EDA დაკვირვება | მოდელირების გადაწყვეტილება |
|---|---|
| Store/Dept წყვილებს განსხვავებული მასშტაბი აქვთ | `Store` და `Dept` კატეგორიულ feature-ებად |
| ძლიერი წლიური სეზონურობა | `Sales_Lag52_Origin` მთავარ anchor-ად |
| Thanksgiving/Christmas spike-ები | holiday flags და holiday proximity feature-ები |
| Holiday შეცდომა ხუთჯერ ძვირია | training/validation/early stopping-ში holiday row-ებს წონა 5 |
| Markdown-ებში ბევრი missing | 0-ით შევსება + `MarkDown_Total`, `MarkDown_Count` |
| კალენდარული ცვლადები ციკლურია | `Week_Sin`, `Week_Cos`, `Month_Sin`, `Month_Cos` |
| Target-ში მაღალი spike-ებია | სრული level-ის ნაცვლად seasonal baseline correction |
| იშვიათმა extreme correction-ებმა შეიძლება WMAE გააფუჭოს | residual prediction-ები robust quantile-ებით შეზღუდვა |

---

## Data Cleaning

- `Date` გარდაიქმნა datetime ფორმატში.
- `train.csv`, `test.csv`, `features.csv` და `stores.csv` გაერთიანდა `Store`, `Date` და `IsHoliday` სვეტებით.
- Markdown feature-ების missing მნიშვნელობები 0-ით შეივსო, უარყოფითი Markdown-ები 0-ზე შეიზღუდა.
- `CPI` და `Unemployment` შეივსო მაღაზიის შიგნით forward/backward fill-ით.
- მაღაზიის `Type` გარდაიქმნა `Type_Enc`-ად; უარყოფითი `Weekly_Sales` training-ში შენარჩუნდა.
- test მონაცემების საწყისი რიგის შესანარჩუნებლად დაემატა **`_input_row`**.

`_input_row` განსაკუთრებით მნიშვნელოვანი იყო, რადგან feature engineering-ის ზოგიერთი ეტაპი DataFrame-ს `Store`, `Dept` და `Date` მიხედვით ალაგებდა.

---

## Feature Engineering

| ჯგუფი | მაგალითები |
|---|---|
| Calendar | `Month`, `WeekOfYear`, `Quarter` |
| Cyclical calendar | `Week_Sin`, `Week_Cos`, `Month_Sin`, `Month_Cos` |
| Holiday | `IsSuperBowl`, `IsLaborDay`, `IsThanksgiving`, `IsChristmas` |
| Holiday proximity | `WeeksBefore_*`, `WeeksAfter_*` |
| Store | `Store`, `Dept`, `Type_Enc`, `Size` |
| External | `Temperature`, `Fuel_Price` |
| Markdown | `MarkDown1`–`MarkDown5`, `MarkDown_Total`, `MarkDown_Count` |
| Forecast position | `Forecast_Horizon`, `Horizon_Sin`, `Horizon_Cos` |
| Historical origin | `Sales_Lag52_Origin`, origin rolling mean/std/min/max |
| Baseline | `Seasonal_Baseline`, `Lag52_Missing` |

**რატომ ამოვიღეთ `Year`, `CPI` და `Unemployment`.** 
Kaggle-ის test period შეიცავს 2013 წელს, რომელიც თრენინგის მონაცემებში არ გვხვდებოდა — `Year` tree model-ს categorical-like split-ებით შეეძლო დაემუშავებინა და unseen 2013-ზე არასტაბილური extrapolation გაეკეთებინა. `CPI` და `Unemployment` test horizon-ის ნაწილისთვის თავდაპირველად მიუწვდომელია, ხოლო მათი 
imputation-ი train/test distribution mismatch-ს გაზრდიდა. ამის ნაცვლად კალენდარული სეზონურობა უფრო სტაბილური periodic encoding-ებით წარმოვადგინეთ:

\[
WeekSin=\sin\left(2\pi\frac{WeekOfYear}{52}\right), \qquad
WeekCos=\cos\left(2\pi\frac{WeekOfYear}{52}\right)
\]

მსგავსად შეიქმნა `Month_Sin`, `Month_Cos`, `Horizon_Sin` და `Horizon_Cos`.

---

## რატომ არ გამოვიყენეთ ჩვეულებრივი short lag-ები

საწყის მოდელში გამოყენებული იყო `Sales_Lag1`, `Sales_Lag2`, `Sales_Lag4`, `Sales_Lag13`, `Sales_Roll_4w_Mean`, `Sales_Roll_13w_Mean`. Local validation-ზე ამ მოდელმა დაახლოებით **1,500 WMAE** მიიღო, თუმცა Kaggle-ზე შედეგი დაახლოებით **25,000 WMAE** გახდა.

Short lag-ების მთავარი პრობლემაა, რომ 39-კვირიან direct forecast-ში მომავალი კვირების ნამდვილი გაყიდვები არ ვიცით. მაგალითად, test horizon-ის მე-20 კვირის `Lag1` არის მე-19 test კვირის ნამდვილი გაყიდვა — Kaggle-ის prediction მომენტში ეს მნიშვნელობა ხელმისაწვდომი არ არის. თუ validation frame-ზეც lag feature-ები ნამდვილი validation target-ებით იქმნება, მოდელი იღებს ინფორმაციას, რომელიც რეალურ test inference-ში არ ექნება — შედეგად local validation ზედმეტად ოპტიმისტური ხდება.

---

## Origin-style feature contract

პრობლემის გადასაჭრელად training, validation და test მონაცემები ერთი forecast-origin ლოგიკით შეიქმნა:

1. გამოიყენება მხოლოდ origin date-მდე არსებული გაყიდვების ისტორია;
2. forecast horizon არის 39 კვირა;
3. horizon-ის არც ერთი row არ იყენებს სხვა future row-ის ნამდვილ `Weekly_Sales` მნიშვნელობას;
4. rolling statistics origin-ზე ერთხელ ითვლება და მთელი 39 კვირის განმავლობაში უცვლელია;
5. historical feature-ები validation და Kaggle test-ზე ერთნაირი წესით იქმნება.

Training frame იქმნება `build_origin_training_frame()` ფუნქციით, ხოლო validation და test frame — `add_origin_style_features()`
ფუნქციით. ეს უზრუნველყოფს, რომ `training information contract = validation information contract = Kaggle test information contract`.

---

## ზუსტი Lag-52

თავდაპირველი feature helper ზოგან იყენებდა `groupby(["Store", "Dept"])["Weekly_Sales"].shift(52)`-ს — ეს 52 **row**-ით გადაადგილებას
ნიშნავს და არა ყოველთვის ზუსტად 52 **კვირით**. თუ რომელიმე Store–Dept სერიაში კვირა აკლია, 52-ე წინა row შეიძლება წინა წლის შესაბამის თარიღს არ ემთხვეოდეს.

ამიტომ მთავარ seasonal anchor-ად გამოვიყენეთ date-based lookup: \(Lag52(Date)=WeeklySales(Date-364\ days)\). ტექნიკურად ისტორიული `Date` 
52 კვირით წინ გადაიწია (`lag_history["Date"] += pd.Timedelta(weeks=52)`) და შემდეგ merge შესრულდა. თუ ზუსტი Lag-52 მნიშვნელობა არ არსებობს,
fallback-ად გამოიყენება: 52-კვირიანი rolling mean → 26-კვირიანი → 13-კვირიანი → გლობალური მედიანა.

---

## Seasonal baseline

Walmart-ის მონაცემებში წინა წლის შესაბამისი კვირის გაყიდვა ძალიან ძლიერი baseline-ია: \(Baseline_t=Sales_{t-52}\), ხოლო Lag-52-ის missing-ის შემთხვევაში — 
rolling mean fallback-ები. ეს მნიშვნელობა feature frame-ში ინახება როგორც `Seasonal_Baseline`, ასევე დაემატა binary feature `Lag52_Missing`, რათა მოდელმა იცოდეს, 
baseline რეალური წინა წლის მნიშვნელობაა თუ fallback-ით მიღებული შეფასება.

---

## Residual-based LightGBM

საბოლოო მოდელი პირდაპირ `Weekly_Sales` level-ს აღარ პროგნოზირებს. Target გახდა seasonal baseline-ისგან განსხვავება:

\[
ResidualTarget_t = WeeklySales_t - SeasonalBaseline_t
\]

LightGBM სწავლობს \(\widehat{Residual}_t = LightGBM(X_t)\)-ს, საბოლოო პროგნოზი კი: \(\widehat{WeeklySales}_t = SeasonalBaseline_t + \alpha\widehat{Residual}_t\), სადაც \(\alpha\) validation-ზე შერჩეული correction strength-ია.

**რატომ არის ეს მიდგომა უკეთესი.** სრული level-ის პროგნოზირებისას მოდელს ერთდროულად უწევს Store–Dept მასშტაბის, წლიური სეზონურობის, holiday spike-ების და Markdown/context ეფექტების სწავლა. Residual მიდგომაში annual seasonality უკვე baseline-შია ჩადებული, LightGBM-ს მხოლოდ კორექციის სწავლა რჩება — მაგალითად, წინა წელთან შედარებით Markdown-ის გავლენა, მაღაზიის ქცევის ცვლილება, trend-ის ცვლილება, holiday-ის განსხვავებული ეფექტი, temperature და fuel price-ის შესაძლო გავლენა.

---

## Correction blending და clipping

LightGBM-ის correction ყოველთვის სრულად სანდო არ არის, ამიტომ validation-ზე შემოწმდა 25 თანაბრად დაშორებული \(\alpha \in [0,1.2]\): \(\alpha=0\) → საბოლოო პროგნოზი მთლიანად seasonal baseline; \(\alpha=1\) → LightGBM-ის შესწორება სრულად გამოიყენება; \(0<\alpha<1\) → correction shrinkage-ით გამოიყენება. საბოლოოდ ირჩევა ის \(\alpha\), რომელიც validation WMAE-ს მინიმუმამდე ამცირებს. ვინაიდან `alpha=0` (სუფთა baseline) ყოველთვის candidate-ია, შერჩეული blend seasonal baseline-ზე უარესი ვერასდროს იქნება.

იშვიათი high-error row-ების დროს LightGBM-ს შეიძლებოდა ძალიან დიდი correction გამოეთვალა — ამის შესამცირებლად final training residual-ებიდან გამოვთვალეთ 0.25% და 99.75% quantile-ები, და inference-ის დროს correction ამ საზღვრებში შეიზღუდა: \(Correction=clip(\widehat{Residual}, q_{0.0025}, q_{0.9975})\). ეს guardrail ამცირებს ერთი ან რამდენიმე extreme პროგნოზის მიერ მთლიანი WMAE-ის გაფუჭების რისკს.

---

## Validation სტრატეგია

Random split არ გამოგვიყენებია. Training მონაცემების ბოლო **39 უნიკალური კვირა** მთლიანად გამოიყო ვალიდაციის ბლოკად — ეს ზუსტად ემთხვევა Kaggle test horizon-ის სიგრძეს.

1. განისაზღვრა ბოლო 39 კვირა;
2. validation origin გახდა ამ block-მდე ბოლო ცნობილი კვირა;
3. historical feature-ები აშენდა მხოლოდ validation origin-მდე არსებული გაყიდვებით;
4. training origins შეიქმნა მხოლოდ იმ შემთხვევებისთვის, რომელთა სრული 39-კვირიანი target horizon validation-მდე სრულდებოდა;
5. validation-ის 39 კვირა ერთ direct forecast block-ად შეფასდა.

ამით თავიდან ავიცილეთ random split leakage, overlapping validation windows-ის არასწორი ოპტიმისტური შედეგი, validation target-ების lag feature-ებში მოხვედრა და Kaggle-ისგან განსხვავებული მოკლე horizon.

---

## WMAE-ს შესაბამისი სწავლება

კონკურსის შეფასების მეტრიკაა:

\[
WMAE=\frac{\sum_i w_i|y_i-\hat{y}_i|}{\sum_i w_i}, \qquad
w_i=\begin{cases}5,&\text{holiday week}\\1,&\text{ordinary week}\end{cases}
\]

მოდელის დანაკარგის ფუნქცია იყო `regression_l1`, ხოლო training sample weight — `train_weights = np.where(IsHoliday, 5.0, 1.0)`. Validation და early stopping-ის დროსაც holiday row-ებს წონა 5 ჰქონდათ, რადგან ჩვეულებრივი MAE-ს ოპტიმიზაციამ შეიძლება საშუალო კვირები კარგად იწინასწარმეტყველოს, მაგრამ Black Friday და Christmas spike-ებზე დიდი შეცდომა დაუშვას.

---

## Hyperparameter Tuning

Hyperparameter search განზრახ მცირე grid-ზე ჩავატარეთ, რათა ექსპერიმენტები დროის მხრივ მართვადი ყოფილიყო:

| პარამეტრი | მნიშვნელობები |
|---|---|
| `num_leaves` | 31, 63 |
| `learning_rate` | 0.02, 0.04 |
| `min_child_samples` | 50, 100 |
| `reg_lambda` | 1.0, 5.0 |

ყველა მოდელისთვის ფიქსირებული იყო: `objective="regression_l1"`, `n_estimators=2500`, `reg_alpha=0.1`, `max_depth=-1`, `max_bin=255`, `feature_fraction=0.85`, `bagging_fraction=0.85`, `bagging_freq=1`, `early_stopping_rounds=100`.

თითოეული candidate-ისთვის: LightGBM residual target-ს სწავლობდა → early stopping holiday-weighted validation MAE-ს იყენებდა → residual prediction sales level-ად გარდაიქმნებოდა → შეირჩეოდა საუკეთესო correction alpha → საბოლოო candidate selection ხდებოდა WMAE-ით. შედეგები ინახება `best_params`, `best_iteration`, `best_alpha`, `best_score`-ში, საიდანაც საბოლოო მოდელი ყველა ხელმისაწვდომ historical origin-ზე თავიდან სწავლობს.

---

## მთავარი სირთულეები და გამოსწორებები

**1. Local validation ~1,500, Kaggle ~25,000.** ასეთი დიდი სხვაობა ჩვეულებრივ მხოლოდ model generalization-ით არ აიხსნება — ძირითადი მიზეზი იყო submission row alignment. Feature engineering-ის დროს DataFrame რამდენჯერმე ლაგდებოდა (`sort_values(["Store", "Dept", "Date"])`), pipeline პროგნოზებს უკვე გადალაგებულ frame-ზე ითვლიდა, მაგრამ შედეგს თავდაპირველი input index-ით აბრუნებდა ისე, რომ row order აღარ აღდგებოდა. შედეგად ერთი `(Store, Dept, Date)` row-ის პროგნოზი სხვა row-ზე ხვდებოდა (მაგ. `Store 1/Dept 1/Date A`-ს პროგნოზი `Store 20/Dept 72/Date B`-ს უკავშირდებოდა). Validation frame-ზე target და features ერთად იყო გადალაგებული, ამიტომ local metric პრობლემას ვერ ამჩნევდა — Kaggle submission-ში კი prediction-ები არასწორ `Id`-ებს უკავშირდებოდა.

**გამოსწორება:** test frame-ს feature engineering-მდე დაემატა `_input_row`. ყველა transformation-ის შემდეგ:
```python
feat = feat.sort_values("_input_row")
assert np.array_equal(
    feat["_input_row"].to_numpy(),
    np.arange(len(input_df))
)
```

**2. Validation არ ჰგავდა Kaggle-ის რეალურ ამოცანას.** საწყისი validation იყენებდა მოკლე ან გადაფარულ windows-ს, ხოლო Kaggle ითხოვს ერთიან 39-კვირიან პროგნოზს. Overlapping windows-ზე მოდელი ბევრ ერთმანეთის მსგავს მაგალითს ხედავდა და local score ხელოვნურად ოპტიმისტური ხდებოდა. **გამოსწორება:** ბოლო 39 კვირა მთლიანად validation block-ად გამოიყო; არც ერთი training origin-ის target horizon არ კვეთს ამ პერიოდს.

**3. Short lag leakage.** `Lag1`, `Lag2` და სხვა short lag feature-ები validation row-ებისთვის ნამდვილ წინა validation target-ებს იყენებდა, რაც რეალურ 39-კვირიან test horizon-ზე უცნობია. **გამოსწორება:** final feature set-ში გაყიდვების ისტორია მხოლოდ forecast-origin feature-ებით გამოიყენება; მთავარი anchor Lag-52-ია, ხოლო rolling statistics origin-ზე იყინება.

**4. Row-based `shift(52)`.** Sparse series-ზე ყოველთვის ზუსტად წინა წლის კვირას არ ნიშნავს. **გამოსწორება:** Lag-52-ის მთავარი მნიშვნელობა exact date merge-ით გამოითვლება.

**5. სრული sales level-ის არასტაბილური პროგნოზირება.** LightGBM ზოგიერთ holiday spike-ს მნიშვნელოვნად აკლებდა ან ზოგიერთ სერიაზე ზედმეტ პროგნოზს აკეთებდა. **გამოსწორება:** მოდელი გადავიყვანეთ residual formulation-ზე (Lag-52 baseline + validation-selected correction).

**6. Extreme residual correction.** რამდენიმე ძალიან დიდი correction მთლიან WMAE-ს მნიშვნელოვნად ზრდიდა. **გამოსწორება:** correction robust quantile-ებით შეიზღუდა.

**7. Partial-sample pipeline test-ის შეუსაბამობა.** Registered pipeline-ის მხოლოდ 1,000 შემთხვევით row-ზე შემოწმებისას რამდენიმე პროგნოზი full-test prediction-ს ზუსტად არ ემთხვეოდა — ზოგიერთი feature engineering ოპერაცია forecast frame-ის სრულ კონტექსტზე იყო დამოკიდებული, ხოლო full test-ზე და subset-ზე sparse series-ის fallback პირობები განსხვავდებოდა. **გამოსწორება:** საბოლოო equivalence test სრულ test frame-ზე უნდა შესრულდეს:
```python
loaded_predictions = loaded_pipeline.predict(full_test_input)
np.testing.assert_allclose(
    loaded_predictions.to_numpy(),
    direct_full_test_predictions,
)
```


## Feature Importance

<img width="1120" height="1260" alt="image" src="https://github.com/user-attachments/assets/9a343c0e-2705-4a55-9d56-e9f72c83455c" />


გრაფიკზე ყველაზე მაღალი მნიშვნელობა აქვს `Dept`-სა და `Store`-ს, რაც აჩვენებს, რომ გაყიდვების დონე ძლიერ არის დამოკიდებული კონკრეტულ Store–Department კომბინაციაზე. შემდეგ მოდის `WeekOfYear`, `Seasonal_Baseline` და `Sales_Lag52_Origin`, რაც ადასტურებს წლიური სეზონურობის მნიშვნელობას. Rolling საშუალოები დამატებით აღწერს სერიის ჩვეულებრივ დონეს, ხოლო Markdown და გარე ეკონომიკური feature-ები შედარებით მცირე, თუმცა დამხმარე როლს ასრულებს.

Feature importance მიზეზობრივ გავლენას არ ნიშნავს — იგი მხოლოდ აჩვენებს, რამდენად ხშირად და ეფექტურად გამოიყენა LightGBM-მა feature-ები ხეების split-ებში.

---

## Validation Diagnostics

საუკეთესო მოდელისთვის ცალკე იქმნება weekly WMAE ცხრილი, რომელიც გვაძლევს საშუალებას დავინახოთ: რომელ კვირებში იყო ყველაზე დიდი შეცდომა, holiday კვირებზე რამდენად სტაბილურია მოდელი, იზრდება თუ არა prediction error forecast horizon-ის ბოლოს, და რომელი Store–Dept წყვილები ქმნის ყველაზე დიდ შეცდომას. განსაკუთრებული ყურადღება ექცევა Thanksgiving, Christmas, Super Bowl, Labor Day და test horizon-ის ბოლო კვირებს — მხოლოდ საერთო WMAE-მ შეიძლება დამალოს ერთი ან ორი ძალიან ცუდი holiday პერიოდი.

---

## საბოლოო მოდელის სწავლება

Hyperparameter tuning-ის შემდეგ საბოლოო მოდელი თავიდან სწავლობს ყველა historical origin-ზე, რომელსაც სრული 39-კვირიანი target horizon აქვს. Final training იყენებს tuning-ით შერჩეულ `best_params`-ს და `best_iteration`-ს, validation-ზე შერჩეულ `best_alpha`-ს, holiday sample weights-ს, exact Lag-52 seasonal baseline-ს, residual target-ს და robust residual clip boundaries-ს. Final model პირდაპირ `Weekly_Sales` level-ს არ ინახავს target-ად — ის სწავლობს seasonal baseline-ის correction-ს.

---

MLflow-ში ინახება: hyperparameter candidate-ები, validation WMAE, best iteration, correction alpha, Lag-52 baseline WMAE, residual clipping საზღვრები,
feature list, model artifact, history artifact, inference configuration და registered pyfunc pipeline. საბოლოო model registry name არის `WalmartLightGBMFixedPipeline`.

---

## საბოლოო Pipeline და Inference

საბოლოო pipeline:

```text
Raw merged Walmart test dataframe
        -> preserve _input_row
        -> base feature engineering
        -> origin-style historical features
        -> exact date-based Lag-52
        -> cyclical calendar features
        -> seasonal baseline
        -> LightGBM residual correction
        -> correction clipping and alpha blending
        -> restore original row order
        -> Weekly_Sales predictions
```

Pipeline-ის არტეფაქტებია: `lgbm_fixed_model.joblib`, `lgbm_history.parquet`, `lgbm_fixed_config.json`. საბოლოო submission ფაილია `submission_lightgbm_fixed.csv`

---

**გამოყენებული საშუალებები:** Python, Pandas, NumPy, LightGBM, Scikit-learn, Matplotlib, MLflow, Joblib, DagsHub, Kaggle.

---

## Kaggle-ის საბოლოო შედეგი

ფაილი: `submission_lightgbm_fixed.csv`

| ექსპერიმენტი / შედეგი | WMAE |
|---|---:|
| საწყისი local validation (short-lag) | ~1,500–1,800 |
| საწყისი Kaggle submission (row-order ბაგი) | ~25,000 |
| **Public leaderboard (fixed)** | **2958.04691** |
| **Private leaderboard (fixed)** | **2875.34825** |

Private score Public score-ზე დაახლოებით **82.70 პუნქტით** უკეთესია. Leaderboard-ის ორ ნაწილს შორის ასეთი მცირე განსხვავება მიუთითებს, რომ fixed LightGBM pipeline შედარებით სტაბილურად ზოგადდებოდა. დაახლოებით 25,000-იანი საწყისი Kaggle score არ გამოხატავდა LightGBM-ის რეალურ შესაძლებლობას — მთავარი პრობლემა submission row-order bug იყო, არა მოდელის არქიტექტურა. ამ შედეგით LightGBM-მა (Public WMAE 2958.05) N-BEATS-საც (3513.18) აჯობა, რაც ადასტურებს, რომ ძლიერი feature engineering, ზუსტი Lag-52 seasonal baseline და leakage-safe validation ხის მოდელისთვის ამ dataset-ზე ძალიან ეფექტურია.

<img width="1373" height="108" alt="image" src="https://github.com/user-attachments/assets/ba9d2983-0dc6-4cf7-b50d-8f0a634e0fdc" />

---
---

## LightGBM-ის უპირატესობები

- სწრაფად სწავლობს დიდ tabular dataset-ზე;
- ეფექტურად იყენებს numerical და categorical feature-ებს;
- კარგად მუშაობს რთულ არაწრფივ დამოკიდებულებებზე;
- missing მნიშვნელობების დამუშავება შეუძლია;
- early stopping ამცირებს ზედმეტი iteration-ების რისკს;
- feature importance ინტერპრეტაციის საშუალებას იძლევა;
- residual/seasonal baseline-თან მარტივად ერთიანდება;
- neural network მოდელებთან შედარებით ნაკლებ computational რესურსს მოითხოვს.

---

## LightGBM-ის შეზღუდვები

- პირდაპირ time-series არქიტექტურა არ არის;
- მთლიანად feature engineering-ის ხარისხზეა დამოკიდებული;
- short lag-ების არასწორმა გამოყენებამ leakage შეიძლება გამოიწვიოს;
- leaf-wise growth ზედმეტად რთულ ხეებზე overfitting-ის რისკს ზრდის;
- ძლიერი holiday spike-ების სწავლა მხოლოდ საშუალო MAE-ით რთულია;
- prediction pipeline-ის row alignment ცალკე უნდა შემოწმდეს;
- ერთი 39-კვირიანი validation block შეიძლება ყველა შესაძლო სეზონურ რეჟიმს არ მოიცავდეს;
- exact Lag-52 missing-ის შემთხვევაში fallback baseline ნაკლებად ზუსტია.

---

## შესაძლო გაუმჯობესებები

1. რამდენიმე rolling-origin 39-კვირიანი validation fold;
2. holiday-specific residual model;
3. Store–Dept მიხედვით correction clipping;
4. forecast horizon-ის მიხედვით ცალკე model ან parameters;
5. quantile regression და uncertainty interval-ები;
6. Optuna hyperparameter tuning;
7. LightGBM + CatBoost ensemble;
8. LightGBM + TiDE ან TimeXer ensemble;
9. Store–Dept hierarchical aggregate feature-ები;
10. holiday-specific lag feature-ები;
11. exact calendar-based historical joins ყველა lag-ისთვის;
12. SHAP analysis რთული prediction-ების ასახსნელად.

---

## დასკვნა

LightGBM Walmart-ის ყოველკვირეული გაყიდვების პროგნოზირებისათვის ძლიერი და პრაქტიკული მოდელია, მაგრამ მისი წარმატება
მნიშვნელოვნად არის დამოკიდებული სწორ feature engineering-სა და მონაცემთა გაჟონვისგან დაცულ ვალიდაციაზე.

საწყის მოდელს local validation-ზე დაახლოებით 1,500 WMAE ჰქონდა, თუმცა Kaggle-ზე დაახლოებით 25,000 მიიღო — 
ასეთი განსხვავების მთავარი მიზეზი submission row alignment-ის შეცდომა იყო: feature engineering-ის შემდეგ პროგნოზები საწყის test row-ებს სწორად აღარ უკავშირდებოდა.

საბოლოო მიდგომაში: validation 39 კვირაზე გადავიყვანეთ, training origins validation block-მდე შევზღუდეთ, short future-dependent lag feature-ები ამოვიღეთ,
exact Lag-52 seasonal baseline გამოვიყენეთ, LightGBM-ს residual correction ვასწავლეთ, holiday row-ებს წონა 5 მივეცით, correction alpha validation-ზე შევარჩიეთ, 
extreme correction-ები შევზღუდეთ, test row order `_input_row`-ით აღვადგინეთ და pipeline MLflow Model Registry-ში დავარეგისტრირეთ.

Fixed LightGBM submission-ის Kaggle შედეგები იყო Public WMAE **2958.04691** და Private WMAE **2875.34825** — ეს ადასტურებს, რომ საწყისი დაახლოებით 
25,000-იანი შედეგი მართლაც pipeline-ის შეცდომით იყო გამოწვეული და არა LightGBM-ის არქიტექტურის სისუსტით. ძლიერი feature engineering, 
ზუსტი Lag-52 seasonal baseline და leakage-safe validation tree-based მოდელისთვის ამ dataset-ზე ძალიან ეფექტურია.
