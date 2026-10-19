# Presentation script

Speaker notes for the 19-slide deck ([slides as PDF](presentation-slides.pdf)), one section per slide, labelled with the roadmap stage each slide covers. About 1,316 words, roughly 9 minutes at a normal speaking pace. The silent [presentation video](presentation-video.mp4) (11 minutes) is organised by the eight roadmap stages: a title card opens each stage, then these notes run as captions over that stage's slides, with a short demo of the real `threshold` and `test` commands in stage 7.

## 1. Early Diabetes Risk Screening

Hi, we’re Sumit, Priyamvadha and Amiya. Our question: can the answers adults give on a health survey pick out people with diabetes or prediabetes, while missing as few of them as possible? We used the CDC’s 2015 BRFSS survey, about a quarter of a million adults, and followed an eight-stage roadmap. Each slide is labelled with its stage.

## 2. A screening tool should miss few cases without flagging everyone

A questionnaire costs almost nothing, so if it can tell us who should get a blood test, screening reaches people who would otherwise never be tested. Just under 16% of these adults report diabetes or prediabetes. A missed case is the expensive mistake, so we fixed the target at catching 80% of cases and compare approaches by how many people they must flag, against the paper risk test clinics already hand out.

## 3. Eight connected stages, each with its own checks

Here is the roadmap we followed. Stage one checks the data and stage two cleans it. Stages three and four explore it and test associations. Stage five sets up an honest evaluation, stage six trains and tunes the models, and stage seven chooses the screening cutoff and opens the test set once. Stage eight is this handoff. Every following slide carries its stage number.

## 4. 253,680 rows × 22 columns, checked; prediabetes counts as positive

Stage one, data ingestion. We loaded Kaggle’s version of the 2015 BRFSS and checked its shape, 253,680 rows by 22 columns, its column types, value ranges and labels, and recorded hashes so anyone can confirm the exact file. We used the three-class file because it separates prediabetes from diabetes, and we count both as positive: 15.8%. The roadmap’s 14% is diabetes alone. A sensitivity analysis looks at prediabetes by itself.

## 5. Nothing imputed or dropped; odd answers flagged, not changed

Stage two, cleaning. There were no missing or invalid codes, and the pipeline is built to stop rather than guess, so nothing was imputed or dropped. About 24,000 rows repeat someone else’s answers, but with 21 coarse answers different people often answer identically, so we kept them and tested their effect later. We sorted the answers into 14 yes/no, four ordered bands and three counts, and flagged unusual but valid answers, like a BMI over 60, without changing them.

## 6. One adult in six is positive; five answers carry the signal

Stage three, exploration. About one adult in six reports diabetes or prediabetes, so accuracy would mislead: always answering no scores 84%. People with the condition have a median BMI of 31 against 27 and are older. On the correlation heatmap, general health, high blood pressure, BMI, difficulty walking and high cholesterol stand out, and several of them move together, which is why we later adjust for them.

## 7. All 25 tests significant, so we read effect sizes

Stage four, hypothesis tests. We planned 25 tests: chi-square for the yes/no and ordered answers, and Mann–Whitney U for BMI, health days and the bands, with Benjamini–Hochberg correction. All 25 were significant, which with a quarter of a million people says little, so we report effect sizes with confidence intervals. High blood pressure goes with almost five times the odds; general health has the largest rank effect. The biggest odds ratio, for a cholesterol check, mostly measures contact with doctors.

## 8. Every decision used training rows; the test set stayed locked

Stage five, splitting and imbalance. We fixed one stratified split for the whole team: 80% for training, divided into five fixed folds, and 20% locked away; the code refuses to release the test rows until a cutoff is saved. Scaling happens inside the model pipelines, so it only learns from training folds. Repeated answers did not inflate the scores. And class weighting, compared with no weighting, didn’t rank people any better; it only inflated risks to 39%, so we kept honest probabilities.

## 9. Every tuned model beats the checklist; XGBoost wins every fold

Stage six, training and tuning. We tuned logistic regression as the interpretable baseline, a random forest and XGBoost, all on the same stratified folds, scored by PR-AUC. The big jump is from the paper checklist to any model: to catch 80% of cases the checklist flags half of everyone, the models 38 to 39%. Under the rule we wrote down first, XGBoost was chosen, winning all five folds. Logistic regression is close, if you need a model you can read line by line.

## 10. The same five answers drive every model

Still stage six: what do the models use? Shuffling one answer at a time, all three agree on BMI, general health, age, high blood pressure and high cholesterol. Separately, a statsmodels regression with all 21 answers gives adjusted odds ratios with intervals, and it warns against one-at-a-time numbers: bad physical-health days reverse once general health is included, smoking’s association disappears and exercise mostly works through BMI and health.

## 11. We locked the cutoff before opening the test set

Stage seven, evaluation. We chose the cutoff on training data only: the saved model was refitted on four folds at a time, so every training person got a risk from a model that never saw them. The rule, written before the run: the highest cutoff that still catches 80% of cases. That is a predicted risk of 0.153. Calibration was already excellent, so no adjustment. Each extra ten points of sensitivity would cost roughly 90 to 130 more people flagged per thousand.

## 12. Most cases found, far fewer false alarms than the checklist

The result on the 50,736 people we held back. At the locked cutoff the model found 79.4% of people with diabetes or prediabetes, right on target, and flagged 37.6% of people; one in three of those flagged has the condition. Against the paper checklist, which finds the same share, that’s 96 fewer false alarms per thousand people. Precision, specificity, F2 and both AUCs match validation, so nothing was overfitted.

## 13. Above the checklist at every cutoff, with honest risks

These are the curves the roadmap asks for. On both the precision-recall and the ROC curve, the two models sit above the paper checklist at every cutoff, and the dots mark each locked operating point. On the right is calibration: predicted risks on the test set match the observed rates, with a slope of 0.99, so the model’s probabilities can be read as real risks.

## 14. Per 1,000 screened: 125 cases found, 33 missed

In clinic terms: of every thousand people screened, about 158 have the condition. The model sends 376 for a blood test, finds 125 and misses 33; that’s the confusion matrix on the left. The paper checklist finds the same number but sends almost 100 more people, so its specificity is much lower. Logistic regression captures about 90% of the gain.

## 15. It misses younger and better-off people, and prediabetes

Now the false negatives. The model leans on BMI, general health, age, blood pressure and cholesterol, so a case who looks ordinary on all five is invisible to it; the missed cases have the same average BMI as people without the condition. Sensitivity is 44% for under-45s and 68% at incomes of 50 thousand dollars and up. And it finds 81% of diagnosed diabetes but only 65% of prediabetes, the group screening is really for.

## 16. For prediabetes alone, the signal is weak

We also asked the harder question: leave out everyone who already has diabetes and look for prediabetes alone, with the same pipeline and rules. The model still beats the checklist, finding 81% of cases against 70%, but it flags 44% of everyone and only 4% of those flagged report prediabetes. Many of them may be undiagnosed cases, which this data can’t show.

## 17. A research prototype, not a diagnostic tool

Stage eight includes being clear about limits. The labels are self-reported at one moment in 2015, so this finds people who already know their status; it predicts nothing about the future. 88% of positives already have a diagnosis. Some strong answers partly measure seeing a doctor. One cutoff gives different groups very different trade-offs. The survey is unweighted and covers one year. And everything here is association, not cause.

## 18. Data sources and reading

Our sources: the Kaggle and UCI versions of the data, the CDC’s MMWR article and NHANES portal, a historical NCHS report whose date should be checked before citing it as current, and the NIDDK risk test we used as the baseline.

## 19. Survey answers can find 8 in 10 cases with a fifth fewer people sent for testing than today’s paper checklist

To sum up: with nothing but survey answers, a tuned model finds about eight in ten people with diabetes or prediabetes while sending a fifth fewer people for testing than the paper checklist. It is weaker for prediabetes and younger people, and it is a research prototype. Every stage is reproducible, backed by 63 automated tests, and the code, notebooks, reports and technical PDF are in our repository. Thank you.
