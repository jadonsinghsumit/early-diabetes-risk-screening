# Presentation script

Speaker notes for the 14-slide deck ([slides as PDF](presentation-slides.pdf)), one section per slide. About 1,528 words, roughly 9.5 to 10 minutes at a normal speaking pace. The [10-minute video](presentation-10min.mp4) shows these notes as captions over the slides; the [5-minute video](presentation-5min.mp4) uses a condensed version, and the 2-minute [demo video](demo-video.mp4) is a short summary. All three are silent.

## 1. Early Diabetes Risk Screening

Hi, we’re Sumit, Priyamvadha and Amiya. Our question is simple to say: can the answers adults give on a health survey, things like blood pressure, weight and how healthy they feel, pick out the people who have diabetes or prediabetes, while missing as few of them as possible? We used the CDC’s 2015 BRFSS survey, about a quarter of a million adults. In the next nine minutes: the data, how we kept the evaluation honest, the models, the screening cutoff we locked before we ever looked at the test set, and the final results, including who the model misses.

## 2. A screening tool should miss few cases without flagging everyone

A questionnaire costs almost nothing, so if it can tell us who should get a blood test, screening can reach people who would never otherwise be tested. We had about 254,000 adults, each with 21 answers: blood pressure, cholesterol, BMI, how healthy they feel, age, income and so on. Just under 16% report diabetes or prediabetes. We treat a missed case as the expensive mistake, so our rule is: catch a fixed share of cases, 80%, and then compare approaches by how many people they have to flag to get there. And the comparison that matters is not against guessing; it’s against the paper risk test clinics already hand out.

## 3. We count prediabetes as positive, on purpose

The data come from Kaggle’s cleaned version of the 2015 BRFSS. We used the three-class file, because it separates prediabetes from diabetes. Our main outcome counts both as positive: that’s 15.8% of people. The roadmap mentioned about 14%; that turns out to be diabetes alone, and we checked that Kaggle’s binary file actually codes prediabetes as negative. We think prediabetes belongs in the positive group, because it’s exactly what screening is meant to catch. We also ran a harder sensitivity analysis on prediabetes alone, where only 2% are positive. One more thing: there were no missing or invalid values, and we kept everyone, including people with identical answers, because there’s no ID to prove they’re duplicates.

## 4. Five answers carry the signal; one mostly measures seeing a doctor

Exploration and the hypothesis tests point to the same five answers: general health, high blood pressure, BMI, difficulty walking and high cholesterol. All 25 of our planned tests were significant, but with a quarter of a million people that tells you very little, so we read effect sizes. High blood pressure goes with almost five times the odds. The biggest odds ratio, though, is for having had a cholesterol check, and that’s a warning: it mostly measures contact with doctors, and doctors are who make the diagnosis. These are also one-answer-at-a-time associations; later we’ll see several of them shrink or even reverse when everything is considered together.

## 5. Every decision used training rows; the test set stayed locked

Before modeling we fixed one split for the whole team: 80% for training, divided into five fixed folds, and 20% locked away. The code literally refuses to hand out the test rows until a saved threshold exists for that exact split. Three checks here. Scaling happens inside the model pipelines, so it only ever learns from training folds. About 15% of people share their exact answers with someone else; we tested whether that leaks, and keeping those people together changed the score by a thousandth at most. And class weighting, the usual fix for imbalance, didn’t rank people any better; it just inflated everyone’s risk from 16% to 39%. So we kept honest probabilities and chose the cutoff ourselves.

## 6. Every tuned model beats the checklist; XGBoost wins every fold

We tuned three models: logistic regression as the interpretable baseline, a random forest, and XGBoost, all on the same five folds and scored by PR-AUC, which suits a rare outcome. The big jump is from the paper checklist to any model: PR-AUC goes from 0.33 to about 0.44 to 0.46. To catch 80% of cases, the checklist flags half of everyone; the models flag 38 to 39%. We wrote the selection rule down before the run: take the best model unless logistic regression isn’t significantly worse. XGBoost won in all five folds against both, so it went forward. But notice how close logistic regression is; if you need a model you can explain line by line, it gives up very little.

## 7. The same five answers drive every model

What do the models actually use? Shuffling one answer at a time and measuring the drop in performance, all three models agree: BMI, general health, age, high blood pressure and high cholesterol. Everything else barely matters once those five are known. We also fitted one ordinary logistic regression with all 21 answers to get adjusted odds ratios, and that’s a useful warning about the earlier one-at-a-time numbers. Bad physical-health days looked risky on their own, but the effect reverses once general health is in the model. Smoking’s association disappears, because smokers are older and more often men. Exercise mostly works through BMI and health. So: associations, not causes.

## 8. We locked the cutoff before opening the test set

Here is the step that makes the final numbers trustworthy. We chose the cutoff using only training data: we refitted the saved model on four folds at a time so every training person got a risk from a model that never saw them, and those predictions reproduced stage 6 almost exactly. The rule, written before the run: the highest cutoff that still catches 80% of cases. That came out at 0.153, a predicted risk of about 15%. The calibration was already excellent, so no adjustment. The cutoff went into a file, and the test command refuses to run without it. On the right you can see the trade-off: every extra ten points of sensitivity costs roughly 90 to 130 more people flagged per thousand.

## 9. Most cases found, far fewer false alarms than the checklist

And here is the result on the 50,736 people we held back. At the locked cutoff, the model found 79.4% of people with diabetes or prediabetes, right on the 80% target, with a tight confidence interval. It flagged 37.6% of people, and one in three of those flagged actually had the condition. Compared with the paper checklist at its own cutoff, which finds the same share of cases, that’s 96 fewer false alarms for every thousand people screened. The test-set scores match what we saw in validation, so nothing was overfitted, and the predicted risks are still well calibrated.

## 10. Per 1,000 screened: 125 cases found, 33 missed

Translated for a clinic: out of every thousand people screened, about 158 have diabetes or prediabetes. The model sends 376 for a blood test, finds 125 of the 158 and misses 33. The paper checklist finds the same number but sends almost 100 more people. Logistic regression lands in between, much closer to XGBoost, so the simple model captures about 90% of the gain. And because the risks are calibrated, a person told their risk is 30% really is in a group where about 30% have the condition.

## 11. It misses younger and better-off people, and prediabetes

Now the honest part: who does it miss? The model leans on BMI, general health, age, blood pressure and cholesterol, so a case who looks ordinary on all five is invisible to it. The missed cases have the same average BMI as people without the condition. That shows up in subgroups: under-45s, only 44% of cases are found; at incomes of 50 thousand dollars and up, 68%. Meanwhile, over-65s and low-income groups get most cases found but also about half of everyone flagged. And it finds 81% of diagnosed diabetes but only 65% of prediabetes, which is the group screening is really for. One cutoff for everyone means different trade-offs for different groups.

## 12. For prediabetes alone, the signal is weak

We also asked the harder question directly: leave out everyone who already has diabetes and look for prediabetes alone. Same pipeline, same rules. The model still beats the checklist, finding 81% of cases against 70%. But it has to flag 44% of everyone to do it, and only 4% of those flagged report prediabetes. That’s the honest headline for early detection from survey answers: it’s possible, but expensive. One caveat in the model’s favour: most people with prediabetes have never been told, so many of these “false alarms” may really be undiagnosed cases, and we simply can’t tell from this data.

## 13. A research prototype, not a diagnostic tool

Six limits we want to be upfront about. The labels are self-reported at one moment in 2015, so this identifies people who already know their status; it predicts nothing about who will develop diabetes. 88% of positives already have a diagnosis, so the model partly learns what a diagnosis looks like. Some of its best answers, like cholesterol checks and blood pressure, partly measure seeing a doctor; without them it loses a little but still beats the checklist. One cutoff gives different groups very different trade-offs. The survey extract has no weights and covers one year. And everything here is association, not cause. So: a research prototype for risk stratification, not a diagnostic tool.

## 14. Survey answers can find 8 in 10 cases with a fifth fewer people sent for testing than today’s paper checklist

To sum up: with nothing but survey answers, a tuned model can find about eight in ten people with diabetes or prediabetes while sending a fifth fewer people for testing than the paper checklist clinics use today. It is weaker for prediabetes and for younger people, and it is a research prototype, not a diagnosis. Everything is reproducible: one command per stage, and we checked that the data preparation, the split, the cutoff and the test results rebuild identically, backed by 63 automated tests. The code, notebooks and reports are all in our repository, and the technical report has the full detail. Thank you, we’re happy to take questions.
