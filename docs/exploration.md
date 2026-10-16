# Stage 3 outputs

`eda/` contains observed class counts and fractions, target-group summaries, histograms and boxplots for BMI, age and health indicators, category-specific positive fractions, Pearson/Spearman matrices and heatmaps, and a generated observations document. Plots use the entire cleaned table without oversampling. Categorical bars show positive fractions with underlying counts in the CSV; rare categories should not be compared without those counts.

Numeric summaries of ordinal codes describe coding, not exact ages or clinical units. Boxplot outliers are hidden only for readability; histograms and exported data retain them. Figures are saved headlessly at 180 DPI. Review the generated observations alongside the actual plots rather than interpreting a small p-value as a large or useful screening effect.
