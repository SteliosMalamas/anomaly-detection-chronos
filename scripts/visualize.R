library(jsonlite)
library(readr)
library(dplyr)
library(ggplot2)
library(pROC)
library(PRROC)
library(grid)

pred_csv <- "/home/stelios/Downloads/results/predictions.csv"
metrics_js <- "/home/stelios/Downloads/results/metrics.json"
series_csv <- "/home/stelios/Downloads/results/series_flags.csv"
out_dir <- "/home/stelios/Downloads/results/viz"

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

preds <- read_csv(pred_csv, show_col_types = FALSE)
metrics <- fromJSON(metrics_js)
series <- read_csv(series_csv, show_col_types = FALSE)

#build combined df for anomaly plotting
cmp <- series %>%
  mutate(
    true_flag = preds$y_true,
    detected_flag = preds$flag
  )

#confusion matrix
cm <- matrix(unlist(metrics$Confusion_Matrix), nrow=2, byrow=TRUE)
rownames(cm) <- c("True 0","True 1")
colnames(cm) <- c("Pred 0","Pred 1")
cm_df <- as.data.frame(as.table(cm))

#heatmap of confusion counts
df_heat <- cm_df
p1 <- ggplot(df_heat, aes(Var2, Var1, fill=Freq)) +
  geom_tile(color="white") +
  geom_text(aes(label=Freq), color="black", size=5) +
  scale_fill_gradient(low="lightblue", high="navy") +
  labs(x="", y="", title="Confusion Matrix") +
  theme_minimal(base_size=14)

#bar chart of each confusion category
conf_counts <- data.frame(
  category = c("True Negative","False Positive","False Negative","True Positive"),
  count = c(
    cm["True 0","Pred 0"],
    cm["True 0","Pred 1"],
    cm["True 1","Pred 0"],
    cm["True 1","Pred 1"]
  )
)
#reorder factor
default_order <- c("True Negative","False Positive","False Negative","True Positive")
conf_counts$category <- factor(conf_counts$category, levels=default_order)

p1b <- ggplot(conf_counts, aes(x=category, y=count, fill=category)) +
  geom_col(show.legend=FALSE) +
  labs(title="Confusion Counts", x="Category", y="Count") +
  theme_minimal(base_size=14) +
  theme(axis.text.x = element_text(angle=45, hjust=1))

ggsave(file.path(out_dir, "confusion_matrix_heatmap.png"), p1, width=4, height=4)
ggsave(file.path(out_dir, "confusion_counts_bar.png"), p1b, width=6, height=4)

#pr curve
fg <- filter(preds, y_true==1) %>% pull(y_score)
bg <- filter(preds, y_true==0) %>% pull(y_score)
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve=TRUE)
pr_df <- data.frame(Recall = pr$curve[,1], Precision = pr$curve[,2])

p2 <- ggplot(pr_df, aes(x=Recall, y=Precision)) +
  geom_line(color="firebrick", linewidth=1) +
  annotate("text", x=0.6, y=0.2,
           label = sprintf("PR AUC = %.3f", metrics$Precision_Recall_AUC),
           size=5) +
  labs(title="Precision–Recall Curve") +
  theme_minimal(base_size=14)

ggsave(file.path(out_dir, "precision_recall_curve.png"), p2, width=6, height=4)

#roc curve
roc_obj <- roc(preds$y_true, preds$y_score)
roc_df  <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR =   roc_obj$sensitivities
)

p3 <- ggplot(roc_df, aes(x=FPR, y=TPR)) +
  geom_line(color="steelblue", linewidth=1) +
  geom_abline(intercept=0, slope=1, linetype="dashed", color="gray") +
  annotate("text", x=0.6, y=0.2,
           label = sprintf("ROC AUC = %.3f", metrics$ROC_AUC),
           size=5) +
  labs(title="ROC Curve") +
  theme_minimal(base_size=14)

ggsave(file.path(out_dir, "roc_curve.png"), p3, width=6, height=4)

#score distribution
thr <- metrics$Best_threshold

p4 <- ggplot(preds, aes(x=y_score, fill=factor(y_true))) +
  geom_histogram(position="identity", alpha=0.6, bins=50, color="black") +
  geom_vline(xintercept=thr, color="red", linetype="dashed", linewidth=1) +
  scale_fill_manual("", values=c("0"="#999999","1"="#E69F00"),
                    labels=c("Benign","Anomaly")) +
  labs(title="Score Distribution by Class",
       x="Anomaly Score", y="Count") +
  theme_minimal(base_size=14)

ggsave(file.path(out_dir, "score_distribution.png"), p4, width=6, height=4)

#time series with detected anomalies
p5 <- ggplot(cmp, aes(x=timestamp, y=value)) +
  geom_line(color="gray60") +
  geom_point(data = filter(cmp, detected_flag==1),
             aes(x=timestamp, y=value),
             color="red", size=0.8) +
  labs(title="Time Series with Detected Anomalies",
       x="Time", y="Flow Bytes/s") +
  theme_minimal(base_size=14)

ggsave(file.path(out_dir, "time_series_anomalies.png"), p5, width=8, height=3)

#dual rug plot
p6 <- ggplot() +
  geom_rug(data = filter(cmp, true_flag==1),
           aes(x=timestamp),
           sides="t", color="black", length=unit(0.05,"npc")) +
  geom_rug(data = filter(cmp, detected_flag==1),
           aes(x=timestamp),
           sides="b", color="red",   length=unit(0.05,"npc")) +
  labs(title="True vs. Detected Anomalies (rug)",
       x="Time", y=NULL) +
  theme_minimal(base_size=14) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank(),
    plot.margin = margin(10,10,10,10)
  )

ggsave(file.path(out_dir, "dual_rug_only.png"), p6, width=10, height=2)

#f1 - threshold
thrs <- seq(min(preds$y_score), max(preds$y_score), length.out=200)
f1s <- sapply(thrs, function(t) {
  p <- as.integer(preds$y_score > t)
  TP <- sum(p == 1 & preds$y_true == 1)
  FP <- sum(p == 1 & preds$y_true == 0)
  FN <- sum(p == 0 & preds$y_true == 1)
  if ((TP + FP) == 0) return(0)
  pr <- TP / (TP + FP)
  rc <- if ((TP + FN) == 0) 0 else TP / (TP + FN)
  if (pr + rc == 0) return(0)
  2 * pr * rc / (pr + rc)
})
df_f1 <- data.frame(threshold=thrs, F1=f1s)

p7 <- ggplot(df_f1, aes(x=threshold, y=F1)) +
  geom_line(color="darkgreen", linewidth=1) +
  geom_vline(xintercept=metrics$Best_threshold,
             linetype="dashed", color="red") +
  annotate("text", x=metrics$Best_threshold, y=max(f1s)*0.9,
           label=sprintf("opt thr=%.3f", metrics$Best_threshold),
           hjust=-0.1) +
  labs(title="F1 Score vs. Anomaly Threshold",
       x="Threshold", y="F1 Score") +
  theme_minimal(base_size=14)

ggsave(file.path(out_dir, "f1_vs_threshold.png"), p7, width=6, height=4)