def calculate_accuracy(results_df):
    if results_df.empty:
        return 0.0

    accuracy = (
        results_df["correct"].mean() * 100
    )

    return round(accuracy, 2)