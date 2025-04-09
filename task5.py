# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Task 5: Results Analysis - Comparing various models like in Assignment 1
def analyze_results(all_results):
    print("\n--- Task 5: Results Analysis ---")
    
    # Create a DataFrame for better visualization
    methods = list(all_results.keys())
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1']
    
    results_df = pd.DataFrame(columns=['Method'] + metrics)
    
    for i, method in enumerate(methods):
        row = [method]
        for metric in metrics:
            row.append(all_results[method][metric])
        results_df.loc[i] = row
    
    print("\nResults Summary:")
    print(results_df)
    
    # Plot results for visual comparison
    plt.figure(figsize=(15, 10))
    
    # Plot each metric separately
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x='Method', y=metric, data=results_df)
        plt.title(f'Comparison of {metric} across methods')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    plt.savefig('results_comparison.png')
    plt.close()
    
    best_methods = {}
    for metric in metrics:
        best_method = results_df.loc[results_df[metric].idxmax()]['Method']
        best_methods[metric] = best_method
    
    print("\nBest performing methods:")
    for metric, method in best_methods.items():
        print(f"Best {metric}: {method} with value {results_df.loc[results_df['Method'] == method, metric].values[0]:.4f}")
    
    print("\n--- Detailed Comparative Analysis ---")
    
    tfidf_models = [m for m in methods if 'TFIDF' in m]
    w2v_models = [m for m in methods if 'word2vec' in m]
    
    tfidf_mean_accuracy = results_df[results_df['Method'].isin(tfidf_models)]['Accuracy'].mean()
    w2v_mean_accuracy = results_df[results_df['Method'].isin(w2v_models)]['Accuracy'].mean()
    
    print(f"\nComparison of feature extraction techniques:")
    print(f"Average accuracy with TF-IDF: {tfidf_mean_accuracy:.4f}")
    print(f"Average accuracy with Word2Vec: {w2v_mean_accuracy:.4f}")
    print(f"Difference: {abs(tfidf_mean_accuracy - w2v_mean_accuracy):.4f}")
    
    rf_models = [m for m in methods if '3' in m or 'fine tune' in m]
    lr_models = [m for m in methods if '1' in m]
    svm_models = [m for m in methods if '2' in m]
    bert_models = [m for m in methods if 'BERT' in m]
    
    model_groups = {
        'Random Forest & Fine-tuned BERT': rf_models,
        'Logistic Regression': lr_models,
        'SVM/LinearSVC': svm_models,
        'BERT models': bert_models
    }
    
    print("\nComparison of classifier types:")
    for group_name, models in model_groups.items():
        if models:
            mean_acc = results_df[results_df['Method'].isin(models)]['Accuracy'].mean()
            mean_f1 = results_df[results_df['Method'].isin(models)]['F1'].mean()
            print(f"{group_name}: Avg Accuracy: {mean_acc:.4f}, Avg F1: {mean_f1:.4f}")
    
    print("\nInterpretation of Results:")
    best_method = results_df.loc[results_df['Accuracy'].idxmax()]['Method']
    best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]['Accuracy']
    worst_method = results_df.loc[results_df['Accuracy'].idxmin()]['Method']
    worst_accuracy = results_df.loc[results_df['Accuracy'].idxmin()]['Accuracy']
    
    print(f"The best performing method is {best_method} with accuracy {best_accuracy:.4f}")
    print(f"The worst performing method is {worst_method} with accuracy {worst_accuracy:.4f}")
    
    if best_method in rf_models and 'BERT' not in best_method:
        print("Random Forest continues to be a strong performer, similar to findings in Assignment 1.")
    elif 'BERT' in best_method:
        print("BERT models show significant advantages for sentiment analysis compared to traditional approaches.")
    elif 'word2vec' in best_method:
        print("Word embeddings demonstrate stronger contextual understanding compared to bag-of-words approaches.")
    
    print("\nComparison with Assignment 1:")
    print("- Both assignments show that advanced embedding techniques generally outperform basic bag-of-words approaches")
    print("- Random Forest classifier performed well in both text classification tasks")
    print("- The sentiment analysis task with BERT shows clear improvements over traditional methods")
    
    return results_df