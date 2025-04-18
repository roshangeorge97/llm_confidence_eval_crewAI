import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from tqdm import tqdm

# List of datasets (Domestic_Robot removed)
datasets = ['GSM8K', 'SVAMP', 'HotpotQA', 'Sports', 'LLC']

def generate_sample_data(dataset_name, sample_size=100):
    """Generate random sample data for a given dataset."""
    results = []
    for _ in tqdm(range(sample_size), desc=f"Generating sample data for {dataset_name}"):
        accuracy = np.random.choice([0, 1])
        confidence = np.random.uniform(0, 1)
        consistency = np.random.choice([True, False], p=[0.7, 0.3])
        initial_correct = np.random.choice([True, False])
        final_correct = (accuracy == 1)
        improvement = not initial_correct and final_correct
        regression = initial_correct and not final_correct
        ground_truth = str(np.random.randint(1, 100)) if dataset_name in ['GSM8K', 'SVAMP', 'LLC'] else \
                       np.random.choice(['yes', 'no']) if dataset_name == 'Sports' else "Sample Answer"
        initial_response = ground_truth if initial_correct else str(np.random.randint(1, 100))
        final_response = ground_truth if final_correct else str(np.random.randint(1, 100))
        
        result = {
            'question': f"Sample question for {dataset_name}",
            'initial_response': initial_response,
            'final_response': final_response,
            'confidence': confidence,
            'accuracy': accuracy,
            'consistency': consistency,
            'improvement': improvement,
            'regression': regression,
            'ground_truth': ground_truth
        }
        results.append(result)
    return results

class Analytics:
    @staticmethod
    def generate_report(results, dataset_name):
        accuracies = [r['accuracy'] for r in results if r is not None]
        confidences = [r['confidence'] for r in results if r is not None]
        consistencies = [r['consistency'] for r in results if r is not None]
        improvements = [r['improvement'] for r in results if r is not None]
        regressions = [r['regression'] for r in results if r is not None]
        
        report = {
            'dataset': dataset_name,
            'accuracy': np.mean(accuracies) if accuracies else 0,
            'confidence': np.mean(confidences) if confidences else 0,
            'accuracy_confidence_correlation': np.corrcoef(accuracies, confidences)[0, 1] if len(accuracies) > 1 else 0,
            'sample_size': len(results),
            'consistency': np.mean(consistencies) if consistencies else 0,
            'improvement': np.mean(improvements) if improvements else 0,
            'regression': np.mean(regressions) if regressions else 0
        }
        
        os.makedirs('results', exist_ok=True)
        
        # 1. Accuracy vs Confidence Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=confidences, y=accuracies, alpha=0.6)
        if len(accuracies) > 1:
            slope, intercept, r_value, _, _ = linregress(confidences, accuracies)
            line = [slope * x + intercept for x in confidences]
            plt.plot(confidences, line, color='red', linestyle='--', label=f'R² = {r_value**2:.2f}')
        plt.title(f'Accuracy vs Confidence - {dataset_name}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'results/{dataset_name}_accuracy_vs_confidence.png')
        plt.close()
        
        # 2. Accuracy Distribution Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(accuracies, bins=[-0.5, 0.5, 1.5], discrete=True)
        plt.title(f'Accuracy Distribution - {dataset_name}')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.xticks([0, 1], ['Incorrect', 'Correct'])
        plt.savefig(f'results/{dataset_name}_accuracy_dist.png')
        plt.close()
        
        # 3. Self-Correction Success Bar Chart
        plt.figure(figsize=(8, 5))
        initial_acc = np.mean([1 if r['initial_response'] == r['ground_truth'] else 0 for r in results if r is not None])
        refined_acc = np.mean(accuracies)
        plt.bar(['Initial', 'Refined'], [initial_acc, refined_acc], color=['blue', 'green'])
        plt.title(f'Self-Correction Success - {dataset_name}')
        plt.ylabel('Accuracy')
        plt.savefig(f'results/{dataset_name}_self_correction_success.png')
        plt.close()
        
        # 4. Answer Changes Pie Chart
        changes = {'No Change': 0, 'Correct-to-Incorrect': 0, 'Incorrect-to-Correct': 0, 'Incorrect-to-Incorrect': 0}
        for r in results:
            if r is None: continue
            init_correct = r['initial_response'] == r['ground_truth']
            final_correct = r['final_response'] == r['ground_truth']
            if init_correct and final_correct:
                changes['No Change'] += 1
            elif init_correct and not final_correct:
                changes['Correct-to-Incorrect'] += 1
            elif not init_correct and final_correct:
                changes['Incorrect-to-Correct'] += 1
            else:
                changes['Incorrect-to-Incorrect'] += 1
        plt.figure(figsize=(8, 8))
        plt.pie(changes.values(), labels=changes.keys(), autopct='%1.1f%%')
        plt.title(f'Answer Changes - {dataset_name}')
        plt.savefig(f'results/{dataset_name}_answer_change_pie.png')
        plt.close()
        
        # 5. Confidence Over Time Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(confidences)), confidences, label='Confidence')
        plt.title(f'Confidence Over Time - {dataset_name}')
        plt.xlabel('Item Index')
        plt.ylabel('Confidence')
        plt.legend()
        plt.savefig(f'results/{dataset_name}_confidence_over_time.png')
        plt.close()
        
        # 6. Confusion Matrix
        conf_matrix = np.zeros((2, 2))
        for r in results:
            if r is None: continue
            conf = 1 if r['confidence'] >= 0.7 else 0
            acc = 1 if r['accuracy'] else 0
            conf_matrix[conf, acc] += 1
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues', xticklabels=['Incorrect', 'Correct'], yticklabels=['Low Confidence', 'High Confidence'])
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.savefig(f'results/{dataset_name}_confusion_matrix.png')
        plt.close()
        
        # 7. Metrics Bar Chart
        metrics = {
            'Consistency': np.mean(consistencies),
            'Improvement': np.mean(improvements),
            'Regression': np.mean(regressions)
        }
        plt.figure(figsize=(8, 5))
        plt.bar(metrics.keys(), metrics.values(), color=['gray', 'green', 'red'])
        plt.title(f'Self-Correction Metrics - {dataset_name}')
        plt.ylabel('Proportion')
        plt.savefig(f'results/{dataset_name}_metrics_bar.png')
        plt.close()

        # 8. Confidence Histogram by Correctness
        correct_conf = [conf for conf, acc in zip(confidences, accuracies) if acc == 1]
        incorrect_conf = [conf for conf, acc in zip(confidences, accuracies) if acc == 0]
        plt.figure(figsize=(10, 6))
        plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect', color='red')
        plt.title(f'Confidence Distribution - {dataset_name}')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'results/{dataset_name}_confidence_hist.png')
        plt.close()
        
        return report

if __name__ == "__main__":
    analytics = Analytics()
    all_results = {}
    
    for dataset in datasets:
        print(f"\nGenerating sample data for {dataset}...")
        sample_results = generate_sample_data(dataset, sample_size=100)
        all_results[dataset] = sample_results
        
        report = analytics.generate_report(sample_results, dataset)
        print(f"\n{report['dataset']} Report:")
        print(f"Accuracy: {report['accuracy']:.2%}")
        print(f"Avg Confidence: {report['confidence']:.2f}")
        print(f"Correlation: {report['accuracy_confidence_correlation']:.2f}")
        print(f"Sample Size: {report['sample_size']}")
        print(f"Consistency: {report['consistency']:.2%}")
        print(f"Improvement: {report['improvement']:.2%}")
        print(f"Regression: {report['regression']:.2%}")
    
    print("\nSample data generation and reporting complete. Check 'results' directory for visualizations.")