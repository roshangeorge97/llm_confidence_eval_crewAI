import google.generativeai as genai
import json
import time
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from groq import Groq
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd
from scipy.stats import linregress

load_dotenv()

genai.configure(api_key="AIzaSyDirrgD8-pTNz3GcZ11fd4S1O9fgBqr-UQ")

class Config:
    def __init__(self):
        self.models = {
            'gemini-1.5-flash': {'temperature': 0.3, 'max_output_tokens': 2048}
        }
        self.datasets = {
            'math': ['GSM8K', 'SVAMP'],
            'commonsense': ['Sports'],
            'symbolic': ['LLC'],
            'multihop': ['Hotpot']
        }
        self.confidence_thresholds = {
            'math': 0.7,
            'commonsense': 0.6,
            'symbolic': 0.5,
            'multihop': 0.65
        }
        self.crew_config = {
            'process': Process.sequential,
            'enable_crew': True
        }
        self.llm = LLM(
            model='gemini/gemini-1.5-flash',
            api_key="AIzaSyDirrgD8-pTNz3GcZ11fd4S1O9fgBqr-UQ"
        )

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_formats = {
            'GSM8K': self.load_gsm8k,
            'SVAMP': self.load_svamp,
            'Hotpot': self.load_hotpot,
            'Sports': self.load_sports,
            'LLC': self.load_llc
        }
    
    def load_dataset(self, dataset_name):
        if dataset_name not in self.dataset_formats:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.dataset_formats[dataset_name]()
    
    def load_gsm8k(self):
        with open("dataset/GSM8K.jsonl") as f:
            return [self.parse_math_line(line) for line in f]
    
    def load_svamp(self):
        with open("dataset/SVAMP.jsonl") as f:
            return [self.parse_math_line(line) for line in f]
    
    def load_hotpot(self):
        with open("dataset/Hotpot.jsonl") as f:
            return [self.parse_hotpot_line(line) for line in f]
    
    def load_sports(self):
        with open("dataset/Sports.jsonl") as f:
            return [self.parse_sports_line(line) for line in f]
    
    def load_llc(self):
        with open("dataset/LLC.jsonl") as f:
            return [self.parse_llc_line(line) for line in f]
    
    def parse_math_line(self, line):
        data = json.loads(line)
        return {
            'question': data['question'],
            'answer': str(data['answer']),
            'type': 'math'
        }
    
    def parse_hotpot_line(self, line):
        data = json.loads(line)
        return {
            'question': data['question'],
            'answer': data['answer'],
            'context': data['context'],
            'type': 'multihop'
        }
    
    def parse_sports_line(self, line):
        data = json.loads(line)
        return {
            'question': data['question'],
            'answer': 'yes' if data['answer'] == 'plausible' else 'no',
            'type': 'commonsense'
        }
    
    def parse_llc_line(self, line):
        data = json.loads(line)
        return {
            'question': data['question'],
            'answer': data['answer'],
            'type': 'symbolic'
        }

class GeminiClient:
    def __init__(self, config):  
        self.config = config
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_content(self, prompt, model_name='gemini-1.5-flash', max_retries=5):
        backoff_time = 1
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.models[model_name]['temperature'],
                        max_output_tokens=self.config.models[model_name]['max_output_tokens']
                    )
                )
                return response.text
            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    tqdm.write(f"⚠️ Gemini Rate limited. Retrying in {backoff_time}s (Attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                else:
                    break
        return None

class CALMCrew:
    def __init__(self, config):
        self.config = config
        self.researcher = Agent(
            role='Senior Researcher',
            goal='Generate accurate initial responses',
            backstory="Expert in complex problem solving",
            llm=config.llm,
            verbose=True
        )
        
        self.evaluator = Agent(
            role='Quality Assurance Engineer',
            goal='Refine and validate responses',
            backstory="Specialist in error detection",
            llm=config.llm,
            verbose=True
        )

    def create_crew(self, question, context, task_type):
        initial_task = Task(
            description=self.build_initial_prompt(question, context, task_type),
            expected_output="Initial response",
            agent=self.researcher
        )
        
        refinement_task = Task(
            description=self.build_refinement_prompt(question, context, "$initial_response", task_type),
            expected_output="Refined response",
            agent=self.evaluator
        )
        
        return Crew(
            agents=[self.researcher, self.evaluator],
            tasks=[initial_task, refinement_task],
            process=self.config.crew_config['process'],
            verbose=True
        )
        
    def build_initial_prompt(self, question, context, task_type):
        base_prompt = ""
        if context:
            base_prompt += f"Context: {context}\n\n"
            
        if task_type == 'math':
            return f"{base_prompt}Question: {question}\nSolve step-by-step. After calculations, write ONLY THE FINAL NUMERICAL ANSWER within double hashes (##) with no units or symbols.\nExample: ##120##"
        elif task_type == 'commonsense':
            return f"{base_prompt}Question: {question}\nAnswer with 'plausible' or 'implausible' within ##."
        elif task_type == 'symbolic':
            return f"{base_prompt}Question: {question}\nPut your final answer within ##."
        return f"{base_prompt}Question: {question}\nAnswer concisely within ##."
    
    def build_refinement_prompt(self, question, context, initial_response, task_type):
        base_refinement = "Review the initial answer and provide a refined answer in ##.\n"
        if context:
            base_refinement += f"Context: {context}\n"
            
        base_refinement += f"Question: {question}\nInitial Answer: {initial_response}\n"
        
        if task_type == 'math':
            return base_refinement + "Check calculations and logic."
        elif task_type == 'commonsense':
            return base_refinement + "Verify factual consistency."
        return base_refinement + "Improve your answer."

class CALMEngine:
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.client = GeminiClient(config)
        self.crew = CALMCrew(config)
        self.last_call_time = 0
        self.calls_per_minute = 60

    def process_dataset(self, dataset_name):
        data = self.loader.load_dataset(dataset_name)
        results = []
        min_interval = 60 / self.calls_per_minute
        
        for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
            elapsed = time.time() - self.last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = self.process_item(item, idx, dataset_name)
            results.append(result)
            self.last_call_time = time.time()
            
        return results

    def process_item(self, item, idx, dataset_name):
        question = item['question']
        context = item.get('context', '')
        answer = item['answer']
        task_type = item['type']
        
        try:
            tqdm.write(f"\n{'='*40}")
            tqdm.write(f"Dataset: {dataset_name} | Item: {idx+1}")
            tqdm.write(f"Question: {question}")
            if context:
                tqdm.write(f"Context: {context[:100]}...")

            if self.config.crew_config['enable_crew']:
                crew = self.crew.create_crew(question, context, task_type)
                crew.kickoff()
                # Extract string content from TaskOutput using str() or .raw
                initial_response = str(crew.tasks[0].output) if crew.tasks[0].output else ""
                refined_response = str(crew.tasks[1].output) if crew.tasks[1].output else ""
            else:
                initial_prompt = self.crew.build_initial_prompt(question, context, task_type)
                initial_response = self.client.generate_content(initial_prompt)
                refinement_prompt = self.crew.build_refinement_prompt(question, context, initial_response, task_type)
                refined_response = self.client.generate_content(refinement_prompt)

            final_answer = self.extract_answer(refined_response)
            initial_correct = self.evaluate_answer(initial_response, answer, task_type)
            accuracy = self.evaluate_answer(final_answer, answer, task_type)
            confidence = self.calculate_confidence(initial_response, final_answer, answer)
            consistency = initial_response == final_answer
            improvement = not initial_correct and accuracy
            regression = initial_correct and not accuracy
            
            tqdm.write(f"[Initial Response]: {initial_response[:200]}...")
            tqdm.write(f"[Refined Response]: {refined_response[:200]}...")
            tqdm.write(f"[Confidence Score]: {confidence:.2f}")
            tqdm.write(f"[Accuracy Check]: {'✅ Correct' if accuracy else '❌ Incorrect'}")
            tqdm.write(f"[Ground Truth]: {answer}")
            tqdm.write(f"[Final Answer]: {final_answer}")
            tqdm.write(f"{'='*40}\n")
            
            return {
                'question': question,
                'initial_response': initial_response,
                'final_response': final_answer,
                'confidence': confidence,
                'accuracy': accuracy,
                'consistency': consistency,
                'improvement': improvement,
                'regression': regression,
                'ground_truth': answer
            }
            
        except Exception as e:
            tqdm.write(f"🚨 Error processing item: {str(e)}")
            return None

    def parse_crew_output(self, output):
        try:
            if hasattr(output, 'tasks'):
                initial_response = output.tasks[0].result
                refined_response = output.tasks[1].result
            else:
                output_str = str(output)
                parts = output_str.split('Final Answer:')
                if len(parts) >= 2:
                    refined_response = parts[-1].strip()
                    initial_response = parts[0].split('Initial Answer:')[-1].strip()
                else:
                    initial_response, refined_response = output_str, output_str
            return initial_response, refined_response
        except Exception as e:
            tqdm.write(f"Error parsing crew output: {str(e)}")
            return "Error", "Error"

    def extract_answer(self, text):
        if not text:
            return ""
        
        boxed_match = re.search(r'\\boxed{(\d+\.?\d*)}', text)
        if boxed_match:
            return self.normalize_response(boxed_match.group(1))
        
        delimiter_match = re.findall(r'##(.*?)##', text, re.DOTALL)
        if delimiter_match:
            candidate = delimiter_match[-1].replace('$', '').strip()
            numbers = re.findall(r'\d+\.?\d*', candidate)
            if numbers:
                return self.normalize_response(numbers[-1])
        
        speed_match = re.search(r'(\d+\.?\d*)\s*(?:mph|miles per hour)', text, re.IGNORECASE)
        if speed_match:
            return self.normalize_response(speed_match.group(1))
        
        equation_match = re.search(r'=\s*[\$€£]?\s*(\d+\.?\d*)\b', text)
        if equation_match:
            return self.normalize_response(equation_match.group(1))
        
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return self.normalize_response(numbers[-1])
        
        return ""

    def normalize_response(self, text):
        if not text: 
            return ""
        
        clean_text = text.replace(',', '').lower().strip()
        
        boxed_match = re.search(r'\\boxed{(\d+\.?\d*)}', clean_text)
        if boxed_match:
            return f"{float(boxed_match.group(1)):g}"
        
        delimiter_match = re.findall(r'##(.*?)##', clean_text, re.DOTALL)
        if delimiter_match:
            candidate = delimiter_match[-1].replace('$', '').strip()
            numbers = re.findall(r'\d+\.?\d*', candidate)
            if numbers:
                return f"{float(numbers[-1]):g}"
        
        money_match = re.search(r'[\$\£\€]?\s*(\d+\.?\d*)\b', clean_text)
        if money_match:
            return f"{float(money_match.group(1)):g}"
        
        unit_match = re.search(r'(\d+\.?\d*)\s*(?:grams?|years?|hours?|%|mph)\b', clean_text)
        if unit_match:
            return f"{float(unit_match.group(1)):g}"
        
        equation_match = re.search(r'(?:=|\()\s*(\d+\.?\d*)', clean_text)
        if equation_match:
            return f"{float(equation_match.group(1)):g}"
        
        numbers = re.findall(r'\d+\.?\d*', clean_text)
        if numbers:
            return f"{float(numbers[-1]):g}"
        
        last_word = re.findall(r'\b(\d+\.?\d*|\w+)\b', clean_text)
        if last_word:
            last = last_word[-1]
            if re.match(r'^\d', last):
                return f"{float(last):g}" if '.' in last else last
        
        return clean_text

    def calculate_confidence(self, initial, refined, ground_truth):
        initial_norm = self.normalize_response(initial)
        refined_norm = self.normalize_response(refined)
        truth_norm = self.normalize_response(ground_truth)
        
        correct = refined_norm == truth_norm
        consistent = initial_norm == refined_norm
        
        if correct:
            return 0.9 if not consistent else 1.0
        return 0.4 if consistent else 0.1

    def evaluate_answer(self, response, ground_truth, task_type):
        response_clean = self.normalize_response(response)
        truth_clean = self.normalize_response(ground_truth)
        
        try:
            if task_type == 'math':
                return abs(float(response_clean) - float(truth_clean)) < 1e-6
            return response_clean == truth_clean
        except:
            return response_clean == truth_clean

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
        
        # Existing Visualizations
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
        
        plt.figure(figsize=(8, 5))
        sns.histplot(accuracies, bins=[-0.5, 0.5, 1.5], discrete=True)
        plt.title(f'Accuracy Distribution - {dataset_name}')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.xticks([0, 1], ['Incorrect', 'Correct'])
        plt.savefig(f'results/{dataset_name}_accuracy_dist.png')
        plt.close()
        
        plt.figure(figsize=(8, 5))
        initial_acc = np.mean([1 if r['initial_response'] == r['ground_truth'] else 0 for r in results if r is not None])
        refined_acc = np.mean(accuracies)
        plt.bar(['Initial', 'Refined'], [initial_acc, refined_acc], color=['blue', 'green'])
        plt.title(f'Self-Correction Success - {dataset_name}')
        plt.ylabel('Accuracy')
        plt.savefig(f'results/{dataset_name}_self_correction_success.png')
        plt.close()
        
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
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(confidences)), confidences, label='Confidence')
        plt.title(f'Confidence Over Time - {dataset_name}')
        plt.xlabel('Item Index')
        plt.ylabel('Confidence')
        plt.legend()
        plt.savefig(f'results/{dataset_name}_confidence_over_time.png')
        plt.close()
        
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
        
        # New Visualizations
        # Metrics Bar Chart
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

        # Confidence Histogram by Correctness
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
    
    @staticmethod
    def generate_additional_visuals(all_results, config):
        all_data = []
        for dataset_category in config.datasets:
            for dataset_name in config.datasets[dataset_category]:
                for r in all_results.get(dataset_name, []):
                    if r is not None:
                        all_data.append({
                            'dataset': dataset_name,
                            'task_type': r.get('type', 'unknown'),
                            'accuracy': r['accuracy'],
                            'confidence': r['confidence'],
                            'initial_correct': r['initial_response'] == r['ground_truth'],
                            'final_correct': r['final_response'] == r['ground_truth'],
                            'consistency': r['consistency'],
                            'improvement': r['improvement'],
                            'regression': r['regression']
                        })
        df = pd.DataFrame(all_data)
        
        # Boxplot: Confidence Distribution Across Datasets
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='dataset', y='confidence', hue='task_type', data=df)
        plt.title('Confidence Distribution Across Datasets')
        plt.xlabel('Dataset')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        plt.savefig('results/confidence_distribution_boxplot.png')
        plt.close()
        
        # Cumulative Accuracy vs Confidence Threshold
        plt.figure(figsize=(10, 6))
        thresholds = np.arange(0, 1.1, 0.1)
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            cum_accuracies = [dataset_df[dataset_df['confidence'] >= t]['accuracy'].mean() for t in thresholds]
            plt.plot(thresholds, cum_accuracies, label=dataset)
        plt.title('Cumulative Accuracy vs Confidence Threshold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('results/cumulative_accuracy_vs_confidence.png')
        plt.close()
        
        # Heatmap: Average Confidence by Task Type and Dataset
        pivot = df.pivot_table(values='confidence', index='task_type', columns='dataset', aggfunc='mean')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Average Confidence by Task Type and Dataset')
        plt.savefig('results/task_type_confidence_heatmap.png')
        plt.close()
        
        # Barplot: Accuracy by Task Type
        plt.figure(figsize=(10, 6))
        sns.barplot(x='task_type', y='accuracy', data=df, ci=None)
        plt.title('Accuracy by Task Type')
        plt.xlabel('Task Type')
        plt.ylabel('Accuracy')
        plt.savefig('results/accuracy_by_task_type.png')
        plt.close()
        
        # Scatterplot: Error Rate vs Confidence
        plt.figure(figsize=(10, 6))
        error_rates = 1 - df['accuracy']
        sns.scatterplot(x=df['confidence'], y=error_rates, alpha=0.6)
        plt.title('Error Rate vs Confidence')
        plt.xlabel('Confidence Score')
        plt.ylabel('Error Rate')
        plt.savefig('results/error_rate_vs_confidence.png')
        plt.close()
        
        # Pie Chart: Correctness Breakdown
        correctness = df['accuracy'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(correctness, labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['green', 'red'])
        plt.title('Correctness Breakdown (All Data)')
        plt.savefig('results/correctness_breakdown_pie.png')
        plt.close()
        
        # Pie Chart: Task Type Distribution
        task_dist = df['task_type'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(task_dist, labels=task_dist.index, autopct='%1.1f%%')
        plt.title('Task Type Distribution')
        plt.savefig('results/task_type_distribution_pie.png')
        plt.close()
        
        # Stacked Bar: Performance Improvement by Dataset
        perf_data = df.groupby('dataset').agg({'initial_correct': 'mean', 'final_correct': 'mean'}).reset_index()
        plt.figure(figsize=(12, 6))
        plt.bar(perf_data['dataset'], perf_data['initial_correct'], label='Initial', color='blue')
        plt.bar(perf_data['dataset'], perf_data['final_correct'] - perf_data['initial_correct'], bottom=perf_data['initial_correct'], label='Improvement', color='green')
        plt.title('Performance Improvement by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xticks(rotation=45)
        plt.savefig('results/performance_improvement_stacked.png')
        plt.close()
        
        # Summary Performance Table
        summary_data = []
        for dataset in df['dataset'].unique():
            d = df[df['dataset'] == dataset]
            summary_data.append({
                'Dataset': dataset,
                'Accuracy (%)': f"{d['accuracy'].mean() * 100:.2f}",
                'Avg Confidence': f"{d['confidence'].mean():.2f}",
                'Correlation': f"{np.corrcoef(d['accuracy'], d['confidence'])[0,1]:.2f}" if len(d) > 1 else "N/A",
                'Sample Size': len(d)
            })
        pd.DataFrame(summary_data).to_csv('results/summary_performance_table.csv', index=False)
        
        # Error Analysis Table
        error_data = []
        for task_type in df['task_type'].unique():
            t_df = df[df['task_type'] == task_type]
            incorrect = t_df[t_df['accuracy'] == 0]
            error_data.append({
                'Task Type': task_type,
                '# Incorrect': len(incorrect),
                'Avg Confidence of Incorrect': f"{incorrect['confidence'].mean():.2f}" if len(incorrect) > 0 else "N/A",
                '% of Total': f"{(len(incorrect) / len(t_df)) * 100:.2f}"
            })
        pd.DataFrame(error_data).to_csv('results/error_analysis_table.csv', index=False)
        
        # Self-Correction Impact Table
        impact_data = df.groupby('dataset').agg({'initial_correct': 'mean', 'final_correct': 'mean'}).reset_index()
        impact_data.columns = ['Dataset', 'Initial Accuracy', 'Refined Accuracy']
        impact_data.to_csv('results/self_correction_impact_table.csv', index=False)
        
        # Confidence Threshold Performance Table
        thresh_data = []
        for thresh in [0.5, 0.7, 0.9]:
            thresh_df = df[df['confidence'] >= thresh]
            thresh_data.append({
                'Threshold': thresh,
                'Accuracy (%)': f"{thresh_df['accuracy'].mean() * 100:.2f}" if len(thresh_df) > 0 else "N/A",
                'Sample Size': len(thresh_df)
            })
        pd.DataFrame(thresh_data).to_csv('results/confidence_threshold_performance_table.csv', index=False)
        
        # Task Type Stats Table
        task_stats = df.groupby('task_type').agg({'accuracy': 'mean', 'confidence': 'mean', 'dataset': 'count'}).reset_index()
        task_stats.columns = ['Task Type', 'Accuracy', 'Avg Confidence', 'Sample Size']
        task_stats.to_csv('results/task_type_stats_table.csv', index=False)
        
        # Model Comparison Table (single model for now)
        model_data = [{'Model': 'Gemini-1.5-Flash', 'Accuracy (%)': f"{df['accuracy'].mean() * 100:.2f}", 'Avg Confidence': f"{df['confidence'].mean():.2f}"}]
        pd.DataFrame(model_data).to_csv('results/model_comparison_table.csv', index=False)

if __name__ == '__main__':
    config = Config()
    engine = CALMEngine(config)
    analytics = Analytics()
    
    all_results = {}
    for dataset_category in config.datasets:
        for dataset_name in config.datasets[dataset_category]:
            print(f"\nProcessing {dataset_name}...")
            results = engine.process_dataset(dataset_name)
            all_results[dataset_name] = results
            report = analytics.generate_report(results, dataset_name)
            print(f"\n{report['dataset']}:")
            print(f"Accuracy: {report['accuracy']:.2%}")
            print(f"Avg Confidence: {report['confidence']:.2f}")
            print(f"Correlation: {report['accuracy_confidence_correlation']:.2f}")
            print(f"Samples: {report['sample_size']}")
            print(f"Consistency: {report['consistency']:.2%}")
            print(f"Improvement: {report['improvement']:.2%}")
            print(f"Regression: {report['regression']:.2%}")
    
    analytics.generate_additional_visuals(all_results, config)
    print("\nEnhanced plots and tables have been saved in the 'results' directory.")